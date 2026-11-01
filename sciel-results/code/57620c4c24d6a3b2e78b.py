"""Minimal logarithmic energy of 150 points on the sphere - improved optimizer."""
import time
import math
import numpy as np

N = 150


def _log_energy(pts):
    """Compute logarithmic energy using math.fsum for precision."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    valid_mask = d_upper > 1e-12
    if not np.all(valid_mask):
        return float('inf'), np.zeros_like(pts)
        
    terms = [math.log(1.0 / d) for d in d_upper]
    energy = math.fsum(terms)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = pts * sum_inv[:, np.newaxis] - weighted_sum
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _halton_sphere(n):
    """Generate n points on sphere using Halton sequence in 3D."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Map to sphere: use a simple mapping
        # Normalize the 3D point
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        if norm > 1e-15:
            points.append(p / norm)
        else:
            points.append(np.array([1.0, 0.0, 0.0]))
    return np.array(points)


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _exp_map(pts, v):
    """Exponential map on sphere: move points along tangent vectors."""
    # v is tangent to sphere at pts
    # exp_map(p, v) = p*cos(|v|) + (v/|v|)*sin(|v|)
    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    safe_norms = np.where(norms_v < 1e-15, 1.0, norms_v)
    cos_t = np.cos(norms_v)
    sin_t = np.sin(norms_v)
    return pts * cos_t + v * (sin_t / safe_norms)


def _retr_tract(pts, v):
    """Retraction: project pts + v to sphere."""
    return _project_to_sphere(pts + v)


def _lbgfs_sphere(pts0, max_iter=500, m=10, time_limit=None, start_time=None, tol=1e-12):
    """L-BFGS optimization on the sphere with projected gradients."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # L-BFGS memory
    s_list = []
    y_list = []
    
    step_size = 0.5
    min_step = 1e-14
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # L-BFGS two-loop recursion
        if len(s_list) == 0:
            direction = -grad
        else:
            q = grad.copy()
            rhos = []
            alphas = []
            
            # Backward loop
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                si_yi = np.sum(si * yi)
                if abs(si_yi) < 1e-16:
                    rho = 0.0
                else:
                    rho = 1.0 / si_yi
                rhos.append(rho)
                ai = np.sum(si * q) * rho
                alphas.append(ai)
                q -= ai * yi
            
            # Compute gamma from most recent pair
            last_s = s_list[-1]
            last_y = y_list[-1]
            sy = np.sum(last_s * last_y)
            if abs(sy) > 1e-16:
                gamma = np.sum(last_s * last_s) / sy
                if gamma < 1e-10 or gamma > 1e10:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = q * gamma
            
            # Forward loop
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[i]
                alpha_i = alphas[i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Ensure direction is on tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Line search with backtracking
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = _retr_tract(pts, alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * direction):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(step_size * 0.01, min_step)
            new_pts = _retr_tract(pts, alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                accepted = True
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
        
        if not accepted:
            break
        
        # Update L-BFGS memory
        s = alpha * direction
        y = new_grad - grad
        sy = np.sum(s * y)
        
        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)
    
    return pts, energy


def _cg_sphere(pts0, max_iter=500, time_limit=None, start_time=None, tol=1e-12):
    """Conjugate gradient optimization on the sphere using exponential maps."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # CG state
    p = -grad  # initial search direction
    p = p - np.sum(p * pts, axis=1, keepdims=True) * pts  # project to tangent
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search along p using exponential map
        alpha = 0.5
        accepted = False
        for _ in range(50):
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * p):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try smaller step
            alpha = 0.01
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                accepted = True
            else:
                break
        
        if not accepted:
            break
        
        # CG update (Polak-Ribiere)
        grad_diff = new_grad - grad
        pr_coeff = np.sum(new_grad * grad_diff)
        if pr_coeff < 0:
            pr_coeff = 0.0
        
        p = -new_grad + pr_coeff * (p / np.sum(p * p))
        
        # Project to tangent space
        p = p - np.sum(p * new_pts, axis=1, keepdims=True) * new_pts
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
    
    return pts, energy


def _newton_polish(pts0, max_iter=50, time_limit=None, start_time=None, tol=1e-14):
    """Newton polish using Gauss-Newton approximation for final convergence."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    n = len(pts)
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Build approximate Hessian using Gauss-Newton
        # H ≈ J^T J where J is Jacobian of gradient
        # For speed, use a simplified approach: scale gradient by local curvature estimate
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        
        # Approximate Hessian diagonal (curvature estimate)
        inv_d2 = 1.0 / dists_sq
        np.fill_diagonal(inv_d2, 0.0)
        
        # Diagonal scaling for Newton step
        hess_diag = np.sum(inv_d2, axis=1)
        hess_diag = np.maximum(hess_diag, 1e-8)
        
        # Newton step in tangent space
        step = -grad / hess_diag[:, np.newaxis]
        
        # Project to tangent space
        dot_step = np.sum(step * pts, axis=1)
        step -= dot_step[:, np.newaxis] * pts
        
        # Line search
        alpha = 1.0
        accepted = False
        for _ in range(20):
            new_pts = _retr_tract(pts, alpha * step)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            break
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
    
    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Halton sequence
    inits.append(_halton_sphere(N))
    
    # 3. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024, 3456, 7890]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # 4. Perturbed Fibonacci with different strengths
    for seed, strength in [(7, 0.05), (99, 0.1), (1234, 0.02), (5678, 0.08), (2468, 0.04)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    # 5. Perturbed Halton
    for seed, strength in [(31, 0.05), (777, 0.1), (1357, 0.03)]:
        rng = np.random.default_rng(seed)
        pts_hal = _halton_sphere(N)
        pts_pert = pts_hal + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Coarse optimization with multiple restarts
    # Allocate ~80% of time to Phase 1
    phase1_deadline = start_time + (deadline - start_time) * 0.8
    
    for idx, init_pts in enumerate(inits):
        if time.time() > phase1_deadline:
            break
        
        remaining = phase1_deadline - time.time()
        # Give more time to early inits (they're more likely to be good)
        if idx < 3:
            time_alloc = min(remaining * 0.2, 15.0)
        else:
            time_alloc = min(remaining * 0.15, 10.0)
        time_alloc = max(time_alloc, 3.0)
        
        # Use L-BFGS for all (it's faster than CG for this problem)
        pts, energy = _lbgfs_sphere(
            init_pts, 
            max_iter=300, 
            m=15,
            time_limit=time_alloc,
            start_time=time.time(),
            tol=1e-12
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Fine-tune best result with L-BFGS
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - time.time() - 10
        if remaining > 5:
            best_pts, best_energy = _lbgfs_sphere(
                best_pts,
                max_iter=500,
                m=20,
                time_limit=remaining,
                start_time=time.time(),
                tol=1e-14
            )
    
    # Phase 3: Newton polish for final convergence
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            best_pts, best_energy = _newton_polish(
                best_pts,
                max_iter=30,
                time_limit=remaining,
                start_time=time.time(),
                tol=1e-16
            )
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_pts = best_pts / norms
    
    # Verify all points are valid and not too close
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
