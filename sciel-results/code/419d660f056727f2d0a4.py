"""Direct approach for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

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
    
    # Energy: sum over i<j of log(1/d)
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    energy = np.sum(np.log(1.0 / d_upper))
    
    # Gradient: for point i, grad_i = sum_{j!=i} (pts_i - pts_j) / d_ij^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = pts * sum_inv[:, np.newaxis] - weighted_sum
    
    # Project to tangent space: grad_i -= dot(grad_i, pts_i) * pts_i
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _lbgfs_sphere(pts0, max_iter=500, m=10, time_limit=None, start_time=None, tol=1e-12):
    """L-BFGS optimization on the sphere with projected gradients."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # L-BFGS memory
    s_list = []  # step directions
    y_list = []  # gradient differences
    
    step_size = 0.5
    min_step = 1e-14
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # L-BFGS two-loop recursion to find search direction
        if len(s_list) == 0:
            direction = -grad
        else:
            # Two-loop recursion (correct implementation)
            q = grad.copy()
            rhos = []
            alphas = []
            
            # First loop (reverse)
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
            
            # Initial Hessian approximation
            # Use gamma = 1.0 as a simple choice, or compute from last pair
            if len(s_list) > 0:
                last_s = s_list[-1]
                last_y = y_list[-1]
                sy = np.sum(last_s * last_y)
                if abs(sy) > 1e-16:
                    gamma = np.sum(last_s * last_s) / sy
                    if gamma < 1e-10 or gamma > 1e10:
                        gamma = 1.0
                else:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = q * gamma
            
            # Second loop (forward)
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Ensure direction is on tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Line search with Armijo + curvature condition
        alpha = step_size
        accepted = False
        for _ in range(40):
            # Move along direction and project to sphere
            new_pts = pts + alpha * direction
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            # Guard against zero norms
            norms = np.where(norms < 1e-15, 1.0, norms)
            new_pts = new_pts / norms
            
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Armijo condition
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * direction):
                # Curvature condition (Wolfe)
                if np.sum(new_grad * direction) > 0.9 * np.sum(grad * direction):
                    accepted = True
                    break
            elif new_energy < energy - 1e-12:
                # Accept if we're making progress even without strict Wolfe
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try smaller step
            alpha = max(step_size * 0.01, min_step)
            new_pts = pts + alpha * direction
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-15, 1.0, norms)
            new_pts = new_pts / norms
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


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci with different strengths
    for seed, strength in [(7, 0.05), (99, 0.1), (1234, 0.02)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Coarse optimization with multiple restarts
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 40:  # Leave 40s for polish
            break
        
        remaining = deadline - time.time()
        # Allocate time proportionally, but ensure minimum iterations
        time_alloc = min(remaining * 0.15, 15.0)  # Up to 15s per restart
        time_alloc = max(time_alloc, 2.0)
        
        pts, energy = _lbgfs_sphere(
            init_pts, 
            max_iter=300, 
            m=10,
            time_limit=time_alloc,
            start_time=time.time(),
            tol=1e-10
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Fine-tune best result with tighter tolerance
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - time.time()
        best_pts, best_energy = _lbgfs_sphere(
            best_pts,
            max_iter=1000,
            m=20,
            time_limit=remaining - 5,
            start_time=time.time(),
            tol=1e-13
        )
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining > 10:
            best_pts, best_energy = _lbgfs_sphere(
                best_pts,
                max_iter=500,
                m=30,
                time_limit=remaining - 2,
                start_time=time.time(),
                tol=1e-15
            )
    
    # Ensure valid output
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
        # Check minimum distance
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
