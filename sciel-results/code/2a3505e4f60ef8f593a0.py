import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
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
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d_sq = np.maximum(dists_sq[iu], 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = np.maximum(dists_sq[iu], 1e-30)
    return -0.5 * np.sum(np.log(d_sq))

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200, m=10):
    """L-BFGS on the sphere using exponential maps and parallel transport."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS history
    s_history = []
    y_history = []
    g_history = []
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        q = grad.copy()
        rho = []
        
        # Two-loop recursion (backward)
        for k in range(len(s_history) - 1, -1, -1):
            s_k = s_history[k]
            y_k = y_history[k]
            g_k = g_history[k]
            
            alpha_k = np.sum(s_k * q) / (np.sum(y_k * y_k) + 1e-30)
            rho.append(alpha_k)
            q -= alpha_k * y_k
        
        r = q
        
        # Two-loop recursion (forward)
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            g_k = g_history[k]
            
            beta_k = np.sum(y_k * r) / (np.sum(y_k * y_k) + 1e-30)
            d += (rho[k] - beta_k) * s_k
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current
        
        # Check if direction is meaningful
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        # Line search (Armijo backtracking with continued search)
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        # If directional derivative is positive, flip direction
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        armijo_satisfied = False
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy > energy + c1 * alpha * directional_deriv:
                break
            
            alpha *= 0.7
            if alpha < 1e-12:
                break
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha_try = 0.01
            for _ in range(5):
                new_pts = _exponential_map(current, alpha_try * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha_try
                    break
                alpha_try *= 0.5
            
            if best_alpha == 0.0:
                break
        
        # Update L-BFGS history
        s_vec = best_pts - current
        y_vec = best_grad - grad
        
        # Project to tangent space
        dot_s = np.sum(s_vec * current, axis=1)
        s_vec -= dot_s[:, np.newaxis] * current
        
        dot_y = np.sum(y_vec * best_pts, axis=1)
        y_vec -= dot_y[:, np.newaxis] * best_pts
        
        s_history.append(s_vec)
        y_history.append(y_vec)
        g_history.append(grad)
        
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)
            g_history.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return current, energy

def _riemannian_cg(pts, max_iter=500, tol=1e-10, time_limit=15.0):
    """Riemannian conjugate gradient on the sphere for log-energy minimization."""
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    def _time_exceeded():
        return time.time() - start_time > time_limit
    
    # Initial gradient
    energy, grad = _energy_and_grad(pts)
    p = -grad  # initial search direction
    
    for it in range(max_iter):
        if _time_exceeded():
            break
        
        # Adaptive Line Search: Backtracking Armijo
        c1 = 1e-4
        alpha = 0.1
        grad_dot_p = np.sum(grad * p)
        if grad_dot_p >= 0:
            # Reset direction if not a descent direction
            p = -grad
            grad_dot_p = np.sum(grad * p)
            if grad_dot_p >= 0:
                break
        
        new_energy = energy
        new_pts = pts
        new_grad = grad
        found = False
        
        for _ in range(20):
            trial_pts = _exponential_map(pts, alpha * p)
            trial_pts = _normalize(trial_pts)
            trial_e, trial_grad = _energy_and_grad(trial_pts)
            
            if trial_e <= energy - c1 * alpha * grad_dot_p:
                new_energy = trial_e
                new_pts = trial_pts
                new_grad = trial_grad
                found = True
                break
            
            alpha *= 0.5
            if _time_exceeded():
                break
        
        if not found:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(new_grad)
        if grad_norm < tol:
            pts = new_pts
            energy = new_energy
            break
        
        # Conjugate gradient update (Polak-Ribiere)
        beta_num = np.sum(new_grad * (new_grad - grad))
        beta_den = np.sum(grad * grad)
        if beta_den > 1e-15:
            beta = beta_num / beta_den
        else:
            beta = 0.0
        beta = max(0.0, min(beta, 1.0))
        
        p = -new_grad + beta * p
        # Project p to tangent space
        dot_p = np.sum(p * new_pts, axis=1, keepdims=True)
        p -= dot_p * new_pts
        
        pts = new_pts
        energy = new_energy
        grad = new_grad
    
    return pts, _log_energy_fast(pts)

def _basin_hopping(pts, time_limit=30.0, max_restarts=8):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization - use CG first, then L-BFGS for polish
    time_left = time_limit - (time.time() - start)
    if time_left > 10:
        # Phase 1: CG for rapid convergence
        cg_time = min(10, time_left - 5)
        current, energy = _riemannian_cg(current, time_limit=cg_time, max_iter=300)
        
        # Phase 2: L-BFGS for fine polish
        time_left = time_limit - (time.time() - start)
        if time_left > 3:
            lbfgs_time = min(8, time_left - 2)
            current, energy = _lbfgs_sphere(current, time_limit=lbfgs_time, max_iter=150, m=12)
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.5
    decay = 0.85
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 8:
            break
        
        # Perturb with adaptive scale - start larger, decay
        rng = np.random.default_rng(rh + 42)
        base_scale = 0.2 * (0.7 ** rh)
        scale = base_scale * (1.0 + 0.3 * math.sin(rh * 1.3))
        perturbations = rng.standard_normal((N, 3)) * scale
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point: CG then L-BFGS
        opt_time = min(15, time_left - 4)
        if opt_time < 5:
            break
        
        # CG phase
        cg_t = min(8, opt_time - 3)
        new_opt_pts, new_energy = _riemannian_cg(new_pts, time_limit=cg_t, max_iter=250)
        
        # L-BFGS phase
        lbfgs_t = min(7, opt_time - 2)
        if lbfgs_t > 2:
            new_opt_pts, new_energy = _lbfgs_sphere(new_opt_pts, time_limit=lbfgs_t, max_iter=100, m=10)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        accept = delta_e < 0
        if not accept and temp > 1e-6:
            try:
                accept = np.random.rand() < math.exp(-delta_e / temp)
            except OverflowError:
                accept = False
        
        if accept:
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        
        temp *= decay
    
    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _farthest_point_sampling(n, seed=0):
    """Generate points using farthest point sampling for good initial spread."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    
    # Start with a random point
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    for i in range(1, n):
        # Sample 200 random candidates
        candidates = rng.standard_normal((200, 3))
        candidates = _normalize(candidates)
        
        best_point = None
        best_min_dist = -1
        
        for cand in candidates:
            dists = np.linalg.norm(pts[:i] - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_point = cand
        
        pts[i] = best_point
    
    return pts

def _deterministic_grid_sphere(n):
    """Generate points using a deterministic spherical grid (icosahedron-based subdivision)."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base_pts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base_pts = _normalize(base_pts)
    
    # Subdivide to get approximately n points
    # For n=150, we can use a few levels of subdivision
    # Level 0: 12 points (icosahedron)
    # Level 1: 42 points
    # Level 2: 162 points (close to 150)
    
    # Use Fibonacci as a good deterministic spread
    return _fibonacci_sphere(n)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initializations - focus on fewer, higher-quality ones
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_farthest_point_sampling(N, seed=0))
    inits.append(_farthest_point_sampling(N, seed=1))
    
    # Perturbed Fibonacci
    for s in range(3):
        rng = np.random.default_rng(s + 100)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.15
        pts = _normalize(pts)
        inits.append(pts)
    
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization with more time per restart
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        time_per_restart = 22.0
        if idx == 0:
            time_per_restart = 25.0
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish with tighter tolerance and more iterations
    if time.time() < deadline - 3:
        polish_time = deadline - time.time() - 1
        try:
            # Multi-stage final polish
            # Stage 1: CG for fine adjustment
            cg_polish_time = min(10, polish_time - 5)
            if cg_polish_time > 3:
                polished_pts, polished_energy = _riemannian_cg(best_pts, time_limit=cg_polish_time, max_iter=500, tol=1e-12)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_pts = polished_pts
            
            # Stage 2: L-BFGS for ultra-fine polish
            lbfgs_polish_time = min(12, polish_time - 3)
            if lbfgs_polish_time > 3:
                final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=lbfgs_polish_time, max_iter=500, m=15)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts
        except Exception:
            pass
            
    return _normalize(best_pts)
