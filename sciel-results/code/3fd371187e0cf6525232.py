import time
import math
import numpy as np

N = 150
_HARD_TIMEOUT = 150.0
_SAFETY_MARGIN = 5.0

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _log_energy_fsum(pts):
    """Compute log energy using math.fsum for exact match with evaluator."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    # Use math.fsum for exact summation as required
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns:
        energy: scalar (using math.fsum)
        grad: (N, 3) array, gradient projected to tangent space.
    """
    n = len(pts)
    # Compute pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal to avoid log(0) and division by zero
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy using math.fsum
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dists_sq[iu]
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    d_upper = np.sqrt(d_sq_upper)
    terms = np.log(1.0 / d_upper)
    energy = math.fsum(terms.tolist())
    
    # Gradient
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    weighted_sum_pts = inv_d2 @ pts
    
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad

def _exponential_map(pts, tangent_vecs):
    """
    Apply exponential map on the sphere.
    For a unit vector x and tangent vector v, the exponential map is:
    exp_x(v) = x*cos(||v||) + (v/||v||)*sin(||v||)
    If ||v|| is very small, exp_x(v) ≈ x + v.
    """
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    cos_vals = np.cos(norms)
    sin_vals = np.sin(norms)
    # Handle small norms: use first-order approximation
    small_mask = norms < 1e-8
    result = pts * cos_vals + (tangent_vecs / safe_norms) * sin_vals
    # For small norms, use x + v
    result[small_mask] = pts[small_mask] + tangent_vecs[small_mask]
    return _normalize(result)

def _riemannian_cg(pts_init, time_limit, max_iter=200):
    """
    Riemannian Conjugate Gradient on the sphere.
    """
    start_time = time.time()
    pts = pts_init.copy()
    energy, grad = _energy_and_grad(pts)
    
    # Initial search direction: negative gradient
    d = -grad.copy()
    beta = 0.0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 0.5:
            break
        
        # Line search: find step size alpha that minimizes energy
        # Use backtracking line search
        alpha = 1.0
        c1 = 1e-4  # Armijo condition
        rho = 0.5
        
        for ls_iter in range(30):
            new_pts = _exponential_map(pts, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * grad^T * d
            if new_energy <= energy + c1 * alpha * np.sum(grad * d):
                break
            alpha *= rho
        
        # Check for convergence
        if np.linalg.norm(new_grad) < 1e-10:
            pts = new_pts
            energy = new_energy
            break
        
        # Compute beta (Polak-Ribiere)
        if beta > 0:
            grad_diff = np.sum(new_grad * (-grad))
            beta = max(0, np.sum(new_grad * (-grad)) / (np.sum(grad * grad) + 1e-12))
        
        d = -new_grad + beta * d
        pts = new_pts
        energy = new_energy
        grad = new_grad
    
    return pts, energy

def _optimize_local(pts_init, time_limit, max_iter=200):
    """
    Run Riemannian CG optimization on the sphere.
    """
    start_time = time.time()
    
    try:
        best_pts, best_energy = _riemannian_cg(pts_init, time_limit, max_iter=max_iter)
        return best_pts, best_energy, True
    except Exception as e:
        return pts_init, _log_energy_fsum(pts_init), False

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

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts = _normalize(pts)
    return pts

def _basin_hopping(pts_init, time_limit, max_restarts=20):
    """
    Perform basin hopping search with adaptive perturbation.
    """
    start_time = time.time()
    current_pts = pts_init.copy()
    current_energy = _log_energy_fsum(current_pts)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    temp = 0.5
    decay = 0.85
    
    for rh in range(max_restarts):
        elapsed = time.time() - start_time
        time_left = time_limit - elapsed
        if time_left < 3:  # Need time for at least one more optimization
            break
        
        # Adaptive perturbation scale based on gradient magnitude
        _, grad = _energy_and_grad(current_pts)
        grad_norm = np.linalg.norm(grad)
        
        # Scale perturbation inversely with gradient norm (larger grad = smaller step)
        base_scale = 0.15
        scale = base_scale / (1.0 + grad_norm / 10.0)
        scale = max(0.01, min(0.5, scale))
        
        rng = np.random.default_rng(rh + 2000)
        perturbations = rng.standard_normal((N, 3)) * scale
        
        # Project to tangent space
        dot_prods = np.sum(perturbations * current_pts, axis=1, keepdims=True)
        perturbations -= dot_prods * current_pts
        
        new_pts = current_pts + perturbations
        new_pts = _normalize(new_pts)
        
        # Optimize
        opt_time = min(8.0, time_left - 1.0)
        if opt_time < 2.0:
            break
            
        new_opt_pts, new_energy, success = _optimize_local(new_pts, opt_time, max_iter=100)
        
        # Metropolis Criterion
        delta_e = new_energy - current_energy
        accept = False
        if delta_e < 0:
            accept = True
        else:
            if temp > 1e-6:
                try:
                    prob = math.exp(-delta_e / temp)
                    if np.random.rand() < prob:
                        accept = True
                except OverflowError:
                    accept = False
        
        if accept:
            current_pts = new_opt_pts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        temp *= decay
        
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + _HARD_TIMEOUT - _SAFETY_MARGIN

    # 1. Generate diverse initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Random inits with different seeds
    for s in range(4):
        inits.append(_random_sphere(N, seed=s * 17 + 3))
        
    # Perturbed Fibonacci
    for s in range(3):
        rng = np.random.default_rng(s * 100 + 50)
        scale = 0.03 + 0.02 * s
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Quick local optimization of all inits
    candidates = []
    for i, init_pts in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 10:  # Reserve 10s for final polish
            break
            
        # Allocate time: start with 4s, decrease for later inits
        opt_time = max(2.0, 4.0 - 0.3 * i)
        
        opt_pts, opt_energy, _ = _optimize_local(init_pts, opt_time, max_iter=50)
        
        candidates.append((opt_energy, opt_pts))
        
    # Sort candidates by energy
    candidates.sort(key=lambda x: x[0])
    
    # Take the top 3 candidates for deeper search
    top_candidates = [c[1] for c in candidates[:3]]
    
    if top_candidates:
        best_pts = top_candidates[0]
        best_energy = _log_energy_fsum(best_pts)
    else:
        best_pts = inits[0]
        best_energy = _log_energy_fsum(best_pts)

    # Phase 2: Basin Hopping on top candidates
    reserve_time = 10.0  # Reserve for final polish
    remaining_for_bh = deadline - time.time() - reserve_time
    
    if remaining_for_bh > 8:
        # Distribute time among top candidates
        bh_time_per_candidate = remaining_for_bh / max(1, len(top_candidates))
        
        for i, cand_pts in enumerate(top_candidates):
            remaining = deadline - time.time()
            if remaining < reserve_time + 3:
                break
                
            bh_time = min(bh_time_per_candidate, remaining - reserve_time - 1)
            if bh_time < 4:
                break
                
            bh_pts, bh_energy = _basin_hopping(cand_pts, bh_time, max_restarts=15)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pts = bh_pts.copy()

    # Phase 3: Final Polish
    time_left = deadline - time.time()
    if time_left > 2:
        polish_time = time_left - 0.5
        final_pts, final_energy, _ = _optimize_local(best_pts, polish_time, max_iter=500)
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts

    # Final check and normalization
    best_pts = _normalize(best_pts)
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci if something went wrong
        best_pts = _fibonacci_sphere(N)
        
    return best_pts
