import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_HARD_TIMEOUT = 150.0
_SAFETY_MARGIN = 5.0


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
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
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())


def _energy_and_riemannian_grad(pts):
    """
    Compute logarithmic energy and Riemannian gradient (projected to tangent space).
    pts: (N, 3) array of points on the unit sphere.
    Returns:
        energy: scalar
        riem_grad: (N, 3) array, gradient projected to tangent space.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy using vectorized computation
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dists_sq[iu]
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    d_upper = np.sqrt(d_sq_upper)
    terms = np.log(1.0 / d_upper)
    energy = np.sum(terms)
    
    # Euclidean gradient: grad_eucl[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    # grad_eucl[i] = -sum_j diff[i,j] / dist2[i,j]
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Weighted sum: sum_j (x_j - x_i)/d^2 = sum_j x_j/d^2 - x_i * sum_j 1/d^2
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    weighted_sum_pts = inv_d2 @ pts
    
    eucl_grad = weighted_sum_pts - pts * sum_inv_d2[:, np.newaxis]
    
    # Project to tangent space: grad_riem = eucl_grad - (eucl_grad . x_i) x_i
    dot = np.sum(eucl_grad * pts, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * pts
    
    return energy, riem_grad


def _exponential_map(pts, tangent):
    """
    Exponential map on the sphere: move points along tangent vectors.
    pts: (N, 3) points on sphere
    tangent: (N, 3) tangent vectors
    Returns: new points on sphere
    """
    theta = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    theta_safe = np.where(theta < 1e-12, 1.0, theta)
    sin_theta_over_theta = np.sin(theta_safe) / theta_safe
    cos_theta = np.cos(theta_safe)
    
    new_pts = cos_theta * pts + sin_theta_over_theta * tangent
    return _normalize(new_pts)


def _riemannian_cg(pts_init, time_limit, max_iter=2000, ftol=1e-14, gtol=1e-10):
    """
    Riemannian Conjugate Gradient with Polak-Ribiere update.
    """
    start_time = time.time()
    x = pts_init.copy()
    energy, grad = _energy_and_riemannian_grad(x)
    best_pts = x.copy()
    best_energy = energy
    
    # Initial direction: negative gradient
    d = -grad.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Line search: backtracking
        step = 1.0
        for ls_iter in range(30):
            new_x = _exponential_map(x, step * d)
            new_energy, new_grad = _energy_and_riemannian_grad(new_x)
            
            # Armijo condition
            if new_energy <= energy + 1e-4 * step * np.sum(grad * d):
                break
            step *= 0.5
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_x.copy()
        
        # Check convergence
        grad_norm = np.linalg.norm(new_grad)
        if grad_norm < gtol:
            break
        if abs(energy - new_energy) < ftol:
            break
        
        # Polak-Ribiere update
        grad_dot = np.sum(new_grad * grad)
        beta = max(0.0, grad_dot / (np.sum(grad * grad) + 1e-30))
        d = -new_grad + beta * d
        
        # Project d to tangent space
        dot_d = np.sum(d * x, axis=1, keepdims=True)
        d = d - dot_d * x
        
        # Update
        x = new_x
        energy = new_energy
        grad = new_grad
    
    return best_pts, best_energy


def _lbfgs_sphere(pts_init, time_limit, max_iter=3000, ftol=1e-14, gtol=1e-10):
    """
    L-BFGS-B optimization on the sphere using ambient coordinates.
    Normalizes points after each evaluation.
    """
    start_time = time.time()
    x0 = pts_init.flatten().copy()
    best_pts = pts_init.copy()
    best_energy = _log_energy_fsum(pts_init)
    
    def obj_and_grad(x):
        nonlocal best_pts, best_energy
        if time.time() - start_time > time_limit:
            return float('inf'), np.zeros_like(x)
        
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        energy, grad = _energy_and_riemannian_grad(pts)
        
        # Track best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        return energy, grad.flatten()
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'gtol': gtol,
                'maxfun': max_iter * 10,
                'maxls': 50,
            }
        )
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_pts = _normalize(result.x.reshape(N, 3))
    except Exception:
        pass
    
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


def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts = _normalize(pts)
    return pts


def _fps_fast(n, seed=42, batch_size=500):
    """Fast farthest point sampling using vectorized operations."""
    rng = np.random.default_rng(seed)
    # Start with a random point
    first_pt = rng.standard_normal(3)
    first_pt = first_pt / np.linalg.norm(first_pt)
    pts_arr = first_pt.reshape(1, 3)
    
    for _ in range(n - 1):
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)
        
        # Compute min distance from each candidate to existing points
        min_dists = np.full(batch_size, np.inf)
        chunk = 30
        for i in range(0, len(pts_arr), chunk):
            chunk_pts = pts_arr[i:i+chunk]
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))
        
        best_idx = np.argmax(min_dists)
        pts_arr = np.vstack([pts_arr, candidates[best_idx:best_idx+1]])
    
    return _normalize(pts_arr)


def _basin_hopping(pts_init, time_limit, max_restarts=25):
    """
    Perform basin hopping search with adaptive perturbation.
    Uses Riemannian CG for local optimization (faster than L-BFGS-B).
    """
    start_time = time.time()
    current_pts = pts_init.copy()
    current_energy = _log_energy_fsum(current_pts)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    temp = 0.4
    decay = 0.8
    
    for rh in range(max_restarts):
        elapsed = time.time() - start_time
        time_left = time_limit - elapsed
        if time_left < 5:  # Need time for at least one more optimization
            break
        
        # Adaptive perturbation scale
        perturb_scale = max(0.02, min(0.3, 0.05 * temp + 0.02))
        
        rng = np.random.default_rng(rh + 5000)
        perturbations = rng.standard_normal((N, 3)) * perturb_scale
        
        # Project to tangent space
        dot_prods = np.sum(perturbations * current_pts, axis=1, keepdims=True)
        perturbations -= dot_prods * current_pts
        
        new_pts = current_pts + perturbations
        new_pts = _normalize(new_pts)
        
        # Optimize using Riemannian CG (faster)
        opt_time = min(8.0, time_left - 2.0)
        if opt_time < 3.0:
            break
            
        new_opt_pts, new_energy = _riemannian_cg(new_pts, opt_time, max_iter=1500)
        
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
    inits.append(_fps_fast(N, seed=42))
    
    # Random inits with different seeds
    for s in range(3):
        inits.append(_random_sphere(N, seed=s * 17 + 3))
        
    # Perturbed Fibonacci
    for s in range(2):
        rng = np.random.default_rng(s * 100 + 50)
        scale = 0.03 + 0.02 * s
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Quick local optimization of all inits using Riemannian CG
    candidates = []
    for i, init_pts in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 15:  # Reserve 15s for deeper search
            break
            
        # Allocate time: start with 4s, decrease for later inits
        opt_time = max(2.0, 4.0 - 0.3 * i)
        
        opt_pts, opt_energy = _riemannian_cg(init_pts, opt_time, max_iter=800)
        
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
    
    if remaining_for_bh > 10:
        # Distribute time among top candidates
        bh_time_per_candidate = remaining_for_bh / max(1, len(top_candidates))
        
        for i, cand_pts in enumerate(top_candidates):
            remaining = deadline - time.time()
            if remaining < reserve_time + 3:
                break
                
            bh_time = min(bh_time_per_candidate, remaining - reserve_time - 1)
            if bh_time < 5:
                break
                
            bh_pts, bh_energy = _basin_hopping(cand_pts, bh_time, max_restarts=20)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pts = bh_pts.copy()

    # Phase 3: Final high-precision Polish with Riemannian CG
    time_left = deadline - time.time()
    if time_left > 3:
        polish_time = time_left - 0.5
        final_pts, final_energy = _riemannian_cg(best_pts, polish_time, max_iter=5000, ftol=1e-15, gtol=1e-12)
        
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
