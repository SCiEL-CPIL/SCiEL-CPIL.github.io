import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_HARD_TIMEOUT = 150.0
_SAFETY_MARGIN = 5.0  # Seconds to leave before hard timeout

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns:
        energy: scalar
        grad: (N, 3) array, gradient projected to tangent space.
    """
    n = len(pts)
    # Compute pairwise squared distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal to avoid log(0) and division by zero
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # d_ij = sqrt(dists_sq_ij)
    # log(d_ij) = 0.5 * log(dists_sq_ij)
    # So energy = -0.5 * sum_{i<j} log(dists_sq_ij)
    
    # We only need the upper triangle for energy
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dists_sq[iu]
    # Clip to avoid log(0) if points are extremely close (should not happen if normalized and distinct)
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    # Gradient
    # E = sum_{i<j} -log(||x_i - x_j||)
    # dE/dx_i = sum_{j != i} - (1/||x_i - x_j||^2) * (x_i - x_j)
    # Let A be the matrix where A[i,j] = 1/||x_i - x_j||^2 for i!=j, 0 for i==j
    # Then grad_i = sum_{j != i} -A[i,j] * (x_i - x_j)
    #             = -x_i * sum_{j != i} A[i,j] + sum_{j != i} A[i,j] * x_j
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # sum_inv_d2[i] = sum_{j != i} A[i,j]
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    # weighted_sum_pts[i] = sum_{j != i} A[i,j] * x_j
    # This is a matrix multiplication: inv_d2 @ pts
    weighted_sum_pts = inv_d2 @ pts
    
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space: grad - (grad . x_i) x_i
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad

def _objective(x_flat):
    """
    Objective function for scipy.optimize.minimize.
    x_flat: (3N,) vector of flattened points.
    Returns:
        energy
        gradient (flat)
    """
    pts = x_flat.reshape(N, 3)
    # Normalize to ensure we are on the sphere (scipy might step slightly off)
    pts = _normalize(pts)
    
    energy, grad = _energy_and_grad(pts)
    
    # Flatten gradient
    grad_flat = grad.flatten()
    
    return energy, grad_flat

def _optimize_local(pts_init, time_limit, max_iter=500):
    """
    Run L-BFGS-B optimization on the sphere.
    """
    start_time = time.time()
    x0 = pts_init.flatten()
    
    # We use a wrapper to enforce normalization inside the objective,
    # but we also want to ensure the initial guess is normalized.
    # SciPy's L-BFGS-B doesn't support manifold constraints natively,
    # so we rely on the projection in _objective.
    
    # To improve convergence, we can use a custom line search or just rely on default.
    # Default L-BFGS-B is robust.
    
    # We need to be careful with the time limit inside the optimizer.
    # SciPy doesn't have a direct time limit, so we run it and hope it finishes,
    # or we use a callback to stop if time runs out.
    
    stop_flag = [False]
    
    def callback(xk):
        if time.time() - start_time > time_limit - 1.0:
            stop_flag[0] = True
            return True  # Stop optimization
        return False
    
    try:
        res = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'callback': callback
            }
        )
        
        # Normalize the result
        best_pts = _normalize(res.x.reshape(N, 3))
        best_energy = _log_energy_fast(best_pts)
        
        return best_pts, best_energy, res.success
        
    except Exception as e:
        # If optimization fails, return the best we have or the init
        return pts_init, _log_energy_fast(pts_init), False

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

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

def _basin_hopping(pts_init, time_limit, max_restarts=10):
    """
    Perform basin hopping search.
    """
    start_time = time.time()
    current_pts = pts_init.copy()
    current_energy = _log_energy_fast(current_pts)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    temp = 1.0
    decay = 0.8
    
    for rh in range(max_restarts):
        elapsed = time.time() - start_time
        time_left = time_limit - elapsed
        if time_left < 5:  # Need time for at least one more optimization
            break
        
        # Perturb
        rng = np.random.default_rng(rh + 1000)
        scale = 0.1 * (0.9 ** rh)
        perturbations = rng.standard_normal((N, 3)) * scale
        
        # Project to tangent space
        dot_prods = np.sum(perturbations * current_pts, axis=1, keepdims=True)
        perturbations -= dot_prods * current_pts
        
        new_pts = current_pts + perturbations
        new_pts = _normalize(new_pts)
        
        # Optimize
        # Allocate time for this optimization: min(10s, remaining time - 2s)
        opt_time = min(10.0, time_left - 2.0)
        if opt_time < 2.0:
            break
            
        new_opt_pts, new_energy, success = _optimize_local(new_pts, opt_time, max_iter=100)
        
        # Metropolis Criterion
        delta_e = new_energy - current_energy
        accept = False
        if delta_e < 0:
            accept = True
        else:
            # Accept with probability exp(-delta_e / temp)
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
    
    # Random inits
    for s in range(5):
        inits.append(_random_sphere(N, seed=s * 10 + 5))
        
    # Perturbed Fibonacci
    for s in range(3):
        rng = np.random.default_rng(s * 100 + 50)
        scale = 0.05 + 0.02 * s
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)

    best_pts = None
    best_energy = float('inf')
    
    # 2. Basin Hopping for each initialization
    # We have ~140s. Let's allocate time for each init.
    # There are 10 inits. 14s each?
    # Or we can do a quick pass first, then refine the best ones.
    
    # Strategy: 
    # Phase 1: Quick optimization of all inits to find a good candidate.
    # Phase 2: Basin hopping on the top candidates.
    # Phase 3: Final polish.
    
    # Phase 1: Quick local optimization (max 10s each)
    candidates = []
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 15:  # Reserve 15s for final polish
            break
            
        # Allocate time for this quick optimization
        # We want to finish all inits in the first half of the budget?
        # Let's say we spend 5s per init for Phase 1.
        opt_time = 5.0
        
        opt_pts, opt_energy, _ = _optimize_local(init_pts, opt_time, max_iter=50)
        
        candidates.append((opt_energy, opt_pts))
        
    # Sort candidates by energy
    candidates.sort(key=lambda x: x[0])
    
    # Take the top 3 candidates for deeper search
    top_candidates = [c[1] for c in candidates[:3]]
    
    if top_candidates:
        best_pts = top_candidates[0]
        best_energy = _log_energy_fast(best_pts)
    else:
        best_pts = inits[0]
        best_energy = _log_energy_fast(best_pts)

    # Phase 2: Basin Hopping on top candidates
    # We have remaining time. Let's distribute it among the top candidates.
    # Say we have 3 candidates. We give each 1/3 of the remaining time (minus reserve).
    
    reserve_time = 15.0  # Reserve for final polish
    remaining_for_bh = deadline - time.time() - reserve_time
    
    if remaining_for_bh > 10:
        # Distribute time among top candidates
        # We can also just run basin hopping on the single best one if time is tight.
        # Let's run BH on the best one, and if time permits, on the second.
        
        bh_time_per_candidate = remaining_for_bh / max(1, len(top_candidates))
        
        for i, cand_pts in enumerate(top_candidates):
            elapsed = time.time() - start_time
            remaining = deadline - time.time()
            if remaining < reserve_time + 5:
                break
                
            bh_time = min(bh_time_per_candidate, remaining - reserve_time - 2)
            if bh_time < 5:
                break
                
            bh_pts, bh_energy = _basin_hopping(cand_pts, bh_time, max_restarts=5)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pts = bh_pts.copy()

    # Phase 3: Final Polish
    # Use remaining time for a high-precision L-BFGS run on the best candidate.
    time_left = deadline - time.time()
    if time_left > 2:
        polish_time = time_left - 1.0
        final_pts, final_energy, _ = _optimize_local(best_pts, polish_time, max_iter=1000)
        
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
