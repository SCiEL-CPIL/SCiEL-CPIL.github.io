"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from math import cos, sin, pi

def _score(points):
    """Compute Delta / n^n for 8 points, where Delta is product of ordered pair distances
    after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=float)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists / diam * 2.0
    # Product of ordered pair distances (exclude self-distances which are 0)
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = scaled_dists[mask]
    # Delta = product of all ordered pair distances
    # Use log to avoid underflow
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    return np.exp(log_score)

def _neg_log_score(points):
    """Negative log-score for optimization. Returns (value, gradient)."""
    pts = np.array(points, dtype=float).reshape(-1, 2)
    n = len(pts)
    
    # Pairwise differences and distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0)
    mask = ~np.eye(n, dtype=bool)
    dists_masked = dists[mask]
    if np.min(dists_masked) < 1e-12:
        return 1e10, np.zeros_like(pts.flatten())
        
    # Log of product of distances (ordered)
    # Note: The product is over all i != j. 
    # log(prod_{i!=j} d_ij) = sum_{i!=j} log(d_ij)
    # Since d_ij = d_ji, this is 2 * sum_{i<j} log(d_ij)
    # But let's just sum over the masked matrix to be safe and consistent with _score
    log_prod = np.sum(np.log(dists_masked))
    
    # Diameter
    D = np.max(dists)
    
    # Objective: log_prod - n(n-1) * log(D)
    # Wait, _score computes:
    # log_delta = sum(log(scaled_dists)) where scaled = dists/D * 2
    # log_delta = sum(log(dists) - log(D) + log(2))
    #            = sum(log(dists)) - n(n-1)log(D) + n(n-1)log(2)
    # score = exp(log_delta - n log n)
    # neg_log_score = - (log_delta - n log n)
    #               = - log_delta + n log n
    #               = - [sum(log(dists)) - n(n-1)log(D) + const] + const'
    #               = - sum(log(dists)) + n(n-1)log(D) + const''
    
    val = -log_prod + n * (n - 1) * np.log(D)
    
    # Gradient
    # grad log_prod = sum_{j != i} (p_i - p_j) / d_ij^2  (for each coordinate)
    # Let's compute grad of -log_prod + n(n-1) log D
    # grad (-log_prod) = - sum_{j != i} (p_i - p_j) / d_ij^2
    # grad (n(n-1) log D) = n(n-1) / D * grad D
    
    # grad D is non-smooth. We approximate it using the pair achieving the max,
    # or a soft max. For L-BFGS-B, numerical gradients might be more robust 
    # if we don't provide an analytic gradient, or we can provide the analytic 
    # gradient for the smooth part and rely on the optimizer's finite difference 
    # for the kink, or just use finite differences for the whole thing.
    # Given the complexity of the max gradient, let's use finite differences 
    # in the optimizer call, or implement a robust analytic gradient.
    
    # Let's try to compute analytic gradient for the smooth part and handle D carefully.
    # Actually, providing a gradient that is discontinuous at the max pair change 
    # can hurt BFGS. 
    # Strategy: Use finite differences for the global search (DE) and 
    # for local polish, use L-BFGS-B with analytic gradient for the 
    # "sum log d" part and a subgradient for the "max d" part.
    
    # However, to keep it simple and robust for the 120s limit, 
    # let's just return the value and let scipy use numerical gradients 
    # if we return None for grad, or provide a good approximation.
    # The current code provides a gradient. Let's stick to a reliable one.
    
    # Analytic gradient for sum_{i!=j} log(d_ij):
    # d/dp_k sum_{i!=j} log(d_ij) = sum_{j!=k} (p_k - p_j) / d_kj^2
    
    dists_inv_sq = np.zeros_like(dists)
    np.fill_diagonal(dists, 1.0) # avoid div by 0 in calc
    dists_inv_sq = 1.0 / (dists * dists)
    np.fill_diagonal(dists_inv_sq, 0.0)
    
    # grad_log_prod[k] = sum_{j} (p_k - p_j) / d_kj^2
    # diff is p_i - p_j. So diff[k, j] = p_k - p_j.
    # We want sum_j dists_inv_sq[k, j] * diff[k, j]
    grad_log_prod = np.einsum('ij,ijk->ik', dists_inv_sq, diff)
    
    # grad of -log_prod is -grad_log_prod
    
    # grad of n(n-1) log D
    # D = max d_ij
    # Let (a, b) be the pair with max distance.
    # grad D = (p_a - p_b) / D for p_a, and -(p_a - p_b) / D for p_b.
    # We find the argmax.
    max_idx = np.unravel_index(np.argmax(dists), dists.shape)
    i, j = max_idx
    if dists[i, j] < 1e-15:
        grad_D = np.zeros_like(pts)
    else:
        vec = (pts[i] - pts[j]) / dists[i, j]
        grad_D = np.zeros_like(pts)
        grad_D[i] = vec
        grad_D[j] = -vec
        
    grad_penalty = n * (n - 1) * (1.0 / D) * grad_D
    
    total_grad = -grad_log_prod + grad_penalty
    
    # Flatten gradient to match input shape
    return val, total_grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    best_score = 0.0
    best_points = None

    def evaluate_and_update(points):
        nonlocal best_score, best_points
        # Check distinctness
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-8:
            return
        
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # 1. Baseline: Regular Octagon
    octagon = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    evaluate_and_update(octagon)

    # 2. Global Search: Differential Evolution
    # Bounds: [-1.5, 1.5] for x and y. 
    # The diameter scaling makes the absolute scale irrelevant, 
    # but points too far apart might cause numerical issues or 
    # just be equivalent to smaller ones. 
    # Actually, since we scale to diameter 2, any non-collinear set works.
    # Bounds of [-1, 1] should be sufficient to capture the shape.
    bounds = [(-1.5, 1.5)] * 16
    
    def de_objective(x_flat):
        if time.time() > deadline:
            return 1e10
        pts = x_flat.reshape(8, 2)
        # Quick distinctness check
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-8:
            return 1e10
        val, _ = _neg_log_score(pts)
        return val

    # Run DE
    # popsize=10, maxiter=100 is a reasonable start. 
    # We can increase if time permits.
    try:
        result = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-6,
            polish=False # We will polish manually for better control
        )
        if time.time() < deadline - 5:
            evaluate_and_update(result.x.reshape(8, 2))
    except Exception:
        pass

    # 3. Local Polish on the best found point (and the octagon)
    # Use L-BFGS-B which is robust and handles the gradient well.
    candidates = []
    if best_points is not None:
        candidates.append(best_points.flatten())
    candidates.append(octagon.flatten())
    
    # Also add a few random perturbations of the octagon to help escape if DE failed
    for _ in range(5):
        if time.time() > deadline - 10: break
        noise = np.random.normal(0, 0.1, 16)
        candidates.append(octagon.flatten() + noise)

    for x0 in candidates:
        if time.time() > deadline - 5: break
        
        try:
            # L-BFGS-B with analytic gradient
            res = minimize(
                _neg_log_score, 
                x0, 
                jac=True,
                method='L-BFGS-B', 
                options={'gtol': 1e-10, 'maxiter': 200, 'maxfun': 5000}
            )
            if res.success or res.fun < 1e9:
                p_opt = res.x.reshape(8, 2)
                evaluate_and_update(p_opt)
        except Exception:
            pass

    # 4. Fallback
    if best_points is None:
        best_points = octagon

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
