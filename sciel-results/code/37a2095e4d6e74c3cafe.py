"""Pack 200 equal circles in a unit square using smooth optimization with deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 145.0  # safety margin before 150s kill


def _score(pts):
    """Compute the radius for a given set of points."""
    # Distance to boundary
    boundary = np.minimum(0.5 - np.abs(pts), 0.5)
    min_boundary = boundary.min()
    # Pairwise distances
    from scipy.spatial.distance import pdist
    min_pair = pdist(pts, 'euclidean').min()
    return min(min_boundary, min_pair / 2.0)


def _hex_init(n):
    """Initialize points in a hexagonal packing pattern within the square."""
    pts = []
    # Hexagonal packing: rows offset
    row_spacing = 0.5 * np.sqrt(3)
    col_spacing = 1.0
    # Start from top-left, go down
    y = 0.5 - 0.25
    row = 0
    while y >= -0.5:
        x_offset = (0.5 * col_spacing) if (row % 2 == 1) else 0.0
        x = -0.5 + 0.25 + x_offset
        while x <= 0.5:
            pts.append([x, y])
            x += col_spacing
        y -= row_spacing
        row += 1
    pts = np.array(pts)
    # Trim or pad to N
    if len(pts) >= n:
        # Select n points that are most spread out (simple: take first n)
        pts = pts[:n]
    else:
        # Need more points - add some in remaining space
        while len(pts) < n:
            pts.append([0.0, 0.0])
        pts = np.array(pts)
    return pts


def _random_init(n, seed=42):
    """Random initialization within the square."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(n, 2))


def _smooth_objective(pts, temperature=0.01):
    """
    Smooth surrogate for the min-radius objective using log-sum-exp.
    We want to maximize r = min(min_pair/2, min_boundary).
    Equivalent to minimizing -r.
    Using soft-min: softmin(x) = -1/t * log(sum(exp(-t*x)))
    """
    # Pairwise distances
    from scipy.spatial.distance import pdist
    pair_dists = pdist(pts, 'euclidean')
    pair_half = pair_dists / 2.0
    
    # Boundary distances
    boundary = np.minimum(0.5 - np.abs(pts), 0.5).ravel()
    
    # Combine all constraints
    all_vals = np.concatenate([pair_half, boundary])
    
    # Soft-min via log-sum-exp for numerical stability
    t = 1.0 / temperature
    shifted = -t * all_vals
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -log_sum_exp / t
    
    return -soft_min  # We minimize negative of soft-min = maximize soft-min


def _smooth_grad(pts, temperature=0.01):
    """Gradient of the smooth objective."""
    from scipy.spatial.distance import pdist, squareform
    
    # Pairwise distances
    pair_dists = pdist(pts, 'euclidean')
    pair_half = pair_dists / 2.0
    
    # Boundary distances
    boundary = np.minimum(0.5 - np.abs(pts), 0.5).ravel()
    
    all_vals = np.concatenate([pair_half, boundary])
    
    t = 1.0 / temperature
    shifted = -t * all_vals
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Weights for soft-min
    weights = exp_shifted / sum_exp
    
    # Gradient w.r.t. pair distances
    pair_weights = weights[:len(pair_half)]
    boundary_weights = weights[len(pair_half):]
    
    # Reconstruct pairwise distance matrix
    sq_dist = squareform(pair_dists)
    n = len(pts)
    # Avoid division by zero
    sq_dist[sq_dist == 0] = 1e-10
    
    # Gradient of pair_half[i,j] w.r.t. pts[k]
    # pair_half[i,j] = ||pts[i] - pts[j]|| / 2
    # d/d(pts[k]) = (pts[k] - pts[j]) / (2 * dist) if k==i, (pts[k] - pts[i])/(2*dist) if k==j
    
    grad = np.zeros((n, 2))
    
    # For each pair, add contribution
    # This is O(n^2) but vectorized
    # For pair (i,j): contribution to grad[i] += w[i,j] * (pts[i]-pts[j]) / (2*dist[i,j])
    #                to grad[j] += w[i,j] * (pts[j]-pts[i]) / (2*dist[i,j])
    
    # Vectorized approach
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
    norms = np.sqrt(np.sum(diff**2, axis=2))  # (n, n)
    norms[norms == 0] = 1e-10
    
    # w[i,j] * diff[i,j] / (2 * norms[i,j])
    w_matrix = squareform(pair_weights)
    w_matrix = np.triu(w_matrix, 1)  # Only upper triangle
    
    # Contribution to each point
    # grad[i] = sum_j w[i,j] * (pts[i]-pts[j]) / (2*norms[i,j])
    #         - sum_j w[j,i] * (pts[j]-pts[i]) / (2*norms[i,j])
    
    # First term: sum_j w[i,j] * diff[i,j] / (2*norms[i,j])
    term1 = np.sum(w_matrix[:, :, np.newaxis] * diff / (2 * norms[:, :, np.newaxis]), axis=1)
    
    # Second term: -sum_j w[j,i] * diff[j,i] / (2*norms[j,i])
    # w_matrix is upper triangle, so w[j,i] for j<i is in lower triangle of original
    w_lower = np.triu(w_matrix, 1).T  # This gives w[j,i] for j<i as upper triangle of transposed
    # Actually, let's just use the full weight matrix
    w_full = squareform(pair_weights)
    # diff[j,i] = -diff[i,j], norms[j,i] = norms[i,j]
    # term2[i] = -sum_j w[j,i] * diff[j,i] / (2*norms[j,i])
    #           = -sum_j w[j,i] * (-diff[i,j]) / (2*norms[i,j])
    #           = sum_j w[j,i] * diff[i,j] / (2*norms[i,j])
    term2 = np.sum(w_full.T[:, :, np.newaxis] * diff / (2 * norms[:, :, np.newaxis]), axis=1)
    
    grad_pair = term1 + term2
    
    # Boundary gradient
    # boundary_k = min(0.5 - |x_k|, 0.5 - |y_k|)
    # This is non-smooth at |x| = |y|, but we can use a smooth approximation
    # For now, use subgradient: if 0.5-|x| < 0.5-|y|, grad_x = -sign(x), grad_y = 0
    # else grad_x = 0, grad_y = -sign(y)
    
    abs_x = np.abs(pts[:, 0])
    abs_y = np.abs(pts[:, 1])
    
    # Smooth the min using the boundary weights
    for i in range(n):
        bx = 0.5 - abs_x[i]
        by = 0.5 - abs_y[i]
        # Weight for this point's boundary constraint
        bw = boundary_weights[i]
        # Subgradient of min(bx, by)
        if bx < by:
            gx = -np.sign(pts[i, 0]) if pts[i, 0] != 0 else 0.0
            gy = 0.0
        elif by < bx:
            gx = 0.0
            gy = -np.sign(pts[i, 1]) if pts[i, 1] != 0 else 0.0
        else:
            gx = -0.5 * np.sign(pts[i, 0]) if pts[i, 0] != 0 else 0.0
            gy = -0.5 * np.sign(pts[i, 1]) if pts[i, 1] != 0 else 0.0
        grad[i, 0] += bw * gx
        grad[i, 1] += bw * gy
    
    return -grad_pair - grad  # Negative because we're maximizing


def construct():
    start_time = time.time()
    
    # Try multiple initializations and pick the best
    best_pts = None
    best_score = 0.0
    
    inits = []
    # Hexagonal
    inits.append(_hex_init(N))
    # Random with different seeds
    for seed in [42, 123, 456]:
        inits.append(_random_init(N, seed))
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > DEADLINE - 30:
            break
        
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
        
        # Annealed smooth optimization
        temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
        
        for temp in temperatures:
            if time.time() > DEADLINE - 20:
                break
            
            result = minimize(
                _smooth_objective,
                current_pts.ravel(),
                args=(temp,),
                method='L-BFGS-B',
                jac=lambda x, t: _smooth_grad(x.reshape(-1, 2), t).ravel(),
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _smooth_objective(current_pts, temp):
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score >= current_score - 1e-10:
                    current_pts = new_pts
                    current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
    
    # Final polish with very low temperature
    if time.time() < DEADLINE - 10 and best_pts is not None:
        for temp in [0.001, 0.0005]:
            if time.time() > DEADLINE - 5:
                break
            result = minimize(
                _smooth_objective,
                best_pts.ravel(),
                args=(temp,),
                method='L-BFGS-B',
                jac=lambda x, t: _smooth_grad(x.reshape(-1, 2), t).ravel(),
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success:
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > _score(best_pts):
                    best_pts = new_pts
    
    # Ensure validity: clip to square
    if best_pts is not None:
        best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
