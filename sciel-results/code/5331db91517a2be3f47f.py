"""Improved packing of 300 equal circles in a unit circle.
Uses hexagonal lattice initialization followed by local optimization
with a smooth surrogate (soft-min) and an explicit deadline guard.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 300
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _compute_score(pts):
    """Compute the packing score: min(min pairwise dist/2, min(1 - |c_i|))."""
    n = len(pts)
    # Distance from origin
    r = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - r
    # Pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(boundary))

def _soft_min(values, beta=50.0):
    """Soft minimum via log-sum-exp trick: -log(sum(exp(-beta*x)))/beta."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = np.min(x)
    # Avoid overflow
    exponents = -beta * (x - m)
    # Clip to avoid overflow
    exponents = np.clip(exponents, -500, 500)
    s = np.sum(np.exp(exponents))
    return m - np.log(s) / beta

def _objective_and_grad(pts, beta=30.0):
    """
    Objective to minimize: -soft_min of constraints.
    Constraints: pairwise distances >= 2r, and 1 - |c_i| >= r.
    We want to maximize r, so minimize -r.
    Using soft surrogate: f = -soft_min(constraints), where constraints are
    the values that should be >= 0 for feasibility.
    """
    n = pts.shape[0]
    pts = pts.reshape(n, 2)
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    r_origin = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - r_origin
    
    # Collect all constraint values (should be >= 0)
    # Pairwise: dist/2 (we want dist/2 >= r, so constraint is dist/2 - r)
    # But we don't know r yet. Instead, we maximize the minimum of all these.
    # Let's define the "score" as min of all these values, and maximize it.
    # So we minimize -score.
    
    # All pairwise half-distances
    pair_half = dists / 2.0
    # Exclude diagonal (inf)
    pair_vals = pair_half[np.triu_indices(n, k=1)]
    
    # Boundary values
    bound_vals = boundary
    
    # All constraint values
    all_vals = np.concatenate([pair_vals, bound_vals])
    
    # Soft min
    sm = _soft_min(all_vals, beta=beta)
    
    # Objective: minimize -sm
    obj = -sm
    
    # Gradient
    # d(-sm)/d(pts) = -d(sm)/d(pts)
    # sm = m - log(sum(exp(-beta*(x-m))))/beta where m = min(x)
    # For soft min, the gradient is weighted average of gradients of each x_i
    # weights = exp(-beta*(x_i - m)) / sum(exp(-beta*(x_j - m)))
    
    m = np.min(all_vals)
    exponents = -beta * (all_vals - m)
    exponents = np.clip(exponents, -500, 500)
    weights = np.exp(exponents)
    weights /= np.sum(weights)
    
    # Now compute gradient of each x_i w.r.t. pts
    # For pairwise distances: d(dist_ij)/d(pts_k) 
    # For boundary: d(1 - |c_i|)/d(pts_i)
    
    grad = np.zeros_like(pts)
    
    # Boundary contribution
    # d(boundary_i)/d(pts_i) = -pts_i / |pts_i|
    for i in range(n):
        r_i = np.linalg.norm(pts[i])
        if r_i > 1e-12:
            grad[i] += weights[n + i] * (-pts[i] / r_i)
    
    # Pairwise contribution
    # For each pair (i,j), weight w_{ij}, d(dist_ij/2)/d(pts_i) = (pts_i - pts_j) / (2 * dist_ij)
    # d(dist_ij/2)/d(pts_j) = (pts_j - pts_i) / (2 * dist_ij)
    
    # We need to map pair indices back to (i,j)
    # pair_vals is from triu_indices, so index k corresponds to (i_k, j_k)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for k in range(len(pair_vals)):
        i = i_idx[k]
        j = j_idx[k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = weights[k]
        # d(pair_half_ij)/d(pts_i) = (pts_i - pts_j) / (2*d)
        # d(pair_half_ij)/d(pts_j) = (pts_j - pts_i) / (2*d)
        grad[i] += w * (pts[i] - pts[j]) / (2.0 * d)
        grad[j] += w * (pts[j] - pts[i]) / (2.0 * d)
    
    return obj, -grad  # gradient of -sm is -grad(sm)

def construct():
    start_time = time.time()
    
    # Step 1: Hexagonal lattice initialization
    def hex_init():
        # Try to find a good hexagonal lattice spacing
        best_pts = None
        best_score = -1
        for trial in range(20):
            spacing = 0.055 * (1.0 - trial * 0.002)
            # Generate hexagonal points
            k = int(1.0 / spacing) + 3
            points = []
            for j in range(-k, k + 1):
                y = j * spacing * np.sqrt(3.0) / 2.0
                if abs(y) > 1.0:
                    continue
                offset = 0.5 * spacing if (j % 2) else 0.0
                for i in range(-k, k + 1):
                    x = i * spacing + offset
                    if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) < N:
                continue
            # Select N points closest to origin
            r = np.linalg.norm(points, axis=1)
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        return best_pts, best_score
    
    pts, score = hex_init()
    if pts is None:
        # Fallback: random
        pts = np.random.randn(N, 2) * 0.5
        score = _compute_score(pts)
    
    best_pts = pts.copy()
    best_score = score
    
    # Step 2: Local optimization with smooth surrogate
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        return best_pts
    
    # Try multiple restarts with different betas and initial perturbations
    betas = [20.0, 30.0, 50.0]
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        # Start from current best with small perturbation
        x0 = best_pts + np.random.randn(N, 2) * 0.001
        x0 = x0.reshape(-1)
        
        def obj_func(x):
            p = x.reshape(N, 2)
            val, _ = _objective_and_grad(p, beta=beta)
            return val
        
        def grad_func(x):
            p = x.reshape(N, 2)
            _, g = _objective_and_grad(p, beta=beta)
            return g.reshape(-1)
        
        try:
            result = minimize(
                obj_func, x0, jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-6, 'maxls': 20}
            )
            new_pts = result.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Step 3: Final polish with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 15:
        # Do a final round of optimization with higher beta
        x0 = best_pts.reshape(-1)
        
        def obj_func2(x):
            p = x.reshape(N, 2)
            val, _ = _objective_and_grad(p, beta=100.0)
            return val
        
        def grad_func2(x):
            p = x.reshape(N, 2)
            _, g = _objective_and_grad(p, beta=100.0)
            return g.reshape(-1)
        
        try:
            result = minimize(
                obj_func2, x0, jac=grad_func2,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8, 'maxls': 30}
            )
            new_pts = result.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Ensure validity: clip points slightly inside
    r = np.linalg.norm(best_pts, axis=1)
    mask = r > 0.999
    if np.any(mask):
        best_pts[mask] *= 0.999 / r[mask]
    
    return best_pts
