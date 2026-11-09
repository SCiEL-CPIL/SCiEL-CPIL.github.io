"""
Hexagonal seed + L-BFGS-B optimization on soft-min distance.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
HALF = 0.5
SEED = 1234567

def _radius_of(c):
    """Compute packing radius for centers c (N,2)."""
    # Wall clearance
    w = HALF - np.abs(c)
    wall = np.min(w)
    # Pairwise
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = np.min(dist) / 2.0
    return float(min(wall, pair))

def _hexagon_seed(n):
    """
    Generate a hexagonal seed for n circles.
    We aim for a hexagonal close packing arrangement.
    """
    # Estimate optimal radius for hex packing in square
    # Area per circle ~ 2r^2 * sqrt(3)
    # 200 * 2 * r^2 * 1.732 <= 1 => r^2 <= 1 / (200 * 2 * 1.732) ~ 0.00144 => r ~ 0.038
    # Known best is ~0.0366
    
    # Generate a grid of points on a hex lattice
    # Spacing dx = 2*r, dy = r*sqrt(3)
    # We want to fit as many as possible and then trim to n
    
    best_pts = None
    best_r = 0.0
    
    # Try a range of radii to find the one that fits best
    for r in np.linspace(0.035, 0.038, 30):
        dx = 2 * r
        dy = r * np.sqrt(3)
        
        # Number of rows and columns
        # y from -0.5+r to 0.5-r
        y_min = -HALF + r
        y_max = HALF - r
        n_rows = int((y_max - y_min) / dy) + 1
        
        # x spacing
        x_min = -HALF + r
        x_max = HALF - r
        
        pts = []
        for i in range(n_rows):
            y = y_min + i * dy
            # Stagger rows
            x_offset = dx / 2 if i % 2 else 0.0
            x_start = x_min + x_offset
            # Number of columns in this row
            # x from x_start to x_max
            n_cols = int((x_max - x_start) / dx) + 1
            for j in range(n_cols):
                x = x_start + j * dx
                # Check if inside
                if -HALF + r <= x <= HALF - r and -HALF + r <= y <= HALF - r:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= n:
            # We have enough points. 
            # To get the best subset, we should probably keep the ones closest to the center
            # or just take the first n. Taking first n from a centered grid is okay.
            # Better: sort by distance to center? Or just take a contiguous block?
            # For a seed, contiguous is fine.
            # However, if we have more than n, trimming arbitrarily might break symmetry.
            # Let's just take the first n.
            current_pts = pts[:n]
            current_r = _radius_of(current_pts)
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts
        
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:n]
        
    return best_pts

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min of constraints and its gradient.
    x: (2N,) array
    beta: temperature for soft-min
    """
    c = x.reshape(-1, 2)
    n = len(c)
    
    # Constraints:
    # 1. Wall distances: 0.5 - |x_i|, 0.5 - |y_i|
    # 2. Pairwise distances: dist(i,j) / 2
    
    # Wall constraints
    # For each point i, wall distance is min(0.5-|x|, 0.5-|y|)
    # Actually, the constraint is min_k (0.5 - |c_ik|)
    # Let's define the wall clearance for point i as w_i = min(0.5 - |c_ix|, 0.5 - |c_iy|)
    # And the overall wall constraint is min_i w_i
    
    # Pairwise constraints
    # p_ij = dist(i,j) / 2
    # Overall pair constraint is min_{i<j} p_ij
    
    # We want to maximize min( min_i w_i, min_{i<j} p_ij )
    # Equivalent to minimizing -softmin( min_i w_i, min_{i<j} p_ij )
    
    # To make it differentiable, we use a soft-min over ALL constraints.
    # List of constraints:
    # - w_i for each i (4 walls per point, so 4N constraints? Or just the min per point?)
    #   It's better to treat the 4 walls per point as separate constraints in the soft-min
    #   to ensure smoothness.
    #   Constraints:
    #   c_1 = 0.5 - c_0x
    #   c_2 = 0.5 + c_0x
    #   c_3 = 0.5 - c_0y
    #   c_4 = 0.5 + c_0y
    #   ...
    #   c_{4i+1} = 0.5 - c_ix
    #   c_{4i+2} = 0.5 + c_ix
    #   c_{4i+3} = 0.5 - c_iy
    #   c_{4i+4} = 0.5 + c_iy
    
    #   Pairwise:
    #   c_{4N + k} = dist(i,j) / 2 for i<j
    
    # Total constraints M = 4N + N(N-1)/2
    # M = 800 + 19900 = 20700
    
    # Soft-min: -beta * log( sum exp(-c_k / beta) )
    
    # Compute all constraint values
    # Wall constraints
    walls = np.stack([
        HALF - c[:, 0],
        HALF + c[:, 0],
        HALF - c[:, 1],
        HALF + c[:, 1]
    ], axis=1).flatten() # (4N,)
    
    # Pairwise constraints
    # Compute distance matrix
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    # We only need i < j
    iu = np.triu_indices(n, k=1)
    pairs = dist[iu] / 2.0 # (N(N-1)/2,)
    
    # Combine
    constraints = np.concatenate([walls, pairs])
    
    # Soft-min calculation
    # For numerical stability, subtract max constraint
    max_c = np.max(constraints)
    # exp(-(c - max_c) / beta)
    exp_terms = np.exp(-(constraints - max_c) / beta)
    sum_exp = np.sum(exp_terms)
    
    obj = -(max_c - beta * np.log(sum_exp))
    
    # Gradient
    # d obj / d x = d obj / d c_k * d c_k / d x
    # d obj / d c_k = - ( -1/beta * exp(-c_k/beta) / sum_exp ) = exp(-c_k/beta) / (beta * sum_exp)
    # Wait, obj = -softmin = - (max_c - beta * log(sum_exp)) = -max_c + beta * log(sum_exp)
    # d obj / d c_k = beta * (1/sum_exp) * d(sum_exp)/d c_k
    # d(sum_exp)/d c_k = exp(-c_k/beta) * (-1/beta)
    # So d obj / d c_k = - exp(-c_k/beta) / sum_exp
    
    # Let w_k = exp(-c_k/beta) / sum_exp
    # Then d obj / d c_k = -w_k
    
    w = exp_terms / sum_exp
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # c_{4i+1} = 0.5 - c_ix => d/dc_ix = -1
    # c_{4i+2} = 0.5 + c_ix => d/dc_ix = 1
    # c_{4i+3} = 0.5 - c_iy => d/dc_iy = -1
    # c_{4i+4} = 0.5 + c_iy => d/dc_iy = 1
    
    # w[0:4N] corresponds to walls
    w_walls = w[:4*n].reshape(n, 4)
    
    # d obj / d c_ix = - (w_{4i+1} * (-1) + w_{4i+2} * 1) = w_{4i+1} - w_{4i+2}
    # d obj / d c_iy = - (w_{4i+3} * (-1) + w_{4i+4} * 1) = w_{4i+3} - w_{4i+4}
    
    grad[0::2] = w_walls[:, 0] - w_walls[:, 1]
    grad[1::2] = w_walls[:, 2] - w_walls[:, 3]
    
    # Pairwise gradients
    # c_{4N + k} = dist(i,j) / 2
    # d(dist(i,j)/2) / d c_i = (c_i - c_j) / dist(i,j)
    # d(dist(i,j)/2) / d c_j = (c_j - c_i) / dist(i,j)
    
    w_pairs = w[4*n:]
    
    # We need to accumulate gradients for each point
    # Loop over pairs is O(N^2) which is 20k, acceptable.
    # But we can vectorize if we want, or just loop.
    # Let's loop for clarity and correctness.
    
    # iu[0] are indices i, iu[1] are indices j
    idx_i = iu[0]
    idx_j = iu[1]
    
    # dist_pairs = pairs * 2
    dist_pairs = pairs * 2.0
    
    # Avoid division by zero (should not happen with valid input)
    safe_dist = np.where(dist_pairs < 1e-10, 1e-10, dist_pairs)
    
    # Vectorized accumulation is tricky with sparse indices.
    # Let's use np.add.at
    # Direction from i to j: (c_i - c_j) / dist
    # Grad contribution to c_i: -w_k * (c_i - c_j) / dist
    # Grad contribution to c_j: -w_k * (c_j - c_i) / dist
    
    # Actually, d obj / d c_k = -w_k.
    # d c_k / d c_i = (c_i - c_j) / dist
    # So d obj / d c_i += -w_k * (c_i - c_j) / dist
    
    # Let's compute vectors
    diff = c[idx_i] - c[idx_j] # (M_pairs, 2)
    norm = safe_dist[:, None]
    unit_diff = diff / norm # (M_pairs, 2)
    
    # Weights
    weights = -w_pairs[:, None] # (M_pairs, 1)
    
    # Contributions
    contrib_i = weights * unit_diff
    contrib_j = weights * (-unit_diff)
    
    # Accumulate
    np.add.at(grad, idx_i * 2, contrib_i[:, 0])
    np.add.at(grad, idx_i * 2 + 1, contrib_i[:, 1])
    np.add.at(grad, idx_j * 2, contrib_j[:, 0])
    np.add.at(grad, idx_j * 2 + 1, contrib_j[:, 1])
    
    return obj, grad

def _optimize(c0, deadline):
    """
    Run L-BFGS-B optimization.
    """
    x0 = c0.flatten()
    
    # Bounds
    bounds = [(-HALF, HALF)] * (2 * N)
    
    # Initial beta
    beta = 10.0
    
    best_x = x0.copy()
    best_obj = _radius_of(c0)
    
    t_start = time.time()
    
    # We can try a few different betas or just one good one.
    # A smaller beta makes the objective closer to the true min, but sharper.
    # A larger beta is smoother but less accurate.
    # Start with 10, then try 5, then 2.
    
    for b in [10.0, 5.0, 2.0]:
        if time.time() - t_start > deadline - 5:
            break
            
        res = minimize(
            _soft_min_obj_and_grad,
            x0,
            args=(b,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        if res.success or res.status in [0, 1, 2]: # 0: success, 1: maxiter, 2: maxfun
            x_opt = res.x
            c_opt = x_opt.reshape(-1, 2)
            r_opt = _radius_of(c_opt)
            
            if r_opt > best_obj:
                best_obj = r_opt
                best_x = x_opt.copy()
            
            # Update x0 for next iteration (annealing)
            x0 = best_x
    
    return best_x.reshape(-1, 2), best_obj

def construct():
    t_start = time.time()
    deadline = 140.0 # Safety margin
    
    rng = np.random.default_rng(SEED)
    
    # Generate seed
    c0 = _hexagon_seed(N)
    
    # Ensure valid start
    c0 = np.clip(c0, -HALF + 1e-9, HALF - 1e-9)
    
    best_c = c0.copy()
    best_r = _radius_of(best_c)
    
    # Optimization
    rem = deadline - (time.time() - t_start)
    if rem > 10:
        c_opt, r_opt = _optimize(c0, rem)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt
    
    # Final safety check
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
        
    return best_c

if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
