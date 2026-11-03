"""Optimized 50-circle packing in unit square."""
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 50


def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    # Distance to walls
    wall_dists = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    pd = pdist(centers)
    if len(pd) == 0:
        min_pair = np.inf
    else:
        min_pair = np.min(pd)
    
    return min(np.min(wall_dists), min_pair / 2.0)


def _soft_min_radius(centers, beta=50.0):
    """Soft minimum of all constraints (wall and pairwise)."""
    # Wall distances
    wall_dists = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    
    # Pairwise distances / 2
    from scipy.spatial.distance import pdist
    pd = pdist(centers) / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([wall_dists, pd])
    
    # Soft minimum: -log(sum(exp(-beta * d))) / beta
    # For numerical stability
    shifted = all_dists - np.min(all_dists)
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) + (-beta * np.min(all_dists))
    return -log_sum / beta


def _soft_min_radius_grad(centers, beta=50.0):
    """Gradient of soft minimum radius."""
    # Wall distances and their gradients
    # For center i, wall dists are:
    # 0.5 - cx_i, cx_i + 0.5, 0.5 - cy_i, cy_i + 0.5
    cx, cy = centers[:, 0], centers[:, 1]
    
    wall_dists = np.array([
        0.5 - cx,
        cx + 0.5,
        0.5 - cy,
        cy + 0.5
    ])  # shape (4, N)
    
    # Pairwise distances / 2
    from scipy.spatial.distance import pdist
    pd = pdist(centers) / 2.0
    
    all_dists = np.concatenate([wall_dists.flatten(), pd])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # Soft min value
    soft_min = min_d - np.log(sum_exp) / beta
    
    # Gradient with respect to centers
    grad = np.zeros_like(centers)
    
    # Wall gradient contributions
    # For wall dist 0.5 - cx: grad w.r.t. cx is -1
    # For wall dist cx + 0.5: grad w.r.t. cx is +1
    # For wall dist 0.5 - cy: grad w.r.t. cy is -1
    # For wall dist cy + 0.5: grad w.r.t. cy is +1
    
    # The soft min gradient w.r.t. each distance d_k is: -exp(-beta*(d_k - min_d)) / sum_exp * (-1)
    # Wait, let me redo this. soft_min = min_d - log(sum(exp(-beta*(d_k - min_d))))/beta
    # Actually, let's use the standard soft min formula:
    # soft_min = -1/beta * log(sum(exp(-beta * d_k)))
    # d(soft_min)/d(d_k) = exp(-beta * d_k) / sum(exp(-beta * d_j))
    
    # Recompute with full exp (not shifted) for clarity, but use shifted for stability
    # d(soft_min)/d(d_k) = exp(-beta * d_k) / sum(exp(-beta * d_j))
    # Using shifted: = exp(-beta * (d_k - min_d)) / sum(exp(-beta * (d_j - min_d)))
    
    weights = exp_vals / sum_exp  # shape (4N + N(N-1)/2,)
    
    # Wall part
    wall_weights = weights[:4 * N].reshape(4, N)
    # Wall 0: 0.5 - cx, grad cx = -1
    grad[:, 0] += -wall_weights[0]
    # Wall 1: cx + 0.5, grad cx = +1
    grad[:, 0] += wall_weights[1]
    # Wall 2: 0.5 - cy, grad cy = -1
    grad[:, 1] += -wall_weights[2]
    # Wall 3: cy + 0.5, grad cy = +1
    grad[:, 1] += wall_weights[3]
    
    # Pairwise part
    pair_weights = weights[4 * N:]
    # pdist returns distances in order: (0,1), (0,2), ..., (0,N-1), (1,2), ...
    # For pair (i,j) with i<j, dist = |c_i - c_j|/2
    # grad w.r.t. c_i: (c_i - c_j) / |c_i - c_j| * 1/2 * weight
    # grad w.r.t. c_j: (c_j - c_i) / |c_i - c_j| * 1/2 * weight
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = pair_weights[idx]
            if d > 1e-12:
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    unit = diff / dist
                    # d(dist/2)/d(c_i) = unit/2, d(dist/2)/d(c_j) = -unit/2
                    grad[i] += unit * d / 2.0
                    grad[j] += -unit * d / 2.0
            idx += 1
    
    return soft_min, grad


def construct():
    """Construct 50 circle centers in unit square maximizing minimum radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_centers = None
    best_radius = 0.0
    
    # Initial guess: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init_centers = grid[:N].astype(float)
    
    r_init = _compute_radius(init_centers)
    best_centers = init_centers.copy()
    best_radius = r_init
    
    # Multiple restarts with different initial configurations
    rng = np.random.RandomState(42)
    
    n_restarts = 8
    beta_schedule = [30.0, 50.0, 80.0, 120.0]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Generate diverse initial configurations
        if restart == 0:
            centers = init_centers.copy()
        elif restart == 1:
            # Hexagonal packing
            centers = _hex_init(N)
        elif restart == 2:
            # Perturbed grid
            centers = init_centers + rng.normal(0, 0.01, size=(N, 2))
            centers = np.clip(centers, -0.49, 0.49)
        elif restart == 3:
            # Random
            centers = rng.uniform(-0.45, 0.45, size=(N, 2))
        elif restart == 4:
            # Grid with different offset
            m2 = 7
            ticks2 = (np.arange(m2) + rng.uniform(0, 0.5)) / m2 - 0.5
            grid2 = np.array([(x, y) for y in ticks2 for x in ticks2])
            centers = grid2[:N].astype(float)
            centers = np.clip(centers, -0.49, 0.49)
        elif restart == 5:
            # Two-row pattern
            centers = _two_row_init(N)
        elif restart == 6:
            # Perturbed hex
            centers = _hex_init(N) + rng.normal(0, 0.005, size=(N, 2))
            centers = np.clip(centers, -0.49, 0.49)
        else:
            # Random with symmetry
            centers = rng.uniform(-0.45, 0.45, size=(N, 2))
        
        # Optimize with increasing beta
        for beta in beta_schedule:
            if time.time() > deadline - 5.0:
                break
            
            try:
                result = minimize(
                    lambda c: -_soft_min_radius(c, beta),
                    centers,
                    jac=lambda c: -_soft_min_radius_grad(c, beta)[1],
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 2 * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                centers = result.x
            except Exception:
                pass
            
            # Compute actual radius
            r = _compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
        
        # Local refinement with high beta
        if time.time() < deadline - 3.0:
            try:
                result = minimize(
                    lambda c: -_soft_min_radius(c, 200.0),
                    centers,
                    jac=lambda c: -_soft_min_radius_grad(c, 200.0)[1],
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 2 * N,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                centers = result.x
                r = _compute_radius(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers.copy()
            except Exception:
                pass
    
    # Final validation
    if best_centers is None:
        best_centers = init_centers
    
    # Ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers


def _hex_init(n):
    """Hexagonal packing initialization."""
    centers = []
    r = 1.0 / (2 * np.sqrt(3) * np.sqrt(n))  # approximate
    # Use a hex grid
    dx = 1.0 / np.sqrt(n)
    dy = dx * np.sqrt(3) / 2
    
    row = 0
    while len(centers) < n:
        offset = 0 if row % 2 == 0 else dx / 2
        col = 0
        while len(centers) < n:
            x = (col + 0.5) * dx - 0.5 + offset
            y = (row + 0.5) * dy - 0.5
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                centers.append([x, y])
            col += 1
        row += 1
    
    return np.array(centers[:n])


def _two_row_init(n):
    """Two-row staggered initialization."""
    centers = []
    half = n // 2
    extra = n - half
    
    # Top row
    for i in range(half):
        x = (i + 0.5) / half - 0.5
        y = 0.25
        centers.append([x, y])
    
    # Bottom row
    for i in range(extra):
        x = (i + 0.5) / extra - 0.5
        y = -0.25
        centers.append([x, y])
    
    return np.array(centers)
