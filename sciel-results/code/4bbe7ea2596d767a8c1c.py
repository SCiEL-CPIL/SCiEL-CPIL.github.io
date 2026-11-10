"""Physics-based optimization for packing 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 200
SIDE = 1.0
HALF = 0.5

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    wall_dist = np.minimum.reduce([
        HALF - pts[:, 0],
        HALF + pts[:, 0],
        HALF - pts[:, 1],
        HALF + pts[:, 1]
    ])
    min_wall = np.min(wall_dist)
    
    if min_wall < 0:
        return 0.0
    
    dists = pdist(pts)
    if len(dists) == 0:
        return min_wall
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _soft_min_energy(pts, beta=100.0, r_target=0.035):
    """
    Smooth surrogate for min-distance using log-sum-exp.
    Minimizes -log(sum(exp(-beta*d_i))) which approximates -min(d_i) for large beta.
    """
    pts = np.clip(pts, -HALF + 1e-10, HALF - 1e-10)
    
    # Wall distances
    wall_dists = np.minimum.reduce([
        HALF - pts[:, 0],
        HALF + pts[:, 0],
        HALF - pts[:, 1],
        HALF + pts[:, 1]
    ])
    
    # Pairwise distances
    dists = pdist(pts)
    
    # Combine all constraints into a single list
    all_dists = np.concatenate([wall_dists, dists])
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * d)))
    # For numerical stability, subtract max
    beta_d = -beta * all_dists
    max_beta_d = np.max(beta_d)
    log_sum_exp = max_beta_d + np.log(np.sum(np.exp(beta_d - max_beta_d)))
    
    # Energy is negative of soft min (we want to maximize min distance)
    energy = log_sum_exp / beta
    
    # Add wall penalty for hard constraint
    wall_violation = np.maximum(0, np.abs(pts) - (HALF - r_target))
    wall_energy = np.sum(wall_violation ** 2) * 1000.0
    
    return energy + wall_energy


def _soft_min_grad(pts, beta=100.0, r_target=0.035):
    """Gradient of soft-min energy."""
    pts_clipped = np.clip(pts, -HALF + 1e-10, HALF - 1e-10)
    
    # Wall distances
    wall_dists = np.minimum.reduce([
        HALF - pts_clipped[:, 0],
        HALF + pts_clipped[:, 0],
        HALF - pts_clipped[:, 1],
        HALF + pts_clipped[:, 1]
    ])
    
    # Pairwise distances
    dists = pdist(pts_clipped)
    
    all_dists = np.concatenate([wall_dists, dists])
    
    # Compute weights for soft min
    beta_d = -beta * all_dists
    max_beta_d = np.max(beta_d)
    exp_terms = np.exp(beta_d - max_beta_d)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # These are the softmax weights
    
    # Gradient of soft min w.r.t. distances: -weights
    # We need chain rule for gradient w.r.t. pts
    
    grad = np.zeros_like(pts_clipped)
    
    # Wall gradient contribution
    for i in range(N):
        for j in range(2):
            # Which wall is closest?
            d_left = HALF - pts_clipped[i, j]
            d_right = HALF + pts_clipped[i, j]
            # For x-axis
            if j == 0:
                w_left = weights[i]  # weight for d_left
                w_right = weights[i + N]  # weight for d_right
                # d(d_left)/dx = -1, d(d_right)/dx = 1
                grad[i, j] += -w_left * (-1.0) + w_right * (1.0)
            # For y-axis
            else:
                w_left = weights[i + 2*N]  # weight for d_left (y)
                w_right = weights[i + 3*N]  # weight for d_right (y)
                grad[i, j] += -w_left * (-1.0) + w_right * (1.0)
    
    # Pairwise gradient contribution
    # dists[k] corresponds to pair (i, j) where k = i*N - i*(i+1)/2 + (j-1) for i < j
    # Actually, scipy pdist uses condensed form: index k for pair (i,j) with i<j
    # Let's use squareform for clarity
    dist_matrix = squareform(dists)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # weights for pairwise distances start at index 2*N
    pair_weights = weights[2*N:]
    
    # Map condensed index to (i, j)
    # For condensed vector of length M = N*(N-1)/2
    # Index k corresponds to pair (i, j) where:
    # k = i*(N-1) - i*(i+1)/2 + (j-1) for i < j
    # Simpler: iterate
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            w = pair_weights[k]
            d = dist_matrix[i, j]
            if d > 1e-10:
                diff = pts_clipped[i] - pts_clipped[j]
                # d(dist)/d(pts_i) = diff/d, d(dist)/d(pts_j) = -diff/d
                factor = w / d
                grad[i] += factor * diff
                grad[j] -= factor * diff
            k += 1
    
    # Wall penalty gradient
    wall_violation = np.maximum(0, np.abs(pts_clipped) - (HALF - r_target))
    wall_grad = np.zeros_like(pts_clipped)
    for i in range(N):
        for j in range(2):
            val = pts_clipped[i, j]
            limit = HALF - r_target
            if val > limit:
                wall_grad[i, j] += 2 * 1000.0 * (val - limit)
            elif val < -limit:
                wall_grad[i, j] += 2 * 1000.0 * (val + limit)
    
    return energy, wall_grad + grad


def _energy_and_grad(pts, r_target=0.035, beta=100.0):
    """Energy and gradient using soft-min surrogate."""
    energy = _soft_min_energy(pts, beta, r_target)
    grad = _soft_min_grad(pts, beta, r_target)[1]
    return energy, grad


def _hexagonal_init():
    """Hexagonal lattice initialization."""
    # Hexagonal packing: rows offset by half spacing
    # For N circles, find approximate grid size
    # Hexagonal: each row has ~sqrt(N) circles, spacing = 2r
    # Let's use a rectangular grid with hexagonal offset
    
    # Start with square grid, then perturb to hexagonal
    m = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    n_rows = m
    n_cols = int(np.ceil(N / n_rows))
    
    # Hexagonal spacing
    # In hexagonal packing, horizontal spacing = 2r, vertical = sqrt(3)*r
    # For unit square, we want to fit N circles
    # Let's use a grid and then offset alternating rows
    
    pts = np.zeros((N, 2))
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            # Hexagonal offset
            x = (col + 0.5 * (row % 2)) / n_cols - 0.5
            y = (row + 0.5) / n_rows - 0.5
            # Scale to fit in [-0.5, 0.5] with margin
            x = x * 0.9
            y = y * 0.9
            pts[idx] = [x, y]
            idx += 1
        if idx >= N:
            break
    
    return pts


def _grid_init():
    """Grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init(rng):
    """Random initial configuration inside the square."""
    return rng.uniform(-HALF + 0.02, HALF - 0.5, size=(N, 2))


def _polish_pass(pts, r_target, time_budget=30.0):
    """High-precision polish with tighter tolerances."""
    start = time.time()
    
    try:
        result = minimize(
            _energy_and_grad,
            pts.flatten(),
            args=(r_target, 200.0),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-HALF + 1e-10, HALF - 1e-10)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        pts_opt = result.x.reshape(N, 2)
        pts_opt = np.clip(pts_opt, -HALF + 1e-10, HALF - 1e-10)
        return pts_opt
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_pts = None
    best_radius = 0.0
    
    rng = np.random.default_rng(42)
    
    # Try multiple random restarts with L-BFGS-B
    n_restarts = 12
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = _hexagonal_init()
        elif restart == 1:
            pts = _grid_init()
        else:
            pts = _random_init(rng)
        
        # Optimize with soft-min surrogate
        try:
            result = minimize(
                _energy_and_grad,
                pts.flatten(),
                args=(0.035, 100.0),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-HALF + 1e-10, HALF - 1e-10)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            pts_opt = result.x.reshape(N, 2)
        except Exception:
            pts_opt = pts
        
        # Clip to valid region
        pts_opt = np.clip(pts_opt, -HALF + 1e-10, HALF - 1e-10)
        
        # Compute radius
        r = _compute_radius(pts_opt)
        
        if r > best_radius:
            best_radius = r
            best_pts = pts_opt.copy()
        
        # If we have a good radius, try to refine with a higher target
        if best_radius > 0.035 and time.time() < deadline - 25:
            # Try local refinement with higher r_target and higher beta
            try:
                result2 = minimize(
                    _energy_and_grad,
                    best_pts.flatten(),
                    args=(best_radius, 200.0),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-HALF + 1e-10, HALF - 1e-10)] * (2 * N),
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts_refined = result2.x.reshape(N, 2)
                pts_refined = np.clip(pts_refined, -HALF + 1e-10, HALF - 1e-10)
                r2 = _compute_radius(pts_refined)
                if r2 > best_radius:
                    best_radius = r2
                    best_pts = pts_refined.copy()
            except Exception:
                pass
    
    # Final high-precision polish pass if we have time
    if best_pts is not None and time.time() < deadline - 10:
        try:
            polished = _polish_pass(best_pts, best_radius, time_budget=20.0)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_pts = polished.copy()
        except Exception:
            pass
    
    # Final check
    if best_pts is None:
        best_pts = _hexagonal_init()
    
    return best_pts
