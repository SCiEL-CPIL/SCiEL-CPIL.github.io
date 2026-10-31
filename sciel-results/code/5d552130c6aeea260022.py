import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    return dists

def score(pts):
    """Compute the squared ratio of max to min pairwise distance."""
    dists = pairwise_distances(pts)
    # Exclude self-distances (diagonal)
    mask = ~np.eye(N, dtype=bool)
    valid_dists = dists[mask]
    if len(valid_dists) == 0:
        return np.inf
    min_d = np.min(valid_dists)
    max_d = np.max(valid_dists)
    if min_d < 1e-10:
        return np.inf
    return (max_d / min_d) ** 2

def objective(x):
    """Objective function for optimization."""
    pts = x.reshape(N, D)
    return score(pts)

def hexagonal_initial():
    """Create a hexagonal/triangular lattice arrangement for 16 points."""
    # Arrange in a hexagonal pattern
    # 4 rows: 4, 4, 4, 4 with offset
    pts = []
    spacing = 1.0
    # Row 0: 4 points
    for j in range(4):
        pts.append([j * spacing, 0])
    # Row 1: 4 points, offset by 0.5
    for j in range(4):
        pts.append([j * spacing + 0.5, spacing * np.sqrt(3) / 2])
    # Row 2: 4 points
    for j in range(4):
        pts.append([j * spacing, spacing * np.sqrt(3)])
    # Row 3: 4 points, offset by 0.5
    for j in range(4):
        pts.append([j * spacing + 0.5, spacing * np.sqrt(3) * 1.5])
    return np.array(pts, dtype=float)

def grid_initial():
    """Create a 4x4 grid."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    return np.array(pts, dtype=float)

def circle_initial():
    """Create points on concentric circles."""
    pts = []
    # Center point
    pts.append([0, 0])
    # Ring 1: 6 points
    for k in range(6):
        angle = 2 * np.pi * k / 6
        pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 9 points
    for k in range(9):
        angle = 2 * np.pi * k / 9
        pts.append([2 * np.cos(angle), 2 * np.sin(angle)])
    return np.array(pts, dtype=float)

def random_initial(seed=42):
    """Create random initial points."""
    rng = np.random.RandomState(seed)
    return rng.randn(N, D)

def optimize_from_initial(x0, max_iter=5000, tol=1e-8):
    """Optimize from initial configuration using L-BFGS-B."""
    res = minimize(objective, x0, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
    return res.x, res.fun

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try multiple initial configurations
    initial_configs = [
        grid_initial(),
        hexagonal_initial(),
        circle_initial(),
    ]
    
    # Add some random initializations
    for seed in range(10):
        initial_configs.append(random_initial(seed))
    
    best_pts = None
    best_score = np.inf
    
    for i, x0 in enumerate(initial_configs):
        if time.time() - start_time > deadline - 5:
            break
        
        try:
            x_opt, s = optimize_from_initial(x0)
            if s < best_score:
                best_score = s
                best_pts = x_opt
        except Exception:
            continue
    
    if best_pts is None:
        # Fallback to grid
        best_pts = grid_initial()
    
    return best_pts
