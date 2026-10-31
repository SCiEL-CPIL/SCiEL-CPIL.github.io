"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_dists_sq(pts):
    """Compute all pairwise squared distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(n, 1)
    return dists_sq[iu]

def _score_normalized(pts):
    """
    Objective function for optimization.
    Minimizes (max_dist / min_dist)^2.
    By scaling points such that min_dist = 1, we minimize max_dist^2.
    This removes the scale invariance issue and improves numerical stability.
    """
    d_sq = _pairwise_dists_sq(pts)
    min_d_sq = np.min(d_sq)
    
    # If points are too close, penalize heavily
    if min_d_sq < 1e-10:
        return 1e10
        
    # Scale factor to make min_dist = 1
    # min_dist = sqrt(min_d_sq)
    # scaled_pts = pts / min_dist
    # new_max_dist_sq = max_d_sq / min_d_sq
    # Objective is (max_d_sq / min_d_sq)
    return np.max(d_sq) / min_d_sq

def _init_hexagonal_cluster():
    """
    Create a hexagonal cluster of 16 points.
    """
    pts = []
    # Generate a large hex grid
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    
    # Calculate distances to center and pick the 16 closest
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    return pts[idx]

def _init_rings():
    """Two rings: 8 inner, 8 outer."""
    inner_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi/8
    r_in = 1.0
    r_out = 1.8 
    inner = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    outer = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    return np.vstack([inner, outer])

def _init_grid():
    """4x4 grid."""
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts

def _init_random(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-2, 2, size=(N, D))

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_pts = None
    best_score = np.inf
    
    # Initial candidates
    candidates = [
        _init_hexagonal_cluster(),
        _init_rings(),
        _init_grid(),
        _init_random(1),
        _init_random(2),
        _init_random(3)
    ]
    
    # Score initial candidates using the normalized objective (which equals the true score)
    for pts in candidates:
        sc = _score_normalized(pts)
        if sc < best_score:
            best_score = sc
            best_pts = pts.copy()
            
    # Optimization Loop
    # Use Nelder-Mead on the normalized objective.
    # This is robust to the non-smoothness of min/max and handles scale invariance well.
    
    for pts_init in candidates:
        if time.time() > deadline - 10:
            break
            
        x0 = pts_init.flatten()
        
        try:
            res = minimize(
                lambda x: _score_normalized(x.reshape(N, D)),
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            pts_opt = res.x.reshape(N, D)
            sc_opt = _score_normalized(pts_opt)
            
            if sc_opt < best_score:
                best_score = sc_opt
                best_pts = pts_opt.copy()
                
        except Exception:
            pass

    # Final check for distinctness
    if best_pts is None:
        best_pts = _init_hexagonal_cluster()
        
    # Ensure points are distinct
    d_check = np.min(_pairwise_dists_sq(best_pts))
    if d_check < 1e-8:
        best_pts = _init_hexagonal_cluster()

    return best_pts
# EVOLVE-BLOCK-END
