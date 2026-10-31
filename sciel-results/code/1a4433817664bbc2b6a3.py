import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _exact_objective(params):
    """
    Exact objective for local refinement.
    """
    P = params.reshape(-1, 2)
    return _score(P)

def _local_optimize(P_init, maxiter=1000):
    """
    Run Nelder-Mead on a specific configuration.
    Returns the best P and score found.
    """
    x0 = P_init.flatten()
    
    # Nelder-Mead options
    res = minimize(
        _exact_objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
    )
    
    P_opt = res.x.reshape(-1, 2)
    score = _score(P_opt)
    
    return P_opt, score

def _get_diverse_starts():
    """
    Generate a list of diverse initial configurations.
    """
    starts = []
    rng = np.random.default_rng(42)
    
    # 1. Hexagonal Lattice (best known packing structure)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 9.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points**2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    hex_points = hex_points[sorted_indices]
    starts.append(hex_points[:16])
    
    # Jittered Hex
    jittered_hex = hex_points[:16] + rng.normal(0, 0.1, (16, 2))
    starts.append(jittered_hex)
    
    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    starts.append(grid)
    starts.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # 3. Concentric Circles
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([1.0 * np.cos(ang), 1.0 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([0.5 * np.cos(ang), 0.5 * np.sin(ang)])
    starts.append(np.array(pts_circ))
    
    # 4. Random Uniform
    for _ in range(5):
        starts.append(rng.uniform(-1, 1, (16, 2)))
        
    # 5. Random on Ring
    for _ in range(5):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0.5, 1.5, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        starts.append(pts)

    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    starts = _get_diverse_starts()
    rng = np.random.default_rng(123) # Separate rng for random restarts
    
    best_score = float('inf')
    best_P = None
    
    # Phase 1: Local optimization from diverse starts
    for P0 in starts:
        if time.time() > deadline - 20:
            break
        try:
            P_opt, score = _local_optimize(P0, maxiter=500)
            if score < best_score:
                best_score = score
                best_P = P_opt
        except Exception:
            pass

    # Phase 1b: Additional random restarts if time permits
    while time.time() < deadline - 20:
        P0 = rng.uniform(-2, 2, (16, 2))
        try:
            # Use shorter iterations for random restarts to fit more
            P_opt, score = _local_optimize(P0, maxiter=200)
            if score < best_score:
                best_score = score
                best_P = P_opt
        except Exception:
            pass

    # Phase 2: Polish the best result with a longer optimization
    if best_P is not None and time.time() < deadline - 5:
        try:
            P_final, score_final = _local_optimize(best_P, maxiter=2000)
            if score_final < best_score:
                best_score = score_final
                best_P = P_final
        except Exception:
            pass

    if best_P is None:
        # Fallback to first heuristic
        best_P = starts[0]
        
    return best_P

# EVOLVE-BLOCK-END
