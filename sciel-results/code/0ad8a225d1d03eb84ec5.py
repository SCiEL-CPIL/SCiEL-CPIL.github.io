"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _score(pts):
    """Compute the squared ratio of max to min pairwise distance."""
    n = len(pts)
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    # Exclude self-distances
    np.fill_diagonal(sq_dists, np.inf)
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    if min_sq <= 0:
        return 1e10
    return (max_sq / min_sq) ** 2

def _score_from_dists(sq_dists):
    """Score from precomputed squared distances (diagonal already inf)."""
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    if min_sq <= 0:
        return 1e10
    return (max_sq / min_sq) ** 2

def _make_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Triangular lattice cluster
    # Take a hexagonal/triangular lattice and pick 16 points
    # Triangular lattice: points at (i + j/2, j * sqrt(3)/2)
    pts = []
    for j in range(5):
        for i in range(5):
            x = i + j / 2.0
            y = j * np.sqrt(3) / 2.0
            pts.append([x, y])
    # Pick 16 points that form a compact cluster - use the first 16 in a spiral-like order
    # Actually, let's just take a 4x4 block of the triangular lattice
    tri_pts = []
    for j in range(4):
        for i in range(4):
            x = i + j / 2.0
            y = j * np.sqrt(3) / 2.0
            tri_pts.append([x, y])
    configs.append(np.array(tri_pts[:16], dtype=float))
    
    # 2. Two concentric rings
    # 8 points on inner ring, 8 on outer ring
    r1 = 1.0
    r2 = 2.0
    pts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8  # offset by half step
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    configs.append(np.array(pts, dtype=float))
    
    # 3. Three concentric rings: 1 center + 6 inner + 9 outer
    pts = [[0, 0]]
    r1 = 1.0
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 2.0
    for i in range(9):
        angle = 2 * np.pi * i / 9
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    configs.append(np.array(pts, dtype=float))
    
    # 4. Rectangular grid 4x4, slightly perturbed
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([float(i), float(j)])
    configs.append(np.array(grid_pts, dtype=float))
    
    # 5. Regular hexagon with internal points
    # 6 vertices of regular hexagon + 6 midpoints of edges + 4 interior
    r = 1.0
    pts = []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    # Add points at radius 0.5 at 6 angles offset
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Add 4 more interior points
    for i in range(4):
        angle = 2 * np.pi * i / 4
        pts.append([0.3 * np.cos(angle), 0.3 * np.sin(angle)])
    configs.append(np.array(pts[:16], dtype=float))
    
    # 6. Random perturbation of grid
    rng = np.random.RandomState(42)
    base = np.array([[i % 4, i // 4] for i in range(16)], dtype=float)
    perturbed = base + rng.randn(16, 2) * 0.3
    configs.append(perturbed)
    
    # 7. Another triangular lattice arrangement
    # Use a different subset of triangular lattice
    tri_pts2 = []
    for j in range(4):
        for i in range(4):
            x = i + j / 2.0
            y = j * np.sqrt(3) / 2.0
            tri_pts2.append([x, y])
    # Shift to center
    center = np.mean(tri_pts2, axis=0)
    tri_pts2 = np.array(tri_pts2) - center
    configs.append(tri_pts2)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_score = np.inf
    
    # Get initial configurations
    configs = _make_initial_configs()
    
    # Evaluate all initial configs
    for cfg in configs:
        if time.time() > deadline:
            break
        s = _score(cfg)
        if s < best_score:
            best_score = s
            best_pts = cfg.copy()
    
    # Local optimization using Powell's method
    for cfg in configs:
        if time.time() > deadline:
            break
        
        # Objective function for minimization
        def objective(x):
            pts = x.reshape(-1, 2)
            return _score(pts)
        
        try:
            result = minimize(
                objective,
                cfg.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < best_score:
                s = result.fun
                if s < best_score:
                    best_score = s
                    best_pts = result.x.reshape(-1, 2).copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead for some configs
        if time.time() > deadline:
            break
        try:
            result2 = minimize(
                objective,
                cfg.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_pts = result2.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # If we have time, try more random restarts
    if time.time() < deadline - 10:
        rng = np.random.RandomState(123)
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Random perturbation of best solution
            if best_pts is not None:
                noise = rng.randn(16, 2) * 0.5
                cfg = best_pts + noise
            else:
                cfg = rng.randn(16, 2)
            
            def objective(x):
                pts = x.reshape(-1, 2)
                return _score(pts)
            
            try:
                result = minimize(
                    objective,
                    cfg.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_pts = result.x.reshape(-1, 2).copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to grid
        best_pts = np.array([[i % 4, i // 4] for i in range(16)], dtype=float)
    
    return best_pts
# EVOLVE-BLOCK-END
