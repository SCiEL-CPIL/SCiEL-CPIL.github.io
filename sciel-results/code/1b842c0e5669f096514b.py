import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective(x):
    points = x.reshape(-1, D)
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals <= 0):
        return 1e10
    
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    return d_max_sq / d_min_sq

def construct():
    # --- Seed generation ---
    
    # Seed 1: 4x4 Grid
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    seed1 = np.array(grid, dtype=float)
    seed1 -= np.mean(seed1, axis=0)
    
    # Seed 2: Hexagonal subset (16 closest to center in radius 2 hex)
    hex_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= 2:
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                hex_candidates.append((x, y))
    hex_arr = np.array(hex_candidates)
    d_orig = np.sum(hex_arr**2, axis=1)
    sorted_indices = np.argsort(d_orig)
    seed2 = hex_arr[sorted_indices[:N]]
    seed2 -= np.mean(seed2, axis=0)

    # Seed 3: Random
    np.random.seed(42)
    seed3 = np.random.rand(N, D) * 2 - 1
    seed3 -= np.mean(seed3, axis=0)

    # Seed 4: 2x8 Grid
    grid2 = list(itertools.product(range(2), range(8)))[:N]
    seed4 = np.array(grid2, dtype=float)
    seed4 -= np.mean(seed4, axis=0)
    
    # Seed 5: 8x2 Grid
    grid5 = list(itertools.product(range(8), range(2)))[:N]
    seed5 = np.array(grid5, dtype=float)
    seed5 -= np.mean(seed5, axis=0)

    # Seed 6: Circle (16 points evenly spaced)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    seed6 = np.column_stack([np.cos(angles), np.sin(angles)])

    # Seed 7: Two concentric circles (8+8)
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    seed7 = np.vstack([
        np.column_stack([0.5*np.cos(angles1), 0.5*np.sin(angles1)]),
        np.column_stack([np.cos(angles2), np.sin(angles2)])
    ])

    seeds = [seed1, seed2, seed3, seed4, seed5, seed6, seed7]
    
    # Evaluate seeds
    best_score = np.inf
    best_points = None
    for seed in seeds:
        s = objective(seed.flatten())
        if s < best_score:
            best_score = s
            best_points = seed.copy()
    
    # --- Differential Evolution ---
    # Bounds: each coordinate in [-2, 2]
    bounds = [(-2.0, 2.0)] * (N * D)
    
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init='latinhypercube'
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_points = de_result.x.reshape(-1, D)
    except Exception:
        pass
    
    # --- Local polish with Powell ---
    # Run Powell from the best DE result and from the best seed
    polish_starts = [best_points]
    for seed in seeds[:3]:  # Also try polishing the best few seeds
        polish_starts.append(seed)
    
    for start in polish_starts:
        try:
            res = minimize(
                objective, 
                start.flatten(), 
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(-1, D)
        except Exception:
            pass
    
    # --- Final local polish with Nelder-Mead from best ---
    try:
        res = minimize(
            objective, 
            best_points.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_points = res.x.reshape(-1, D)
    except Exception:
        pass
    
    # Ensure distinct points
    d_sq = pairwise_distances_sq(best_points)
    mask = ~np.eye(N, dtype=bool)
    if np.any(d_sq[mask] < 1e-10):
        # Fallback to best seed
        for seed in seeds:
            s = objective(seed.flatten())
            if s < best_score:
                best_score = s
                best_points = seed.copy()
    
    return best_points
