import numpy as np
from scipy.optimize import differential_evolution
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
    
    if np.any(d_sq_vals < 1e-12):
        return 1e10
    
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    return d_max_sq / d_min_sq

def construct():
    # Generate diverse seeds
    seeds = []
    
    # Seed 1: 4x4 Grid
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    seed1 = np.array(grid, dtype=float)
    seed1 -= np.mean(seed1, axis=0)
    seeds.append(seed1)
    
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
    seeds.append(seed2)
    
    # Seed 3: Regular 16-gon (all on unit circle)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    seed3 = np.column_stack([np.cos(angles), np.sin(angles)])
    seeds.append(seed3)
    
    # Seed 4: Two concentric octagons
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    ring2 = 0.5 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    seed4 = np.vstack([ring1, ring2])
    seeds.append(seed4)
    
    # Seed 5: Three rings: 1 center, 6 inner, 9 outer (approximate)
    center = np.array([[0.0, 0.0]])
    angles_mid = np.linspace(0, 2*np.pi, 6, endpoint=False)
    mid_ring = 0.5 * np.column_stack([np.cos(angles_mid), np.sin(angles_mid)])
    angles_outer = np.linspace(0, 2*np.pi, 9, endpoint=False)
    outer_ring = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    seed5 = np.vstack([center, mid_ring, outer_ring])
    seeds.append(seed5)
    
    # Seed 6: Random
    np.random.seed(42)
    seed6 = np.random.rand(N, D) * 2 - 1
    seed6 -= np.mean(seed6, axis=0)
    seeds.append(seed6)
    
    # Seed 7: 2x8 Grid
    grid2 = list(itertools.product(range(2), range(8)))[:N]
    seed7 = np.array(grid2, dtype=float)
    seed7 -= np.mean(seed7, axis=0)
    seeds.append(seed7)
    
    # Evaluate all seeds first
    best_score = np.inf
    best_points = None
    for seed in seeds:
        s = objective(seed.flatten())
        if s < best_score:
            best_score = s
            best_points = seed.copy()
    
    # Use differential evolution for optimization
    # Bounds: each coordinate in [-2, 2]
    bounds = [(-2, 2)] * (N * D)
    
    # Run DE from the best seed as initial population hint
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True,
            init='best'
        )
        if result.fun < best_score:
            best_score = result.fun
            best_points = result.x.reshape(-1, D)
    except Exception:
        pass
    
    # Also try DE from other good seeds
    for idx, seed in enumerate(seeds[:3]):
        try:
            # Create a local optimization around this seed
            # Use a smaller range
            local_bounds = [(-1, 1)] * (N * D)
            result_local = differential_evolution(
                objective,
                bounds=local_bounds,
                seed=42 + idx,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                mutation=(0.2, 0.8),
                recombination=0.7,
                polish=True
            )
            if result_local.fun < best_score:
                best_score = result_local.fun
                best_points = result_local.x.reshape(-1, D)
        except Exception:
            pass
    
    return best_points
