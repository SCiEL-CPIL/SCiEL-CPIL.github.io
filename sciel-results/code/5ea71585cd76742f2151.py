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
    best_score = np.inf
    best_points = None
    
    # Helper to generate points from ring parameters
    def generate_ring_points(radii, offsets, counts):
        points = []
        for r, off, n in zip(radii, offsets, counts):
            if n == 0:
                continue
            angles = off + 2 * np.pi * np.arange(n) / n
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points.append(np.column_stack([x, y]))
        if not points:
            return np.zeros((N, D))
        return np.vstack(points)

    # Helper objective for ring parameters
    def objective_ring(params, counts):
        k = len(counts)
        radii = np.abs(params[:k])
        offsets = params[k:]
        points = generate_ring_points(radii, offsets, counts)
        if len(points) != N:
            return 1e10
        return objective(points.flatten())

    # Define several symmetric topologies
    topologies = [
        [16],
        [8, 8],
        [4, 8, 4],
        [6, 4, 6],
        [4, 4, 4, 4],
        [2, 6, 4, 4],
        [1, 5, 5, 5],
        [1, 3, 6, 6],
        [1, 2, 4, 4, 5],
    ]
    
    for counts in topologies:
        if sum(counts) != N:
            continue
            
        k = len(counts)
        # Variables: k radii, k angles
        bounds = [(0.0, 3.0)] * k + [(-np.pi, np.pi)] * k
        
        try:
            res = differential_evolution(
                objective_ring, 
                bounds, 
                args=(counts,),
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            
            if res.fun < best_score:
                best_score = res.fun
                radii = np.abs(res.x[:k])
                offsets = res.x[k:]
                best_points = generate_ring_points(radii, offsets, counts)
                
        except Exception:
            pass

    # If no valid points found, fallback to grid
    if best_points is None:
        grid = list(itertools.product(range(4), repeat=2))[:N]
        best_points = np.array(grid, dtype=float)
        best_score = objective(best_points.flatten())

    # Final polish: Nelder-Mead on full 32D space
    # This allows breaking symmetry if it helps, or fine-tuning
    try:
        res_nm = minimize(
            objective, 
            best_points.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_points = res_nm.x.reshape(-1, D)
    except Exception:
        pass

    # Ensure distinct points
    d_sq = pairwise_distances_sq(best_points)
    mask = ~np.eye(N, dtype=bool)
    if np.any(d_sq[mask] < 1e-10):
        # Fallback
        grid = list(itertools.product(range(4), repeat=2))[:N]
        best_points = np.array(grid, dtype=float)
        
    return best_points
