"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 150.0  # seconds; evaluator kills at 150s, use safety margin


def _bary_to_cart(bary):
    """Convert barycentric (a, b) with c=1-a-b to Cartesian.
    Triangle vertices: V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    Point = a*V0 + b*V1 + c*V2 where c = 1-a-b
    """
    a = bary[:, 0]
    b = bary[:, 1]
    c = 1.0 - a - b
    x = b * 1.0 + c * 0.5
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])


def _min_triangle_area_vec(points):
    """Compute minimum triangle area among all triples. points: (n,2) array."""
    n = points.shape[0]
    # Use vectorized approach: for each pair (i,j), compute cross product with all k
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            d = pj - pi  # (2,)
            # For all k > j, compute cross product d x (pk - pi)
            if j + 1 < n:
                rest = points[j + 1:]
                # cross = d[0]*(pk[1]-pi[1]) - d[1]*(pk[0]-pi[0])
                cross = d[0] * (rest[:, 1] - pi[1]) - d[1] * (rest[:, 0] - pi[0])
                areas = 0.5 * np.abs(cross)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area


def _objective_flat(flat):
    """Objective for optimizer: minimize negative min triangle area.
    flat: 2*N array of barycentric coords (a_i, b_i) for each point.
    """
    bary = flat.reshape(N, 2)
    # Clip to valid range (shouldn't be needed with bounds, but safety)
    a = np.clip(bary[:, 0], 0, 1)
    b = np.clip(bary[:, 1], 0, 1 - a)
    bary = np.column_stack([a, b])
    points = _bary_to_cart(bary)
    area = _min_triangle_area_vec(points)
    return -area


def construct():
    start_time = time.time()
    best_flat = None
    best_score = -float('inf')
    
    # Bounds for barycentric coordinates: a in [0,1], b in [0,1]
    # But we need a+b <= 1, so we'll use bounds [0,1] for both and clip in objective
    # Actually, for DE we need rectangular bounds. Use [0,1] for a and [0,1] for b,
    # then clip b to 1-a in the objective.
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Stage 1: Differential Evolution (global search)
    # Use multiple restarts with different seeds
    seeds = [42, 123, 777, 2024, 999]
    
    for seed_idx, seed in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = _DEADLINE - 30.0 - elapsed  # 30s safety margin
        if remaining < 10.0:
            break
        
        # Scale maxiter based on remaining time
        # DE with popsize=15, 22D: roughly 15*22 = 330 evals per generation
        # Each eval is fast (~1ms), so we can do many generations
        # Be conservative: assume ~500 evals/sec for DE
        max_iter = max(10, int(remaining * 500 / (15 * 22)))
        max_iter = min(max_iter, 200)  # cap to avoid too long
        
        try:
            result = differential_evolution(
                _objective_flat,
                bounds,
                seed=seed,
                maxiter=max_iter,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                init='latinhypercube',
                polish=False,  # We'll do our own local refinement
                updating='immediate',
            )
            elapsed = time.time() - start_time
            remaining = _DEADLINE - 30.0 - elapsed
            if remaining < 5.0:
                # Just evaluate the result
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_flat = result.x.copy()
                continue
            
            # Stage 2: Local refinement with L-BFGS-B
            # Use barycentric coords with bounds
            x0 = result.x.copy()
            # Clip to valid region
            a0 = np.clip(x0[0::2], 0, 1)
            b0 = np.clip(x0[1::2], 0, 1 - a0)
            x0 = np.zeros(2 * N)
            x0[0::2] = a0
            x0[1::2] = b0
            
            # Local refinement
            local_result = minimize(
                _objective_flat,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            
            score = -local_result.fun
            if score > best_score:
                best_score = score
                best_flat = local_result.x.copy()
                
        except Exception:
            continue
    
    # If no valid result from optimization, fall back to a reasonable heuristic
    if best_flat is None:
        # Use a grid-like initial placement
        # Place points in a triangular lattice pattern
        best_flat = np.zeros(2 * N)
        # Rough placement: spread points
        for i in range(N):
            # Distribute roughly
            row = i // 4
            col = i % 4
            a = 0.1 + 0.8 * col / 3.0
            b = 0.1 + 0.6 * row / 2.0
            if a + b > 0.9:
                b = 0.9 - a
            best_flat[2*i] = a
            best_flat[2*i+1] = b
    
    # Convert barycentric to Cartesian
    a = best_flat[0::2]
    b = best_flat[1::2]
    # Ensure valid
    b = np.clip(b, 0, 1 - a)
    c = 1.0 - a - b
    x = b * 1.0 + c * 0.5
    y = c * (np.sqrt(3) / 2)
    points = np.column_stack([x, y])
    
    return points
# EVOLVE-BLOCK-END
