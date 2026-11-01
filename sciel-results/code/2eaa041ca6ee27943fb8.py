import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Concentric Rings
    # Structure: 1, 6, 12, 18, 13
    counts = [1, 6, 12, 18, 13]
    # Radii estimation
    # Target radius approx 0.125
    # Outer boundary approx 1 - 0.125 = 0.875
    R_max = 0.85
    # Scale radii: sqrt(k)
    radii = []
    for k in range(1, len(counts) + 1):
        radii.append(R_max * np.sqrt(k / len(counts)))
    
    points = []
    idx = 0
    for i, count in enumerate(counts):
        if count == 1:
            points.append([0.0, 0.0])
        else:
            r_ring = radii[i]
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Add small random jitter to break symmetry and help optimizer
            angles += np.random.uniform(-0.1, 0.1, count)
            for a in angles:
                points.append([r_ring * np.cos(a), r_ring * np.sin(a)])
    
    points = np.array(points)
    
    # Helper to calculate objective for fixed r
    def objective(x_flat, r):
        x = x_flat.reshape(-1, 2)
        # Boundary violations
        dists_origin = np.linalg.norm(x, axis=1)
        boundary_viol = np.maximum(0, dists_origin - (1.0 - r))
        penalty_boundary = np.sum(boundary_viol ** 2)
        
        # Pairwise violations
        # d_ij >= 2r
        # Calculate all pairwise distances
        # Using broadcasting for N=50 is 50x50, which is fine
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask out diagonal
        np.fill_diagonal(dists, np.inf)
        # We only need i < j, but summing all is 2x work, still fast
        # Violation if dist < 2r
        pair_viol = np.maximum(0, 2.0 * r - dists)
        penalty_pair = np.sum(pair_viol ** 2)
        
        return penalty_pair + penalty_boundary

    # Helper to check if a configuration is valid for a given r
    def is_valid(x, r, tol=1e-5):
        dists_origin = np.linalg.norm(x, axis=1)
        if np.any(dists_origin > 1.0 - r + tol):
            return False
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.any(dists < 2.0 * r - tol):
            return False
        return True

    # Start with a small r
    r_current = 0.10
    best_x = points.copy()
    best_r = r_current
    
    # Initial optimization to settle the initial configuration
    # We can run a few iterations of increasing r
    
    step = 0.005
    min_step = 0.0001
    
    while time.time() < deadline:
        r_target = r_current + step
        if r_target > 0.14: # Hard cap, known best is ~0.1258
            break
            
        # Optimize x for fixed r_target
        res = minimize(
            lambda x: objective(x, r_target),
            best_x.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        new_x = res.x.reshape(-1, 2)
        
        # Check if valid
        if is_valid(new_x, r_target, tol=1e-4):
            best_x = new_x
            best_r = r_target
            # If successful, keep step or increase slightly?
            # Keep step same or increase slowly
            # step = min(step * 1.1, 0.01)
        else:
            # If invalid, reduce step
            step *= 0.8
            if step < min_step:
                break
            # Try again with smaller step? 
            # Or just break if we can't make progress
            # Let's try a few retries with smaller step
            # Actually, if L-BFGS-B fails to find a valid config, 
            # increasing r is likely impossible from this local minimum.
            # We should try to restart from a different initialization or break.
            # Given time constraints, breaking is safer.
            # However, sometimes the optimizer gets stuck in a local min of the penalty.
            # Let's try one more time with a smaller step.
            if step > min_step * 2:
                continue
            else:
                break

    # Final polish: Try to squeeze out a bit more with Nelder-Mead on the raw objective?
    # The raw objective is min(dist/2, 1-dist).
    # Maximizing this is equivalent to minimizing -min(...).
    # L-BFGS-B on the penalty is usually sufficient.
    
    # Ensure output is valid
    # Calculate actual score
    def calc_score(x):
        dists_origin = np.linalg.norm(x, axis=1)
        r_boundary = np.min(1.0 - dists_origin)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        return min(r_boundary, r_pair)
    
    current_score = calc_score(best_x)
    
    # If the final configuration is invalid due to floating point errors in is_valid tol,
    # we might need to shrink r slightly.
    # But construct() returns points, the evaluator calculates score.
    # If points are invalid (overlap or out of bounds), score is 0 or invalid.
    # is_valid uses tol=1e-4, so it's likely strictly valid for r_current.
    # The evaluator uses double precision.
    
    return best_x

# EVOLVE-BLOCK-END
