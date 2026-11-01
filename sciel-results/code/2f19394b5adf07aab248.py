import numpy as np
from scipy.optimize import differential_evolution

N = 300

def _score_from_flat(x):
    """Compute the packing radius from a flat array of 2*N coordinates."""
    pts = x.reshape(N, 2)
    # Check for non-finite values
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms > 1.0):
        return -1.0
    min_boundary = 1.0 - np.min(norms)
    
    # Pairwise distances
    # For N=300, this creates a 300x300x2 array, which is fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _negative_score(x):
    """Objective function for minimization (we want to maximize score)."""
    return -_score_from_flat(x)

def construct():
    # Set a seed for reproducibility? The task says "Each evaluation must finish within 120 s".
    # Differential evolution can be stochastic. Let's use a fixed seed for stability in evaluation
    # or just let it run. Usually, for these tasks, deterministic is safer if possible,
    # but DE is inherently stochastic. We will set a seed to ensure the result is consistent
    # across runs if the evaluator runs it multiple times, though typically it runs once.
    # Actually, better not to fix seed if we want to explore, but for a "candidate" we just need one good result.
    # Let's use a fixed seed to be safe against bad random draws.
    np.random.seed(42)
    
    # Initial guess: Sunflower spiral
    i = np.arange(N)
    angle = 2.0 * np.pi * i / (1.0 + np.sqrt(5.0))
    r = np.sqrt((i + 0.5) / N) * 0.95  # Slightly inside to avoid boundary issues initially
    init_points = np.column_stack((r * np.cos(angle), r * np.sin(angle)))
    init_flat = init_points.flatten()
    
    # Define bounds for differential evolution
    # Coordinates must be within [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Run Differential Evolution
    # maxiter=1000 might be too slow for 600 dimensions?
    # Let's check the cost. Each evaluation is O(N^2) ~ 90,000 operations.
    # DE population size is typically 15 * dim = 9000.
    # 1000 iterations * 9000 evals * 90,000 ops = 8.1e11 ops. This is likely too slow for 120s.
    
    # We need a more efficient approach or fewer iterations.
    # Let's reduce the population size and iterations.
    # popsize=5 -> 5 * 600 = 3000 evals per iteration.
    # maxiter=50 -> 150,000 evals.
    # 150,000 * 90,000 = 1.35e10 ops. Still might be tight but NumPy is fast.
    # Let's try popsize=5, maxiter=50, tol=1e-4.
    
    # Alternatively, we can use a smaller initial population and rely on the local improvement.
    # Or use a hybrid approach.
    
    # Let's try a lighter DE run first.
    result = differential_evolution(
        _negative_score,
        bounds,
        maxiter=50,
        popsize=5,
        tol=1e-4,
        seed=42,
        polish=True,  # Use L-BFGS-B for final polish
        init='latinhypercube'
    )
    
    best_flat = result.x
    best_points = best_flat.reshape(N, 2)
    
    # Post-process: Project points back into the unit circle if any are slightly outside
    # due to numerical errors or the optimizer pushing them.
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
    
    # Local refinement: Simple hill climbing on the best point found
    # This helps fix any remaining local issues and fine-tune the radius.
    def calc_min_radius(pts):
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dists_boundary)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_boundary, min_pair)
    
    current_r = calc_min_radius(best_points)
    
    step_size = 0.005
    max_iter = 1000
    no_improve_count = 0
    threshold = 1e-10
    
    for _ in range(max_iter):
        # Identify the bottleneck
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dist_val = np.min(dists)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        dists_boundary = 1.0 - np.linalg.norm(best_points, axis=1)
        min_boundary_val = np.min(dists_boundary)
        min_boundary_idx = np.argmin(dists_boundary)
        
        if min_dist_val / 2.0 < min_boundary_val:
            vec = best_points[i] - best_points[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                direction = vec / norm
                best_points[i] += direction * step_size * 0.5
                best_points[j] -= direction * step_size * 0.5
            else:
                best_points[i] += np.array([step_size, 0])
                best_points[j] -= np.array([step_size, 0])
        else:
            idx = min_boundary_idx
            vec = -best_points[idx]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                direction = vec / norm
                best_points[idx] += direction * step_size
            else:
                pass
        
        norms = np.linalg.norm(best_points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            best_points[outside] /= norms[outside, np.newaxis]
            
        new_r = calc_min_radius(best_points)
        
        if new_r > current_r + threshold:
            current_r = new_r
            no_improve_count = 0
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-7:
                break
                
        if no_improve_count > 100:
            break
            
    return best_points
