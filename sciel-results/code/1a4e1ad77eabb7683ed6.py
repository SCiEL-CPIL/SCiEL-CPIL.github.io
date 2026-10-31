"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    def objective(x):
        if time.time() > deadline:
            return 1e10
        pts = x.reshape(N, 2)
        # Normalize to remove scale invariance, which helps DE converge faster
        # and prevents numerical issues with very large/small values
        pts = normalize(pts)
        return score_raw(pts)

    # Define bounds for the search space
    # Since we normalize inside the objective, the absolute scale doesn't matter,
    # but the relative positions do. A box of [-1, 1] is a reasonable starting range.
    # We can widen it if needed, but normalization makes it less critical.
    bounds = [(-1.0, 1.0)] * (N * 2)

    # Differential Evolution parameters
    # popsize: Number of members in the population. Higher means more exploration.
    # maxiter: Maximum number of generations.
    # tol: Tolerance for termination.
    # seed: Fixed for reproducibility in evaluation, though random is fine.
    
    # We want to balance exploration and exploitation.
    # A population size of 100-200 is often good for N=32 dimensions.
    # We will monitor time and stop if we exceed the deadline.
    
    # DE is stochastic, so we might want multiple runs if time permits, 
    # but a single robust run is usually better than many short ones.
    
    # Let's try a single DE run with sufficient iterations.
    # DE can be slow, so we set a reasonable maxiter and rely on the deadline check 
    # if it runs too long (though DE doesn't easily accept a callback for early stop 
    # in all scipy versions, we can wrap it or just let it run if we estimate time).
    
    # Actually, scipy's differential_evolution doesn't have a built-in time limit callback 
    # that stops the optimization loop cleanly in all versions without checking 
    # inside the objective (which we did above, but it just returns high values, 
    # it doesn't stop the loop).
    
    # To strictly respect the deadline, we should estimate the cost per iteration.
    # Or, we can use a simpler approach: run DE with a limited number of iterations 
    # that we know fits in the time budget.
    
    # Let's estimate: 
    # N=16, D=2 -> 32 dimensions.
    # Popsize=50 -> 50 evaluations per generation.
    # Each eval is O(N^2) = 256 ops. Very fast.
    # 1000 generations * 50 evals = 50,000 evals.
    # 50,000 evals * ~0.0001s/eval = 5s.
    # So we can afford many more iterations.
    
    # Let's try popsize=100, maxiter=2000.
    # 2000 * 100 = 200,000 evals. ~20s.
    # This leaves plenty of time for multiple restarts or a larger population.
    
    # Strategy: Run DE with a large population and many iterations.
    # If it finishes early, we can run another instance with a different seed 
    # to see if we find a better minimum.
    
    best_result = None
    best_score = 1e10
    
    # Run 1: Large population, many iterations
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=3000,
            popsize=100,
            tol=1e-6,
            polish=True  # Use L-BFGS-B to polish the final result, leveraging smoothness at the end
        )
        
        if time.time() < deadline:
            final_pts = res.x.reshape(N, 2)
            raw_score = score_raw(final_pts)
            if raw_score < best_score:
                best_score = raw_score
                best_result = res.x
    except Exception:
        pass

    # Run 2: If time permits, try a different seed or slightly different params
    if time.time() < deadline - 10.0:
        try:
            res2 = differential_evolution(
                objective, 
                bounds, 
                seed=123,
                maxiter=2000,
                popsize=80,
                tol=1e-6,
                polish=True
            )
            
            if time.time() < deadline:
                final_pts2 = res2.x.reshape(N, 2)
                raw_score2 = score_raw(final_pts2)
                if raw_score2 < best_score:
                    best_score = raw_score2
                    best_result = res2.x
        except Exception:
            pass

    # Fallback: If DE failed or took too long, use a grid/circle init
    if best_result is None:
        # Use a known good configuration: regular polygon or grid
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_result = pts.ravel()
        best_score = score_raw(pts)

    return best_result.reshape(N, 2)

# EVOLVE-BLOCK-END
