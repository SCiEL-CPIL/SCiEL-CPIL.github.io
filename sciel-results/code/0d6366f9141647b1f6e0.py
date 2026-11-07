"""Multi-Start Tchebycheff Gradient Descent for RE34."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using Multi-Start Tchebycheff (Min-Max) Scalarization.
    
    Strategy:
    1. Uses a set of predefined weight vectors (uniformly distributed on the simplex) 
       to decompose the multi-objective problem into scalar sub-problems.
    2. For each weight vector, initializes multiple starting points (stratified sampling).
    3. Uses L-BFGS-B (a gradient-based method) to solve each scalar sub-problem.
    4. This approach is structurally different from DE as it exploits local gradient 
       information to aggressively converge to the Pareto front, filling gaps more 
       efficiently than stochastic mutations in a limited budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 290s to be safe.
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        archive = []
        archive_F = []
        
        # 1. Generate Weight Vectors for Tchebycheff
        # We need weights w_i >= 0, sum(w_i) = 1
        # For 3 objectives, we can generate points on the 2D simplex.
        # To ensure coverage, we use a grid or quasi-random points.
        # Given budget 200, let's use ~50 weight vectors.
        # Each weight vector gets ~4 starting points (50 * 4 = 200).
        # However, L-BFGS-B uses multiple evaluations per run.
        # We need to be careful with the budget.
        
        # Let's estimate cost per L-BFGS-B run.
        # In 5D, L-BFGS-B might take 10-30 function evaluations to converge.
        # So we can afford roughly 200 / 20 = 10 runs? That's too few.
        
        # Alternative: Use a cheaper local optimizer or fewer evaluations per run?
        # Or, use a hybrid: Fewer, higher-quality runs.
        # Let's try 20 weight vectors.
        # For each weight vector, we do 1 L-BFGS-B run from a specific start.
        # If budget allows, we can do multiple starts per weight.
        
        # Let's generate 10 weight vectors (corners and center of simplex).
        # Simplex for 3 objectives: w1+w2+w3=1, wi>=0.
        # Points:
        # (1,0,0), (0,1,0), (0,0,1) -> Corners (emphasize one objective)
        # (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) -> Edges
        # (1/3, 1/3, 1/3) -> Center
        # (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8) -> Near corners
        # (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6) -> Mid-radius
        # (0.4, 0.4, 0.2), (0.4, 0.2, 0.4), (0.2, 0.4, 0.4) -> Another mid
        # (0.3, 0.3, 0.4), (0.3, 0.4, 0.3), (0.4, 0.3, 0.3) -> Another mid
        
        # Let's use a more systematic generation:
        # Grid on simplex.
        # For 3 objs, we can parameterize by u, v in [0,1] with u+v<=1.
        # w1 = u, w2 = v, w3 = 1-u-v.
        # Let's pick 10 specific points to cover the space well.
        
        weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3, 1/3, 1/3],
            [0.7, 0.15, 0.15],
            [0.15, 0.7, 0.15],
            [0.15, 0.15, 0.7],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4],
            [0.3, 0.3, 0.4],
            [0.3, 0.4, 0.3],
            [0.4, 0.3, 0.3],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
        
        n_weights = len(weights)
        
        # 2. Define Scalar Objective Function
        # Tchebycheff: min_x max_i { w_i * (f_i(x) - z_i^*) }
        # We need z^* (ideal point). We estimate it from the archive or use a large constant?
        # Better: Use the current minimums in the archive.
        # If archive is empty, use a heuristic.
        # Since we are running sequentially, we can update z^* as we find better points.
        
        def scalar_objective(x, w, z_star, rho):
            """
            Tchebycheff scalarization.
            rho is the augmentation factor (typically 0.05).
            """
            f = problem(x)
            # Update global used counter? No, we need to track evaluations.
            # We must be careful: `problem` is called here.
            nonlocal used
            if used >= self.budget:
                return 0.0 # Dummy value to stop
            
            used += 1
            
            f = np.array(f)
            # Handle non-finite values
            if not np.all(np.isfinite(f)):
                return 1e10
            
            # Calculate Tchebycheff value
            # max_i (w_i * (f_i - z_i))
            # Note: z_star should be the ideal point (min of each objective).
            # If z_star is not known, we can use the min seen so far.
            
            # To avoid division by zero or issues if z_star is bad,
            # we normalize f by the range if possible, but Tchebycheff is scale dependent.
            # Standard Tchebycheff: max_i (w_i * (f_i - z_i^*))
            
            # We need to pass z_star.
            # Since z_star changes, we can't easily cache.
            
            # Actually, for L-BFGS-B, we need a smooth function.
            # The max operation is not smooth. We can use a smooth approximation
            # or just rely on the optimizer's robustness.
            # L-BFGS-B can handle non-smooth functions to some extent, but it's better to be smooth.
            # Let's use a soft-max approximation? Or just use the exact max.
            # SciPy's L-BFGS-B uses numerical gradients by default if jac is not provided.
            # It can handle kinks, but might be slower.
            
            # Let's use the exact Tchebycheff function.
            diffs = f - z_star
            # If any f_i < z_star, the term is negative. Max will ignore it if others are positive.
            # But we want to minimize the max deviation from the ideal point.
            # If a point is better than z_star in one objective, that term is negative.
            # The max will be determined by the worst objective.
            
            # One issue: if z_star is far from the front, the weights don't matter much.
            # As we find better points, z_star updates, making the problem harder.
            
            # To make it robust, we can normalize f by the current range?
            # No, Tchebycheff is defined on the original scale.
            
            # Let's stick to standard Tchebycheff.
            # We need to ensure z_star is not too optimistic.
            # We will update z_star to be the component-wise minimum of the archive.
            
            tcheb = np.max(w * diffs)
            
            # Augmented Tchebycheff: add rho * sum(w_i * (f_i - z_i))
            # This ensures that even if the max is 0, we still push towards the ideal point.
            aug = rho * np.sum(w * diffs)
            
            return tcheb + aug

        # We need to track z_star (ideal point)
        # Initialize z_star to a very large value? Or the first point?
        # Let's initialize z_star to the minimums of the first few random points.
        
        # 3. Initialization
        # Evaluate a few random points to get an initial z_star and archive.
        n_init = 10
        for _ in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.random.uniform(lo, hi)
            f = problem(x)
            used += 1
            f = np.array(f)
            if np.all(np.isfinite(f)):
                archive.append(x)
                archive_F.append(f)
        
        if not archive_F:
            return archive
        
        z_star = np.min(archive_F, axis=0)
        
        # 4. Main Loop: Iterate over weight vectors
        # For each weight, run L-BFGS-B from a few starting points.
        # Starting points: Center of bounds, and random points.
        
        # To maximize coverage, we should start from points that are diverse.
        # For each weight, we can start from:
        # 1. The point in the archive closest to the "ideal" direction defined by w?
        # 2. Or just random points?
        # Random points are safer for multi-start.
        
        # How many L-BFGS-B runs?
        # Budget remaining: 200 - 10 = 190.
        # 22 weights.
        # If we do 1 run per weight, that's 22 runs.
        # Each run uses ~10-20 evals. 22 * 20 = 440 > 190.
        # So we can't do a full L-BFGS-B for all weights if they are expensive.
        
        # Strategy Adjustment:
        # Use a cheaper optimizer? Or fewer weights?
        # Or, use L-BFGS-B with a limited number of iterations?
        # `maxiter` parameter in L-BFGS-B.
        
        # Let's use `maxiter=10` for L-BFGS-B to keep it fast.
        # This makes it more like a steepest descent with line search.
        
        # Also, we can skip weights if we run out of time/budget.
        
        for w in weights:
            if used >= self.budget or time.time() > deadline:
                break
            
            # Update z_star
            z_star = np.min(archive_F, axis=0)
            
            # Define the function for this specific weight and z_star
            # Note: z_star is captured in the closure. If z_star changes during the run,
            # the function definition changes. This is problematic for L-BFGS-B.
            # Solution: Fix z_star for the duration of this L-BFGS-B run.
            # This is standard in multi-start Tchebycheff.
            
            current_z_star = z_star.copy()
            
            def obj_func(x, w=w, z=current_z_star):
                return scalar_objective(x, w, z, rho=0.05)
            
            # Starting points for this weight
            # 1. Random point
            # 2. Point in archive with best scalar value for this weight?
            
            # Let's just use 2 random starts per weight to save budget.
            # 22 weights * 2 starts * 10 evals/start = 440 evals. Still too many.
            
            # Let's reduce to 1 start per weight.
            # 22 weights * 1 start * 10 evals = 222 evals. Close to 190.
            # We might run out.
            
            # Let's dynamically choose starts.
            # Start from a random point.
            x0 = np.random.uniform(lo, hi)
            
            # Check budget
            if used >= self.budget:
                break
            
            # Run L-BFGS-B
            # Bounds for L-BFGS-B
            lb_bounds = list(zip(lo, hi))
            
            try:
                # maxiter=5 to be very fast?
                # L-BFGS-B overhead is low.
                res = minimize(
                    obj_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=lb_bounds,
                    options={'maxiter': 10, 'ftol': 1e-5}
                )
                
                # If the run finished and we used budget, check the result
                if used < self.budget: # Should be true if minimize returned
                    x_best = res.x
                    f_best = problem(x_best)
                    used += 1
                    f_best = np.array(f_best)
                    
                    if np.all(np.isfinite(f_best)):
                        # Check if non-dominated
                        dominated = False
                        for af in archive_F:
                            if np.all(af <= f_best) and np.any(af < f_best):
                                dominated = True
                                break
                        
                        if not dominated:
                            # Remove dominated points from archive
                            new_archive_F = []
                            new_archive = []
                            for i, af in enumerate(archive_F):
                                if not (np.all(f_best <= af) and np.any(f_best < af)):
                                    new_archive_F.append(af)
                                    new_archive.append(archive[i])
                            
                            new_archive_F.append(f_best)
                            new_archive.append(x_best)
                            
                            archive = new_archive
                            archive_F = new_archive_F
                            
                            # Update z_star
                            z_star = np.min(archive_F, axis=0)
                            
            except Exception:
                # If L-BFGS-B fails, ignore
                pass
        
        return archive

# EVOLVE-BLOCK-END
