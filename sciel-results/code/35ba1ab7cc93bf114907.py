"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Powell's method with multi-start initialization for BBOB f12 (Bent Cigar).
    
    Strategy:
    1. Multi-start initialization: Evaluate several random points to find promising regions.
    2. Powell's method: Constructs a sequence of conjugate directions from successful
       coordinate and pattern moves, allowing efficient traversal of the narrow bent-cigar
       valley by aligning search axes with the function's principal curvatures.
    3. Budget management: Strict tracking of evaluations and time to ensure validity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            
            x = np.clip(np.asarray(x, dtype=float).flatten(), -5, 5)
            
            try:
                f = float(func(x))
            except Exception:
                return float('inf')
                
            if not np.isfinite(f):
                return float('inf')

            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        def line_search(x, direction, f0):
            """
            One-dimensional line search along direction.
            Returns (new_x, new_f, new_direction) where new_direction is the
            normalized direction that achieved the best improvement.
            """
            nonlocal eval_count
            
            # Try to find the minimum along the line x + t * direction
            # Using a simple golden-section or Fibonacci-like search
            # But for Powell's method, we typically just do a few steps
            
            # First, try to move in the positive direction
            # We'll use a simple approach: try several step sizes
            
            best_t = 0.0
            best_f = f0
            best_x_ls = x.copy()
            
            # Try a range of step sizes
            # Start with a moderate step
            t = 1.0
            for _ in range(20):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                x_trial = x + t * direction
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = eval_func(x_trial)
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_x_ls = x_trial.copy()
                    best_t = t
                    # Try to go further in this direction
                    t *= 1.5
                else:
                    # Try smaller step in opposite direction
                    t_neg = -t * 0.5
                    x_trial_neg = x + t_neg * direction
                    x_trial_neg = np.clip(x_trial_neg, -5, 5)
                    f_trial_neg = eval_func(x_trial_neg)
                    
                    if f_trial_neg < best_f:
                        best_f = f_trial_neg
                        best_x_ls = x_trial_neg.copy()
                        best_t = t_neg
                        t = abs(t) * 0.5
                    else:
                        # No improvement, try smaller step
                        t *= 0.5
                        if t < 1e-6:
                            break
                        continue
            
            return best_x_ls, best_f, best_t

        def powell_method(start_x, initial_f=None):
            """
            Powell's method for derivative-free optimization.
            """
            nonlocal eval_count
            
            if eval_count >= budget - 10:
                return
            
            x = np.clip(start_x, -5, 5).copy()
            if initial_f is not None:
                f = initial_f
            else:
                f = eval_func(x)
                if eval_count >= budget:
                    return
            
            # Initialize directions to standard basis vectors
            directions = [np.eye(d)[i] for i in range(d)]
            
            # Maximum number of iterations
            max_iter = 100
            tol = 1e-6
            
            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                x_old = x.copy()
                f_old = f
                min_f = f
                min_idx = 0
                
                # Step 1: Minimize along each direction
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        return
                    
                    x_new, f_new, _ = line_search(x, directions[i], f)
                    
                    if f_new < min_f:
                        min_f = f_new
                        min_idx = i
                    
                    x = x_new
                    f = f_new
                
                # Check for convergence
                if abs(f_old - f) < tol * (1 + abs(f)):
                    break
                
                # Step 2: Compute the new direction
                # The new direction is 2 * (x - x_old) - (x_old - x_prev)
                # But in Powell's method, it's simply the vector from the
                # starting point of the iteration to the ending point
                x_start_iter = x_old.copy()
                x_end_iter = x.copy()
                
                # The new direction is the difference between the end and start
                # of the full iteration (all coordinate minimizations)
                new_dir = x_end_iter - x_start_iter
                new_dir_norm = np.linalg.norm(new_dir)
                
                if new_dir_norm < 1e-10:
                    break
                
                # Step 3: Check if we should replace a direction
                # If the improvement along the new direction is greater than
                # the improvement along the worst direction, replace it
                f_new_dir_test = eval_func(x_end_iter - new_dir)
                
                # Actually, in standard Powell's method:
                # We minimize along the new direction from x_old
                x_test, f_test, _ = line_search(x_old, new_dir, f_old)
                
                if f_test < f:
                    # Replace the direction that gave the worst improvement
                    # with the new direction
                    directions[min_idx] = new_dir / new_dir_norm
                    x = x_test
                    f = f_test
                
                # Check convergence again
                if abs(f_old - f) < tol * (1 + abs(f)):
                    break
            
            return

        # --- Phase 1: Multi-Start Initialization ---
        rng = np.random.default_rng(42)
        
        # Generate initial candidates
        # Include origin and random points in the likely range of the minimizer [-4, 4]
        init_points = [np.zeros(d)]
        for _ in range(9):
            init_points.append(rng.uniform(-4, 4, d))
            
        # Evaluate all initial points
        point_scores = []
        for p in init_points:
            if eval_count >= budget:
                break
            fp = eval_func(p)
            point_scores.append((fp, p))
            
        # Sort by function value
        point_scores.sort(key=lambda x: x[0])
        
        # --- Phase 2: Local Refinement using Powell's Method ---
        # Refine the top K points.
        # K=3 is a good balance between exploration and exploitation.
        top_k = min(3, len(point_scores))
        
        for i in range(top_k):
            if eval_count >= budget - 20:
                break
            
            # Refine the i-th best point
            # Pass the known function value to save one evaluation
            powell_method(point_scores[i][1], point_scores[i][0])
            
            # Update best if Powell found a better point
            # (eval_func already updates best_x, best_f)
            
            # If we still have budget, try another restart from the current best
            if i < top_k - 1 and eval_count < budget - 50:
                # Small perturbation to escape potential local minima
                perturbation = rng.uniform(-0.5, 0.5, d)
                new_start = best_x + perturbation
                new_start = np.clip(new_start, -5, 5)
                powell_method(new_start)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
