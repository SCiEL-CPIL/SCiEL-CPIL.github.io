import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0  # Safety margin against 300s/330s limit
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def time_check():
            return (time.time() - start_time) > timeout

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time_check():
                return None
            f = func(x)
            eval_count += 1
            update_best(x, f)
            return f

        # --- Phase 1: Coarse Search (Latin Hypercube) ---
        # Use 30% of the budget for coarse search
        n_coarse = min(50, budget // 3)
        
        # Generate Latin Hypercube points
        # For each dimension, divide [-5, 5] into n_coarse intervals and pick one point in each
        # This ensures good coverage.
        lhp = np.zeros((n_coarse, dim))
        for d in range(dim):
            # Shuffle indices to create the LHS property
            idx = rng.permutation(n_coarse)
            # Uniform points in (0, 1)
            u = (idx + rng.uniform(0, 1, n_coarse)) / n_coarse
            lhp[:, d] = -5.0 + 10.0 * u
        
        for i in range(n_coarse):
            if eval_count >= budget or time_check():
                break
            evaluate(lhp[i])
            
        if best_x is None:
            # Fallback if something went wrong
            x0 = rng.uniform(-5.0, 5.0, dim)
            evaluate(x0)
            if best_x is None:
                return np.zeros(dim), 0.0

        # --- Phase 2: Fine-tuning with Nelder-Mead from top candidates ---
        # We need the fitness values of the LHS points to sort them.
        # Let's re-evaluate the top few? No, we should have stored them.
        # Since we didn't store all LHS results separately, let's just pick the current best
        # and maybe a few random neighbors? 
        # Better approach: Keep track of the best few points from Phase 1.
        
        # Let's restart Phase 1 slightly differently to keep top candidates.
        # Actually, the current logic just keeps the global best. 
        # For NM, starting from the single best LHS point is often sufficient for smooth-ish functions,
        # but for Step Ellipsoidal (which has flat regions), we might need to jump.
        
        # Let's try a multi-start NM strategy.
        # We will run NM from the best point found in Phase 1.
        # If it fails to improve significantly, we might try another point?
        # Given the budget, let's just run a robust NM from the best LHS point.
        
        # Nelder-Mead Parameters
        # Initial simplex: best_x + small perturbations
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        # Initial simplex size
        # Based on the distance to the center or just a fixed small step
        # Since the domain is [-5, 5], a step of 0.5-1.0 is reasonable.
        # Let's use a step relative to the domain size.
        step_size = 1.0
        
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_x
        for i in range(1, dim + 1):
            simplex[i] = best_x.copy()
            simplex[i, i - 1] += step_size
            
        # Evaluate initial simplex
        # simplex[0] is already evaluated (it's best_x)
        f_simplex = np.zeros(dim + 1)
        f_simplex[0] = best_f
        
        for i in range(1, dim + 1):
            f_val = evaluate(simplex[i])
            if f_val is None:
                break
            f_simplex[i] = f_val
            
        # Main NM Loop
        # We have a budget of remaining evaluations
        while eval_count < budget and not time_check():
            # Sort simplex by fitness
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = evaluate(x_ref)
            if f_ref is None:
                break
            
            if f_ref < f_simplex[0]:
                # 2. Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                f_exp = evaluate(x_exp)
                if f_exp is None:
                    break
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_simplex[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
            elif f_ref < f_simplex[-2]:
                # 3. Accept reflection
                simplex[-1] = x_ref
                f_simplex[-1] = f_ref
            else:
                # 4. Contraction
                # Outside or Inside contraction? Standard is Inside if f_ref > f_simplex[-1]
                if f_ref < f_simplex[-1]:
                    # Inside contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = evaluate(x_con)
                    if f_con is None:
                        break
                    
                    if f_con <= min(f_ref, f_simplex[-1]):
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # 5. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_shrink = evaluate(simplex[i])
                            if f_shrink is None:
                                break
                            f_simplex[i] = f_shrink
                        if f_shrink is None:
                            break
                else:
                    # Outside contraction
                    x_con = centroid - rho * (centroid - simplex[-1])
                    f_con = evaluate(x_con)
                    if f_con is None:
                        break
                    
                    if f_con <= f_simplex[-1]:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # 5. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_shrink = evaluate(simplex[i])
                            if f_shrink is None:
                                break
                            f_simplex[i] = f_shrink
                        if f_shrink is None:
                            break
        
        return best_x, best_f
