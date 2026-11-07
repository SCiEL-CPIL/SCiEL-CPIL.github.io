"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex method for BBOB f20 (Schwefel).

    Strategy:
      1. Initialize a diverse set of starting points (center, random, corners).
      2. Run Nelder-Mead simplex optimization from each start point.
      3. Use a shared budget and time deadline to manage the multi-start process.
      4. Return the best result found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Nelder-Mead Simplex Method
        def nelder_mead(start_x, max_iter):
            nonlocal call_count, best_x, best_f
            
            # Initialize simplex
            simplex = []
            # First point is start_x
            f_start = make_call(start_x)
            if f_start is None:
                return
            update_best(start_x, f_start)
            simplex.append((start_x, f_start))
            
            # Other points of the simplex
            for i in range(dim):
                x_i = start_x.copy()
                # Step size: 0.05 * (1 - x_i[i]) if x_i[i] >= 0 else 0.05
                # Standard NM: 0.05 * (1 - x[i]) if x[i] >= 0 else 0.05
                # However, for bounded problems, we might want a fixed step or relative.
                # Let's use a step of 0.5 initially to cover more ground quickly, 
                # or 0.05 * (1 - x_i[i])? 
                # For Schwefel, features are ~4.79. 0.5 is reasonable.
                step = 0.5
                if x_i[i] + step > 5.0:
                    step = -0.5
                x_i[i] += step
                x_i = np.clip(x_i, -5.0, 5.0)
                
                f_i = make_call(x_i)
                if f_i is None:
                    return
                update_best(x_i, f_i)
                simplex.append((x_i, f_i))
                
                if time.time() > deadline - 10:
                    return

            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_sigma = 10.0
            min_sigma = 1e-8

            for _ in range(max_iter):
                if call_count[0] >= budget - 10:
                    break
                if time.time() > deadline - 10:
                    break

                # Sort simplex by function value
                simplex.sort(key=lambda x: x[1])
                
                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_best = simplex[0][0]
                f_best = simplex[0][1]
                
                # Check for convergence
                if f_worst - f_best < 1e-8:
                    break

                # Calculate centroid of all but worst point
                x_centroid = np.mean([x for x, _ in simplex[:-1]], axis=0)

                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = make_call(x_ref)
                if f_ref is None:
                    break
                update_best(x_ref, f_ref)

                if f_ref < f_best:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = make_call(x_exp)
                    if f_exp is None:
                        break
                    update_best(x_exp, f_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = (x_exp, f_exp)
                    else:
                        simplex[-1] = (x_ref, f_ref)
                elif f_ref < simplex[-2][1]:
                    simplex[-1] = (x_ref, f_ref)
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = x_centroid + rho * (x_ref - x_centroid)
                    else:
                        # Inside contraction
                        x_con = x_centroid + rho * (x_worst - x_centroid)
                    
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = make_call(x_con)
                    if f_con is None:
                        break
                    update_best(x_con, f_con)

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = (x_con, f_con)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            x_shrink = x_best + sigma * (simplex[i][0] - x_best)
                            x_shrink = np.clip(x_shrink, -5.0, 5.0)
                            f_shrink = make_call(x_shrink)
                            if f_shrink is None:
                                return
                            update_best(x_shrink, f_shrink)
                            simplex[i] = (x_shrink, f_shrink)
                        # Adjust sigma for next shrink (if it happens again, though usually we break or restart)
                        # Standard NM doesn't adapt sigma dynamically in the loop like this, 
                        # but we can reduce the step size for subsequent shrinks if needed.
                        # For simplicity, we'll just let the loop continue. 
                        # If it keeps shrinking, the simplex will collapse.
                        # We can force a break if the simplex is too small.
                        # Check size
                        size = np.linalg.norm(simplex[-1][0] - x_best)
                        if size < 1e-6:
                            break

        # Phase 1: Multi-start Nelder-Mead
        # Define start points
        start_points = []
        
        # 1. Origin
        start_points.append(np.zeros(dim))
        
        # 2. Random points
        for _ in range(10):
            start_points.append(rng.uniform(-5, 5, dim))
        
        # 3. Points near corners (shifted slightly inwards to avoid boundary issues if any, though clip handles it)
        # Let's add a few deterministic points to ensure coverage
        for d in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[d] = sign * 4.0
                start_points.append(p)
        
        # Shuffle to mix random and deterministic
        rng.shuffle(start_points)
        
        # Allocate budget
        # We have ~500 calls. NM in 5D takes ~50-100 calls to converge roughly.
        # Let's try 5 restarts.
        n_restarts = 5
        alloc_per_restart = budget // n_restarts
        
        for i in range(n_restarts):
            if call_count[0] >= budget - 10:
                break
            if time.time() > deadline - 20:
                break
            
            start_x = start_points[i]
            nelder_mead(start_x, alloc_per_restart)

        return best_x, best_f
# EVOLVE-BLOCK-END
