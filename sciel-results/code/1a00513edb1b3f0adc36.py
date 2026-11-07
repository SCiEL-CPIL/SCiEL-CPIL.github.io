import numpy as np
import time

class Optimizer:
    """Nelder-Mead with adaptive simplex for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(start_x, initial_alpha=0.5, max_iter=5000):
            nonlocal eval_count
            if eval_count >= budget:
                return

            # Initialize simplex
            simplex = np.zeros((d + 1, d))
            fvals = np.zeros(d + 1)
            
            simplex[0] = np.clip(start_x, -5, 5)
            fvals[0] = eval_func(simplex[0])
            
            for i in range(d):
                simplex[i + 1] = simplex[0].copy()
                simplex[i + 1][i] += initial_alpha
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
                fvals[i + 1] = eval_func(simplex[i + 1])

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = eval_func(x_r)
                
                if eval_count >= budget:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = eval_func(x_e)
                    if eval_count >= budget:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = eval_func(x_c)
                    if eval_count >= budget:
                        break
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = eval_func(simplex[i])
                            if eval_count >= budget:
                                break
                        if eval_count >= budget:
                            break

        # Phase 1: Initial Nelder-Mead from origin
        nelder_mead(np.zeros(d), initial_alpha=1.0)

        # Phase 2: If budget remains, restart from best with smaller alpha
        if eval_count < budget - 50 and time.time() < deadline:
            nelder_mead(best_x, initial_alpha=0.5)

        # Phase 3: Final fine-tuning
        if eval_count < budget - 20 and time.time() < deadline:
            nelder_mead(best_x, initial_alpha=0.1)

        return best_x, best_f
