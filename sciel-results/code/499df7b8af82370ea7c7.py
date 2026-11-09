import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        calls = 0
        
        def wrapped_func(x):
            nonlocal calls
            calls += 1
            return func(x)

        best_x = None
        best_f = np.inf

        def nelder_mead(x0):
            nonlocal calls, best_x, best_f
            # Initialize simplex
            delta = 0.5
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0
                simplex[i + 1][i] += delta
            
            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if calls >= budget:
                    return
                fvals[i] = wrapped_func(simplex[i])
            
            # Sort simplex
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            while calls < budget:
                # 1. Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                
                if calls >= budget:
                    break
                f_ref = wrapped_func(x_ref)
                
                if f_ref < fvals[0]:
                    # 2. Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if calls >= budget:
                        break
                    f_exp = wrapped_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # 3. Accept Reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # 4. Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if calls >= budget:
                        break
                    f_con = wrapped_func(x_con)
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # 5. Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls >= budget:
                                break
                            fvals[i] = wrapped_func(simplex[i])
                        
                        if calls >= budget:
                            break

                # Sort simplex
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                
                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

        # Multi-start
        num_restarts = 5
        # Distribute budget roughly evenly
        for i in range(num_restarts):
            if calls >= budget:
                break
            x0 = np.random.uniform(-4, 4, n)
            nelder_mead(x0)

        # Final polish
        if best_x is not None and calls < budget:
            nelder_mead(best_x)

        return best_x, best_f
