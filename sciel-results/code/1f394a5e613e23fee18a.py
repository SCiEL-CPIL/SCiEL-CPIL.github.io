"""Optimizer for BBOB f3 (Rastrigin) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        calls_made = 0
        max_calls = 450  # Leave some buffer below 500

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return np.inf
            calls_made += 1
            return func(x)

        best_x = None
        best_f = np.inf

        def try_candidate(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Generate diverse starting points
        starts = []
        
        # Integer lattice points (Rastrigin minima are at integers)
        # Sample a subset of integer points
        for _ in range(10):
            starts.append(np.round(np.random.uniform(-4, 4, dim)))

        # Random points
        for _ in range(10):
            starts.append(np.random.uniform(-4.5, 4.5, dim))

        # Origin
        starts.append(np.zeros(dim))

        # Shuffle
        np.random.shuffle(starts)

        def nelder_mead(func, x0, max_iter=50, tol=1e-8):
            """Optimized Nelder-Mead with call counting."""
            d = len(x0)
            simplex = [x0.copy()]
            f0 = func(x0)
            simplex_f = [f0]

            for i in range(d):
                x_i = x0.copy()
                if abs(x0[i]) > 1e-10:
                    x_i[i] *= 1.05
                else:
                    x_i[i] += 0.05
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
                simplex_f.append(func(x_i))

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_iter):
                if calls_made >= max_calls:
                    break
                
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                if simplex_f[-1] - simplex_f[0] < tol:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)

                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])

            best_idx = np.argmin(simplex_f)
            return simplex[best_idx], simplex_f[best_idx]

        # Run Nelder-Mead from a few starts
        for x0 in starts[:5]:
            if time.time() > deadline or calls_made >= max_calls:
                break
            try:
                x_nm, f_nm = nelder_mead(safe_call, x0, max_iter=50)
                try_candidate(x_nm, f_nm)
            except Exception:
                pass

        # Simple local refinement around best found so far
        if best_x is not None:
            # Try integer snapping
            x_int = np.round(best_x)
            x_int = np.clip(x_int, -5, 5)
            f_int = safe_call(x_int)
            try_candidate(x_int, f_int)

            # Small perturbations
            for _ in range(20):
                if calls_made >= max_calls:
                    break
                x_pert = best_x + 0.05 * np.random.randn(dim)
                x_pert = np.clip(x_pert, -5, 5)
                f_pert = safe_call(x_pert)
                try_candidate(x_pert, f_pert)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        return best_x, best_f
