import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280 # Safety margin
        
        # Generate starting points
        # Use Halton sequence for good coverage
        def halton(index, base):
            f = 1
            r = 0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index = index // base
            return r

        def halton_sequence(count, dim):
            points = []
            for i in range(1, count + 1):
                point = []
                for d in range(dim):
                    # Use first few primes
                    base = [2, 3, 5, 7, 11][d]
                    point.append(halton(i, base))
                points.append(point)
            return np.array(points)

        # Map [0, 1] to [-5, 5]
        # We want 10 restarts.
        n_restarts = 10
        start_points = halton_sequence(n_restarts, self.dim)
        start_points = start_points * 10 - 5 # Map to [-5, 5]

        best_x = None
        best_f = float('inf')
        evals_used = 0

        for p0 in start_points:
            if evals_used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Run Nelder-Mead from p0
            # Budget for this restart
            remaining_budget = self.budget - evals_used
            # Limit per restart to avoid one outlier taking all
            # But if we have 10 restarts and 500 budget, 50 each is good.
            # Let's allow up to 50 evals per restart, or until convergence.
            
            x, f, used = self._nelder_mead(func, p0, remaining_budget, evals_used, deadline)
            evals_used += used
            
            if f < best_f:
                best_f = f
                best_x = x

        return best_x, best_f

    def _nelder_mead(self, func, x0, max_evals, global_evals, deadline):
        dim = self.dim
        x = np.array(x0)
        f0 = func(x)
        global_evals += 1
        if global_evals >= max_evals + 1: # Check budget
             return x, f0, 1
        
        # Initialize simplex
        simplex = [x]
        simplex_f = [f0]
        
        # Create other vertices
        for i in range(dim):
            xi = x.copy()
            # Step size: 0.05 for first coord, 0.25 for others? Standard is 0.05 or 0.25
            # Let's use 0.25 for robustness in [-5,5]
            step = 0.25 if xi[i] != 0 else 0.05
            xi[i] += step
            # Clip to bounds? The func clips, but we should stay close.
            # The problem says func clips to [-5,5].
            simplex.append(xi)
            fi = func(xi)
            simplex_f.append(fi)
            global_evals += 1
            if global_evals >= max_evals + 1:
                 return x, f0, 1

        # Main loop
        while global_evals < max_evals + 1 and time.time() < deadline:
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Convergence check
            if np.max(simplex_f) - np.min(simplex_f) < 1e-8:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = simplex_f[-1]
            
            # Reflection
            xr = centroid + 1.0 * (centroid - worst)
            fr = func(xr)
            global_evals += 1
            if global_evals >= max_evals + 1:
                break
            
            if simplex_f[0] <= fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            elif fr < simplex_f[0]:
                # Expansion
                xe = centroid + 2.0 * (centroid - worst)
                fe = func(xe)
                global_evals += 1
                if global_evals >= max_evals + 1:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            else:
                # Contraction
                if fr < worst_f:
                    # Outside contraction
                    xc = centroid + 0.5 * (centroid - worst)
                else:
                    # Inside contraction
                    xc = centroid + 0.5 * (worst - centroid)
                
                fc = func(xc)
                global_evals += 1
                if global_evals >= max_evals + 1:
                    break
                
                if fc < min(fr, worst_f):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex_f[i] = func(simplex[i])
                        global_evals += 1
                        if global_evals >= max_evals + 1:
                            break
                    if global_evals >= max_evals + 1:
                        break

        # Find best in simplex
        best_idx = np.argmin(simplex_f)
        return simplex[best_idx], simplex_f[best_idx], global_evals
