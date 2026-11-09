"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_polish(self, func, x0, budget_left, deadline):
        """Nelder-Mead local search."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Initial simplex
        simplex = [x.copy()]
        simplex_f = [fx]
        
        delta = 0.5
        for i in range(d):
            if calls >= budget_left or time.time() > deadline:
                break
            xi = x.copy()
            xi[i] += delta
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            simplex_f.append(func(xi))
            calls += 1
        
        if len(simplex) < d + 1:
            return best_x, best_f

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter = 100
        for _ in range(max_iter):
            if calls >= budget_left or time.time() > deadline:
                break
            
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if calls >= budget_left or time.time() > deadline:
                if fr < simplex_f[-1]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                break
                
            if simplex_f[0] <= fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            elif fr < simplex_f[0]:
                if calls >= budget_left or time.time() > deadline:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    break
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if calls >= budget_left or time.time() > deadline:
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    break
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            else:
                if calls >= budget_left or time.time() > deadline:
                    break
                if fr < simplex_f[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if calls >= budget_left or time.time() > deadline:
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    break
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    for i in range(1, d + 1):
                        if calls >= budget_left or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse-to-fine 1D grid search to identify basin
        # Rastrigin f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Minima are at integer points (0, 0, ..., 0) shifted by hidden offset.
        # The periodicity is 2*pi in the cosine term, but the quadratic term
        # dominates globally. The "integer lattice" refers to the local minima
        # which are spaced by 2 units in the standard Rastrigin (since cos(2*pi*x)
        # has period 1, but the standard Rastrigin is often defined with 2*pi.
        # Wait, standard Rastrigin: f(x) = 10n + sum(x_i^2 - 10 cos(2 pi x_i)).
        # The minima are at x_i = 0, +/-2, +/-4, ... because cos(2 pi * 1) = 1,
        # so x=1 is a local min? No.
        # d/dx (x^2 - 10 cos(2 pi x)) = 2x + 20 pi sin(2 pi x).
        # At x=0: 0. At x=1: 2 + 0 = 2 != 0.
        # Actually, the global minimum is at 0.
        # Local minima are near integers.
        # Let's just do a coarse grid search over the box [-5, 5]^5.
        
        # Budget allocation:
        # 1. Coarse grid: ~100 evaluations
        # 2. Fine grid around best: ~100 evaluations
        # 3. NM Polish: ~300 evaluations
        
        # Coarse Grid
        # 5 dimensions. We can't do a full grid.
        # Let's try a Halton sequence for global exploration but with a twist:
        # We will use a deterministic Latin Hypercube or just a well-spaced Halton.
        # The current program uses Halton. The score dropped from 0.87 to 0.66.
        # Why? Maybe the NM polish was not good enough or the Halton didn't find the basin.
        
        # Let's try a different global search:
        # 1. Evaluate at origin (0,0,0,0,0).
        # 2. Evaluate at +/- 2 along each axis.
        # 3. Use these to guide a local search.
        
        # Actually, let's look at the "Inspiration 2" which scored 1.0063.
        # It likely exploited the structure.
        
        # Let's try a CMA-ES like approach but simple, or just a better Halton + NM.
        # The drop in score suggests the NM polish with delta=0.1 might have been too small
        # to escape local minima if the Halton point wasn't close enough, or the
        # budget was wasted on too many Halton points.
        
        # Strategy:
        # 1. Use 200 evaluations for a diverse set of points (Halton + some random).
        # 2. Take the top 5 points.
        # 3. Run NM polish on each with remaining budget / 5.
        
        # Let's refine the Halton phase to be more efficient and then do multi-start NM.
        
        halton_budget = 150
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        points = []
        values = []
        
        # Always evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        update_best(x0, f0)
        points.append(x0)
        values.append(f0)
        
        # Evaluate axis points +/- 2, +/- 4
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                if calls >= halton_budget or time.time() > deadline:
                    break
                x = np.zeros(dim)
                x[i] = val
                fx = func(x)
                calls += 1
                update_best(x, fx)
                points.append(x)
                values.append(fx)
        
        # Halton sequence for the rest of the coarse budget
        for i in range(1, halton_budget - calls + 1):
            if calls >= halton_budget or time.time() > deadline:
                break
            
            x = np.zeros(dim)
            for j in range(dim):
                f = 1.0
                r = 0.0
                idx = i + 1
                base = bases[j]
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                x[j] = r * 10.0 - 5.0
            
            fx = func(x)
            calls += 1
            update_best(x, fx)
            points.append(x)
            values.append(fx)
        
        # Sort points by value
        order = np.argsort(values)
        top_k = min(5, len(points))
        top_points = [points[i] for i in order[:top_k]]
        
        # Phase 2: Multi-start NM Polish
        remaining = budget - calls
        if remaining > 0 and top_points:
            budget_per_start = remaining // top_k
            for p in top_points:
                if calls >= budget or time.time() > deadline:
                    break
                x_polished, f_polished = self._nm_polish(
                    func, p, min(budget_per_start, budget - calls), deadline
                )
                if f_polished < best_f:
                    best_f = f_polished
                    best_x = x_polished

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
