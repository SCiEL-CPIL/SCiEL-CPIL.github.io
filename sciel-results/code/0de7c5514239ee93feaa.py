# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton-sequence coarse-to-fine search with Nelder-Mead polish.
    
    Phase 1: Systematic Halton sequence sampling over the box
    Phase 2: Adaptive refinement around the best point found
    Phase 3: Nelder-Mead local optimization from the best point
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295

        def halton(index, base):
            """Generate the index-th Halton sequence value in base."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            """Generate the n-th point in the d-dimensional Halton sequence."""
            point = np.zeros(dim)
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(dim):
                point[i] = halton(n + 1, primes[i])
            return point

        def nelder_mead(func, x0, budget_nm, deadline):
            """Run Nelder-Mead simplex method with limited budget."""
            calls = 0
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [func(x0)]
            calls += 1

            for i in range(dim):
                xi = x0.copy()
                step = 0.3
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(func(xi))
                calls += 1

            # Order by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            best_x = simplex[0].copy()
            best_f = fvals[0]

            while calls < budget_nm:
                if time.time() > deadline:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if calls >= budget_nm:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if calls >= budget_nm:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if calls >= budget_nm:
                        break
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls >= budget_nm:
                                break
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1

                # Reorder
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

            return best_x, best_f, calls

        total_calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Coarse Halton sequence sampling
        # Use about 40% of budget for coarse search
        coarse_budget = int(budget * 0.4)
        
        # Sample Halton points over [-5, 5]^dim
        n_coarse = min(coarse_budget, 100)
        
        for n in range(n_coarse):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break
            
            h = halton_point(n, dim)
            x = np.clip(-5 + 10 * h, -5, 5)
            fx = func(x)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Adaptive refinement around best point
        # Use about 35% of budget for refinement
        refine_budget = int(budget * 0.35)
        
        # Start with a moderate radius and shrink
        radius = 3.0
        rng = np.random.default_rng(42)
        
        while total_calls < budget - refine_budget:
            if time.time() > deadline:
                break
            
            # Generate points on a sphere of radius 'radius' around best_x
            n_refine = min(20, budget - total_calls)
            
            for _ in range(n_refine):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Random direction
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                
                x = np.clip(best_x + radius * direction, -5, 5)
                fx = func(x)
                total_calls += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Shrink radius
            radius *= 0.5
            
            if radius < 0.01:
                break
        
        # Phase 3: Nelder-Mead polish from best point
        # Use remaining budget
        nm_budget = budget - total_calls
        if nm_budget > dim + 2:
            if time.time() < deadline:
                x_nm, f_nm, calls_nm = nelder_mead(func, best_x, nm_budget, deadline)
                total_calls += calls_nm
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
