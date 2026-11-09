"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for Rastrigin f3 using grid search and Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_polish(self, func, x0, budget_left, deadline, rng):
        """Nelder-Mead local refinement with budget limit."""
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Build initial simplex
        simplex = [x.copy()]
        simplex_f = [fx]
        
        delta = 0.5
        for i in range(dim):
            xi = x.copy()
            # Adjust step if near boundary
            if xi[i] + delta > 5.0:
                delta = 0.1
            if xi[i] - delta < -5.0:
                delta = 0.1
            xi[i] += delta
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            fxi = func(xi)
            calls += 1
            simplex_f.append(fxi)
            if fxi < best_f:
                best_f = fxi
                best_x = xi.copy()
            if calls >= budget_left or time.time() > deadline:
                break
        
        if len(simplex) < 2:
            return best_x, best_f, calls
        
        # Nelder-Mead iterations
        alpha, gamma, rho = 1.0, 2.0, 0.5
        max_iter = min(100, (budget_left - calls) // (dim + 1))
        
        for _ in range(max_iter):
            if time.time() > deadline or calls >= budget_left:
                break
            
            # Sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
                
            # Check convergence
            if simplex_f[-1] - simplex_f[0] < 1e-10:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
            
            if calls >= budget_left or time.time() > deadline:
                break
            
            if simplex_f[0] <= fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            elif fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                
                if calls >= budget_left or time.time() > deadline:
                    break
                
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                    
                if calls >= budget_left or time.time() > deadline:
                    break
                    
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()
                        if calls >= budget_left or time.time() > deadline:
                            break
                    if calls >= budget_left or time.time() > deadline:
                        break
        return best_x, best_f, calls

    def _grid_search_1d(self, func, x_fixed, axis, start, stop, step, budget_left, deadline, total_calls_holder):
        """Perform a 1D grid search along a specific axis, keeping other coordinates fixed."""
        dim = self.dim
        best_x = None
        best_f = np.inf
        
        n_steps = int((stop - start) / step) + 1
        # Cap steps to avoid excessive calls if step is too small
        # We assume the caller passes reasonable steps
        
        for k in range(n_steps):
            if total_calls_holder[0] >= budget_left or time.time() > deadline:
                break
            
            x = x_fixed.copy()
            x[axis] = start + k * step
            x = np.clip(x, -5, 5)
            
            f = func(x)
            total_calls_holder[0] += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin
        
        total_calls = [0]
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Phase 1: Coarse Grid Search
        # Rastrigin has period 2*pi ~ 6.28. In [-5, 5], we have roughly 1.6 periods.
        # A coarse grid with step 1.0 should capture the basin of attraction for each local minimum.
        # We search each dimension independently, starting from 0.
        
        # Initialize with origin
        x = np.zeros(dim)
        f = func(x)
        total_calls[0] += 1
        best_x = x.copy()
        best_f = f
        
        # Coarse search along each axis
        # We do a simple axis-aligned search.
        # To be efficient, we can do a "hill climbing" style grid search or just scan.
        # Scanning is safer for multimodal functions.
        
        # Grid points: -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 (11 points)
        coarse_points = np.arange(-5, 5.1, 1.0)
        
        for axis in range(dim):
            if total_calls[0] >= budget or time.time() > deadline:
                break
                
            x_fixed = best_x.copy()
            # We search the whole range for this axis, keeping others at best_x
            # This is O(11 * dim) calls = 55 calls. Very cheap.
            for val in coarse_points:
                if total_calls[0] >= budget or time.time() > deadline:
                    break
                x = x_fixed.copy()
                x[axis] = val
                x = np.clip(x, -5, 5)
                f = func(x)
                total_calls[0] += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # If we found a better point, we might want to re-scan other axes?
            # For now, single pass is a good start. But Rastrigin is separable in structure.
            # Let's do a second pass of coarse search with the updated best_x.
        
        # Second pass of coarse search to refine
        for axis in range(dim):
            if total_calls[0] >= budget or time.time() > deadline:
                break
            x_fixed = best_x.copy()
            for val in coarse_points:
                if total_calls[0] >= budget or time.time() > deadline:
                    break
                x = x_fixed.copy()
                x[axis] = val
                x = np.clip(x, -5, 5)
                f = func(x)
                total_calls[0] += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
        # Phase 2: Fine Grid Search
        # Now we are near a local minimum. The basin is roughly width 1.
        # We do a finer grid around the current best point.
        # Step 0.25 -> 8 points per axis.
        # We search in a window of [-0.5, 0.5] around best_x for each axis.
        
        fine_points_1 = np.arange(-0.5, 0.51, 0.25) # -0.5, -0.25, 0, 0.25, 0.5
        
        for axis in range(dim):
            if total_calls[0] >= budget or time.time() > deadline:
                break
            x_fixed = best_x.copy()
            for val in fine_points_1:
                if total_calls[0] >= budget or time.time() > deadline:
                    break
                x = x_fixed.copy()
                x[axis] = x_fixed[axis] + val
                x = np.clip(x, -5, 5)
                f = func(x)
                total_calls[0] += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
        # Phase 3: Very Fine Grid Search
        # Step 0.05 -> 21 points per axis.
        # Window [-0.1, 0.1]
        
        fine_points_2 = np.arange(-0.1, 0.11, 0.05)
        
        for axis in range(dim):
            if total_calls[0] >= budget or time.time() > deadline:
                break
            x_fixed = best_x.copy()
            for val in fine_points_2:
                if total_calls[0] >= budget or time.time() > deadline:
                    break
                x = x_fixed.copy()
                x[axis] = x_fixed[axis] + val
                x = np.clip(x, -5, 5)
                f = func(x)
                total_calls[0] += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
        # Phase 4: Nelder-Mead Polish
        if total_calls[0] < budget and time.time() < deadline:
            polish_budget = min(100, budget - total_calls[0])
            if polish_budget > 10:
                rng = np.random.default_rng(42)
                x_polished, f_polished, calls_polish = self._nm_polish(
                    func, best_x, polish_budget, deadline, rng
                )
                total_calls[0] += calls_polish
                if f_polished < best_f:
                    best_f = f_polished
                    best_x = x_polished.copy()
                    
        # Phase 5: If budget remains, try another NM restart with slight perturbation
        if total_calls[0] < budget and time.time() < deadline:
            polish_budget = min(50, budget - total_calls[0])
            if polish_budget > 10:
                rng = np.random.default_rng(123)
                # Perturb best_x slightly
                x_perturbed = best_x + rng.normal(0, 0.1, dim)
                x_perturbed = np.clip(x_perturbed, -5, 5)
                x_polished2, f_polished2, calls_polish2 = self._nm_polish(
                    func, x_perturbed, polish_budget, deadline, rng
                )
                total_calls[0] += calls_polish2
                if f_polished2 < best_f:
                    best_f = f_polished2
                    best_x = x_polished2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
