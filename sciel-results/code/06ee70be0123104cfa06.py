"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        
        best_x = None
        best_f = np.inf
        call_count = 0

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def cmaes_search(x0, budget_left):
            nonlocal best_x, best_f, call_count, deadline
            
            n = self.dim
            # CMA-ES parameters
            popsize = 4 + int(3 * np.log(n))
            popsize = min(popsize, 10) # Keep small for 5D to save budget
            c_sigma = (n + 1.3) / (n + 3 + 2 * (n / (n + 1)) ** 2)
            c_c = (4 + n / n) / (n + 4 + 2 * (n / (n + 1)) ** 2)
            mu = popsize // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_cov = (2 + 2.0 / (mu_eff + 1)) / (n + 3 + 2.0 / (mu_eff + 1))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_cov

            x = x0.copy()
            sigma = 1.0
            C = np.eye(n)
            Csqrt = np.eye(n)
            Cinv = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            
            # Initial evaluation
            if call_count >= self.budget:
                return
            fx = make_call(x)
            if fx is None:
                return
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # Pre-compute Cholesky update stuff? No, just do it inline.
            # For n=5, Cholesky is cheap.
            
            while call_count < self.budget and (time.time() < deadline - 2.0):
                # Sample
                # We need C^1/2. We can update C and then take cholesky.
                # Standard CMA-ES updates C first, then samples? 
                # Actually, standard is: sample from N(0, C), evaluate, update stats, update C.
                # Wait, the covariance matrix C represents the covariance of the search distribution.
                # So we sample y = C^1/2 * z.
                
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset or use identity
                    C = np.eye(n)
                    L = np.eye(n)
                    
                # Generate population
                ys = np.array([L @ rng.standard_normal(n) for _ in range(popsize)])
                xs = np.array([x + sigma * y for y in ys])
                
                # Evaluate
                fvals = []
                for i in range(popsize):
                    if call_count >= self.budget or time.time() > deadline - 2.0:
                        break
                    xi = np.clip(xs[i], -5, 5)
                    fi = make_call(xi)
                    if fi is None:
                        return
                    fvals.append(fi)
                    if np.isfinite(fi) and fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                
                if len(fvals) < popsize:
                    break
                    
                # Sort
                sorted_indices = np.argsort(fvals)
                xs_sorted = xs[sorted_indices]
                fvals_sorted = [fvals[i] for i in sorted_indices]
                ys_sorted = ys[sorted_indices]
                
                # Update mean
                x = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # Update step size control variables
                ymean = (x - x) / sigma # This is 0? No, x was updated.
                # Wait, the update order in standard CMA-ES:
                # 1. Sample
                # 2. Evaluate
                # 3. Sort
                # 4. Update mean
                # 5. Update ps, pc
                # 6. Update C
                # 7. Update sigma (if using CSA)
                
                # Let's stick to a simpler, robust implementation for 5D.
                # Actually, implementing full CMA-ES correctly is tricky.
                # A simpler alternative: (1, lambda)-ES with coordinate adaptation?
                # Or just use SciPy's minimize with multiple restarts?
                # SciPy's L-BFGS-B is great for smooth functions, but f18 is ill-conditioned.
                # However, L-BFGS-B uses gradients (numerical).
                # Let's try a hybrid: CMA-ES is robust.
                
                # Let's implement a simpler CMA-ES variant:
                # Just keep the covariance matrix update simple.
                
                # Actually, let's use a different strategy: 
                # "Adaptive Path" or just a very well-tuned (1+1)-ES with a 
                # "memory" of successful directions?
                
                # Given the complexity, let's try a different angle:
                # Local search using Nelder-Mead (Simplex) from multiple starts.
                # Nelder-Mead is derivative-free and handles ill-conditioning reasonably well
                # by deforming the simplex.
                
                pass # This implementation is getting messy. Let's pivot to Nelder-Mead.

        # PIVOT: Nelder-Mead Simplex Method with multiple restarts.
        # Nelder-Mead adapts the shape of the search region (simplex) which is
        # better for ill-conditioned problems than spherical (1+1)-ES.
        
        def nelder_mead(x0, budget_left):
            nonlocal best_x, best_f, call_count, deadline
            
            n = self.dim
            # Initialize simplex
            simplex = []
            fvals = []
            
            # Initial point
            x0 = np.clip(x0, -5, 5)
            f0 = make_call(x0)
            if f0 is None:
                return
            simplex.append(x0.copy())
            fvals.append(f0)
            if np.isfinite(f0) and f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            # Create other points in simplex
            for i in range(n):
                xi = x0.copy()
                # Step size in direction i
                if abs(x0[i]) > 0.1:
                    step = 0.05 * abs(x0[i])
                else:
                    step = 0.05
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = make_call(xi)
                if fi is None:
                    return
                simplex.append(xi)
                fvals.append(fi)
                if np.isfinite(fi) and fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                    
            # Parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            while call_count < self.budget and (time.time() < deadline - 2.0):
                # Sort simplex by function value
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                
                # Check for convergence
                if len(fvals) > 1:
                    diff = fvals[-1] - fvals[0]
                    if diff < 1e-10:
                        break
                        
                # Centroid of all but the worst point
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = make_call(x_r)
                if f_r is None:
                    return
                if np.isfinite(f_r) and f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = make_call(x_e)
                    if f_e is None:
                        return
                    if np.isfinite(f_e) and f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = make_call(x_c)
                    if f_c is None:
                        return
                    if np.isfinite(f_c) and f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = make_call(simplex[i])
                            if f_i is None:
                                return
                            fvals[i] = f_i
                            if np.isfinite(f_i) and f_i < best_f:
                                best_f = f_i
                                best_x = simplex[i].copy()
                        
                        # Check budget
                        if call_count >= self.budget:
                            return

        # Strategy:
        # 1. Evaluate origin.
        # 2. Run Nelder-Mead from origin.
        # 3. If budget remains, run Nelder-Mead from a few diverse points (e.g., random, corners).
        
        rng = np.random.default_rng(42)
        
        # Initial probe
        x0 = np.zeros(self.dim)
        f0 = make_call(x0)
        if f0 is not None:
            if np.isfinite(f0) and f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        # Run Nelder-Mead from origin
        if call_count < self.budget:
            nelder_mead(x0, self.budget - call_count)
            
        # If we have significant budget left, try other starts
        # Nelder-Mead is local, so diverse starts are good.
        starts = []
        # Random points
        for _ in range(3):
            starts.append(rng.uniform(-4, 4, self.dim))
        # Corners (scaled)
        for i in range(self.dim):
            pt = np.zeros(self.dim)
            pt[i] = 4.0
            starts.append(pt)
            pt[i] = -4.0
            starts.append(pt)
            
        for s in starts:
            if call_count >= self.budget or time.time() > deadline - 5.0:
                break
            # Simple check: if this start is far from best, maybe run
            # Just run it if we have budget
            nelder_mead(s, self.budget - call_count)
            
        if best_x is None:
            # Fallback
            x = np.zeros(self.dim)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
