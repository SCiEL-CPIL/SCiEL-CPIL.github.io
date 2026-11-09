"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Multi-start Nelder-Mead approach:
  - BBOB f14 is unimodal and smooth under the hidden affine transform.
  - Use Sobol sequence for structured space-filling initialization.
  - Run multiple Nelder-Mead simplexes in parallel (multi-start), then
    converge the best simplex for final precision.
  - Nelder-Mead is effective on smooth unimodal functions and doesn't
    require gradient or Hessian information.
  - Sobol points provide deterministic, low-discrepancy coverage of the
    search space, improving initial exploration over pure random sampling.

Strategy:
  1. Generate Sobol points across [-5,5]^dim for diverse starting locations.
  2. Run Nelder-Mead from each start with a small evaluation budget.
  3. Select the best result and refine with a larger Nelder-Mead budget.
  4. Track best point throughout and return it.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f14 with hidden affine transform."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_points(self, n, dim, rng=None):
        """Generate n Sobol sequence points in [0,1)^dim, mapped to [-5,5]^dim."""
        # Use a simple quasi-random approach based on Halton-like low-discrepancy
        # For true Sobol, we'd need scipy.stats.qmc or implement Van der Corput
        # Here we use a simple deterministic low-discrepancy sequence
        
        # Van der Corput sequence for each dimension (Halton-like)
        def van_der_corput(n, base):
            seq = []
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                seq.append(r)
            return seq
        
        # Use different bases for each dimension (first primes)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        bases = primes[:dim]
        
        points = np.zeros((n, dim))
        for d in range(dim):
            seq = van_der_corput(n, bases[d])
            points[:, d] = seq
        
        # Map from [0,1) to [-5,5]
        points = points * 10.0 - 5.0
        return points

    def _nelder_mead(self, func, x0, budget, calls_counter, deadline, start_time, dim, init_step=1.0):
        """Run Nelder-Mead optimization with a given budget.
        
        Returns (best_x, best_f, calls_made).
        """
        # Initialize simplex
        simplex = [x0.copy()]
        f0 = float(func(x0))
        calls_counter[0] += 1
        simplex_f = [f0]
        
        # Add dim more vertices
        for i in range(dim):
            x_new = x0.copy()
            x_new[i] += init_step
            x_new = np.clip(x_new, -5.0, 5.0)
            f_new = float(func(x_new))
            calls_counter[0] += 1
            simplex.append(x_new)
            simplex_f.append(f_new)
        
        best_x = x0.copy()
        best_f = f0
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage
        max_iter = budget - (dim + 1)  # remaining budget
        
        for _ in range(max_iter):
            if calls_counter[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            # Sort simplex by fitness
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            
            # Check convergence
            f_range = simplex_f[-1] - simplex_f[0]
            if f_range < 1e-12:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = float(func(x_r))
            calls_counter[0] += 1
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = float(func(x_e))
                calls_counter[0] += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = float(func(x_c))
                    calls_counter[0] += 1
                    
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_i = float(func(simplex[i]))
                            calls_counter[0] += 1
                            simplex_f[i] = f_i
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = float(func(x_c))
                    calls_counter[0] += 1
                    
                    if f_c <= simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_i = float(func(simplex[i]))
                            calls_counter[0] += 1
                            simplex_f[i] = f_i
            
            # Update global best
            for i in range(len(simplex_f)):
                if simplex_f[i] < best_f:
                    best_f = simplex_f[i]
                    best_x = simplex[i].copy()
        
        return best_x, best_f, calls_counter[0]

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # 10s margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls_counter = [0]
        
        def elapsed():
            return time.time() - start
        
        def still_ok():
            return elapsed() < deadline and calls_counter[0] < budget
        
        # Generate Sobol points for multi-start
        n_starts = min(8, budget // 20)  # Up to 8 starts, each with ~20+ evals
        sobol_points = self._sobol_points(n_starts, dim)
        
        # Phase 1: Multi-start Nelder-Mead with small budgets
        # Allocate budget: ~30% for exploration, ~70% for exploitation
        explore_budget = int(budget * 0.3)
        exploit_budget = budget - explore_budget
        
        # Run Nelder-Mead from each Sobol point with a small budget
        per_start_budget = max(dim + 2, explore_budget // n_starts)
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_starts):
            if not still_ok():
                break
            
            x0 = sobol_points[i].copy()
            x0 = np.clip(x0, -5.0, 5.0)
            
            # Run Nelder-Mead with limited budget
            bx, bf, _ = self._nelder_mead(
                func, x0, per_start_budget, calls_counter, deadline, start, dim, init_step=1.0
            )
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        # If we haven't used much budget, do a final refinement
        if best_x is not None and still_ok() and calls_counter[0] < budget - 10:
            # Run a final Nelder-Mead from the best point with remaining budget
            remaining = budget - calls_counter[0]
            bx, bf, _ = self._nelder_mead(
                func, best_x, remaining, calls_counter, deadline, start, dim, init_step=0.5
            )
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        if best_x is None:
            return np.zeros(dim), np.inf
        
        return best_x, float(best_f)
