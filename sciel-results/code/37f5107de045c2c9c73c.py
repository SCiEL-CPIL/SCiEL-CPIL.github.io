import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 325
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        def check_deadline():
            return time.time() < deadline - 3.0

        def evaluate(x):
            nonlocal eval_count, best_f, best_x
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial sampling to identify promising basins
        # f11 (Discus) is multimodal with many local optima
        n_init = min(40, max(15, int(self.budget * 0.08)))
        
        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = evaluate(x)
        
        if eval_count >= self.budget or not check_deadline():
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead
        # The Discus function has a rotated, elongated basin structure.
        # Nelder-Mead can navigate this more efficiently than isotropic methods
        # because it adapts the simplex shape to the local geometry.
        
        remaining = self.budget - eval_count
        
        # Collect top points from initial sampling for multi-start
        # We'll use the best few points as starting points for Nelder-Mead
        # But we only tracked the single best, so let's do a small local search
        # from the best point and a few random restarts.
        
        # Strategy: 
        # 1. Run Nelder-Mead from the current best point
        # 2. If budget remains, run Nelder-Mead from a few other random points
        # 3. Finally, polish the best point with a very fine Nelder-Mead
        
        nm_starts = []
        
        # Start 1: the current best
        nm_starts.append(best_x.copy())
        
        # Start 2-5: a few random points near the best (local exploration)
        for _ in range(4):
            if remaining <= 10 or not check_deadline():
                break
            x_near = best_x + rng.normal(0, 1.0, self.dim)
            x_near = np.clip(x_near, -5.0, 5.0)
            nm_starts.append(x_near)
        
        # Start 6-8: fully random points to escape local minima
        for _ in range(3):
            if remaining <= 10 or not check_deadline():
                break
            x_rand = rng.uniform(-5.0, 5.0, self.dim)
            nm_starts.append(x_rand)
        
        # Run Nelder-Mead from each start point
        for start_idx, x_start in enumerate(nm_starts):
            if eval_count >= self.budget or not check_deadline():
                break
            
            # Allocate budget proportionally, but give more to later starts if they improve
            # Simple allocation: use up to 40% of remaining budget per start
            # But cap it to ensure we try multiple starts
            max_nm_evals = min(remaining, max(20, int(remaining * 0.4)))
            
            x_nm, f_nm, evals_used = self._nelder_mead(func, x_start, max_nm_evals, evaluate, check_deadline)
            
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
            
            remaining = self.budget - eval_count
        
        # Final polish: if we have budget left, do a very fine Nelder-Mead from the best
        if eval_count < self.budget and check_deadline() and remaining > 5:
            max_polish = min(remaining, 50)
            x_polish, f_polish, evals_used = self._nelder_mead(func, best_x, max_polish, evaluate, check_deadline, initial_step=0.05)
            
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
        
        return best_x, best_f

    def _nelder_mead(self, func, x0, max_evals, evaluate, check_deadline, initial_step=0.5):
        """
        Nelder-Mead simplex method.
        Returns (best_x, best_f, evals_used)
        """
        dim = self.dim
        
        # Initialize simplex
        # Vertex 0: the starting point
        # Vertices 1..dim: x0 + initial_step * e_i
        simplex = [x0.copy()]
        for i in range(dim):
            v = x0.copy()
            v[i] += initial_step
            v = np.clip(v, -5.0, 5.0)
            simplex.append(v)
        
        # Evaluate initial simplex
        simplex_f = []
        for v in simplex:
            fv = evaluate(v)
            simplex_f.append(fv)
        
        best_x = x0.copy()
        best_f = min(simplex_f)
        
        # Reflection, expansion, contraction, shrink parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        evals_used = 0
        max_iter = max_evals * 2  # safety cap on iterations
        
        for iteration in range(max_iter):
            if evals_used >= max_evals or not check_deadline():
                break
            
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Track best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
            
            # Compute centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = evaluate(xr)
            evals_used += 1
            
            if fr < simplex_f[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = evaluate(xe)
                evals_used += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept reflection
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
                xc = np.clip(xc, -5.0, 5.0)
                fc = evaluate(xc)
                evals_used += 1
                
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = evaluate(simplex[i])
                        evals_used += 1
                        if evals_used >= max_evals or not check_deadline():
                            break
            
            if evals_used >= max_evals or not check_deadline():
                break
            
            # Check for convergence: if the simplex is very small, stop
            if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                break
        
        return best_x, best_f, evals_used
