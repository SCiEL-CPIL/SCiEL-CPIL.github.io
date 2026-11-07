"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with 1/5 success rule and final local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, f0, dim, max_calls, deadline):
        """Nelder-Mead simplex method for local refinement."""
        # Initialize simplex
        simplex = [x0.copy()]
        fx_list = [f0]
        calls_used = 1  # x0 already evaluated
        
        # Initial step size for simplex vertices
        step = 0.5
        
        # Create remaining vertices
        for i in range(dim):
            if time.time() >= deadline:
                break
            if calls_used >= max_calls:
                break
            vi = x0.copy()
            if vi[i] < 5.0 - step:
                vi[i] += step
            else:
                vi[i] -= step
            vi = np.clip(vi, -5, 5)
            fi = func(vi)
            calls_used += 1
            simplex.append(vi)
            fx_list.append(fi)
        
        if len(simplex) < 2:
            return simplex[0], fx_list[0], calls_used
        
        # Nelder-Mead iterations
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        max_iter = max_calls - calls_used
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            if calls_used >= max_calls:
                break
            
            # Sort simplex by function value
            order = np.argsort(fx_list)
            simplex = [simplex[i] for i in order]
            fx_list = [fx_list[i] for i in order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1
            
            if fr < fx_list[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_used += 1
                if fe < fr:
                    simplex[-1] = xe
                    fx_list[-1] = fe
                else:
                    simplex[-1] = xr
                    fx_list[-1] = fr
            elif fr < fx_list[-2]:
                simplex[-1] = xr
                fx_list[-1] = fr
            else:
                # Contraction
                if fr < fx_list[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1
                
                if fc <= min(fr, fx_list[-1]):
                    simplex[-1] = xc
                    fx_list[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_used >= max_calls:
                            break
                        fi = func(simplex[i])
                        calls_used += 1
                        fx_list[i] = fi
        
        # Return best point
        best_idx = np.argmin(fx_list)
        return simplex[best_idx], fx_list[best_idx], calls_used

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        budget = self.budget
        dim = self.dim
        
        # Reserve calls for final Nelder-Mead refinement
        reserve = max(50, budget // 5)  # ~100 calls reserved for NM
        
        # Portfolio structure: multiple independent (1+1)-ES runs
        n_runs = 3
        calls_per_run = (budget - reserve) // n_runs
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Diverse initial sigmas to cover different scales
        initial_sigmas = [3.0, 2.0, 1.5]
        
        for run_idx in range(n_runs):
            if time.time() >= deadline - 30:
                break
            if total_calls >= budget - reserve:
                break
            
            # Each run starts from a different random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            sigma = initial_sigmas[run_idx]
            
            # (1+1)-ES with 1/5 success rule for this run
            for _ in range(calls_per_run - 1):
                if time.time() >= deadline - 30:
                    break
                if total_calls >= budget - reserve:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                
                # Keep sigma within reasonable bounds
                sigma = float(np.clip(sigma, 0.005, 10.0))
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Final local refinement using Nelder-Mead
        if best_x is not None and total_calls < budget - 5 and time.time() < deadline - 10:
            remaining_calls = budget - total_calls - 5
            nm_x, nm_f, nm_calls = self._nelder_mead(
                func, best_x.copy(), best_f, dim, remaining_calls, deadline - 5
            )
            total_calls += nm_calls
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x
        
        if best_x is None:
            # Fallback: should not happen, but be safe
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
