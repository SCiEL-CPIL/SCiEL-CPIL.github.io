"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead with (1+1)-ES refinement for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = float('inf')

        # Strategy: 
        # 1. Multi-start Nelder-Mead (3 starts, ~100 evals each)
        # 2. Polish the best with (1+1)-ES using remaining budget
        
        num_starts = 3
        evals_per_start = int(budget * 0.3)  # ~150 evals per start
        
        candidates = []
        
        for i in range(num_starts):
            if time.time() > deadline:
                break
                
            # Random start in [-5, 5]^dim
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            candidates.append((x0, f0))
            
            # Run Nelder-Mead from this start
            x_opt, f_opt = self._nelder_mead(
                func, x0, f0, 
                budget=evals_per_start, 
                rng=rng,
                deadline=deadline
            )
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Polish the best candidate with (1+1)-ES using remaining budget
        if best_x is not None:
            remaining_budget = budget - (num_starts * evals_per_start)
            if remaining_budget > 10 and time.time() < deadline:
                x_polished, f_polished = self._es_refine(
                    func, best_x, best_f,
                    budget=remaining_budget,
                    rng=rng,
                    deadline=deadline
                )
                if f_polished < best_f:
                    best_f = f_polished
                    best_x = x_polished

        if best_x is None:
            # Fallback if no calls were made
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            best_x = x
            best_f = fx

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, rng, deadline, initial_step=1.0):
        """
        Custom Nelder-Mead implementation with budget tracking and deadline guard.
        """
        dim = len(x0)
        n = dim + 1  # Number of vertices
        
        # Initialize simplex
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        f_values = np.zeros(n)
        f_values[0] = f0
        
        evals_used = 1  # Count the initial evaluation
        
        for i in range(1, n):
            if time.time() > deadline or evals_used >= budget:
                break
            simplex[i] = x0.copy()
            simplex[i][i-1] += initial_step
            f_values[i] = func(simplex[i])
            evals_used += 1
        
        # Order vertices by function value
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        
        best_f = f_values[0]
        
        # Main loop
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Check for convergence
            if np.max(f_values) - np.min(f_values) < 1e-12:
                break
                
            # Calculate centroid of all but the worst vertex
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            evals_used += 1
            
            if f_values[0] <= fr < f_values[-2]:
                # Accept reflection
                simplex[-1] = xr
                f_values[-1] = fr
            elif fr < f_values[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                evals_used += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    # Outside contraction
                    rho = 0.5
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    rho = 0.5
                    xc = centroid - rho * (centroid - simplex[-1])
                
                fc = func(xc)
                evals_used += 1
                
                if fc < min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        if evals_used >= budget or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        f_values[i] = func(simplex[i])
                        evals_used += 1
                    
                    if evals_used >= budget or time.time() > deadline:
                        break
            
            # Re-order
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
            
            # Stagnation check: if no improvement for many iterations, break
            # Simple heuristic: if the best hasn't improved in 20 iterations, stop
            # (This is a rough estimate; we don't track history here for simplicity)
        
        return simplex[0], f_values[0]

    def _es_refine(self, func, x, fx, budget, rng, deadline):
        """
        (1+1)-ES refinement with adaptive step size.
        """
        dim = len(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 0.5  # Start with a smaller step for refinement
        
        calls_made = 0
        successes = 0
        total = 0
        window_size = 10
        
        for i in range(budget):
            if time.time() >= deadline:
                break
            
            # Generate trial point
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls_made += 1
            
            # Update best
            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy
            
            # 1/5 rule for step size adaptation
            if fy < fx:
                successes += 1
                x = y
                fx = fy
            else:
                pass
            
            total += 1
            
            # Adapt sigma every window_size steps
            if total % window_size == 0:
                rate = successes / total
                if rate > 1.0 / 5.0:
                    sigma *= 1.3
                elif rate < 1.0 / 5.0:
                    sigma *= 0.8
                sigma = np.clip(sigma, 1e-6, 2.0)
                successes = 0
                total = 0
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
