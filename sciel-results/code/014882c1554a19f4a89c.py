"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global grid + (1+1)-ES with 1/5-rule adaptation.

    Strategy:
      1. Coarse grid exploration to identify promising regions (early AOCC gains).
      2. Multiple (1+1)-ES restarts from best grid points and random starts.
      3. Simple 1/5-rule step-size adaptation (1.5x on success, 0.87x on failure).
      4. Final local refinement around the best point found using a smaller sigma.
      5. Explicit wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse grid exploration for early AOCC gains
        # Use a small grid to sample the domain systematically
        # 3 points per dimension in [-5, 5] gives 3^5 = 243 points
        # But that's too many for budget=500. Use 2 points per dim = 2^5 = 32 points
        # Or a smarter approach: sample strategically
        
        # Let's use a hybrid: a few strategic points + random sampling
        # First, evaluate the center and corners for quick baseline
        grid_points = []
        
        # Center
        grid_points.append(np.zeros(dim))
        
        # Midpoints of edges
        for d in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[d] = sign * 2.5
                grid_points.append(p)
        
        # Random points for diversity
        for _ in range(10):
            grid_points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate grid points
        for gp in grid_points:
            if call_count[0] >= budget - 50:
                break
            if time.time() > deadline - 30:
                break
            
            gp = np.clip(gp, -5.0, 5.0)
            f = make_call(gp)
            if f is None:
                break
            update_best(gp, f)
        
        # Phase 2: Portfolio of (1+1)-ES restarts from best points and random starts
        reserve = max(30, budget // 8)
        
        # Use best grid point as one restart, plus random restarts
        n_restarts = 4
        alloc_per_restart = (budget - reserve - call_count[0]) // n_restarts if call_count[0] < budget - reserve else 0
        
        # Restart points: best found + random
        restart_inits = []
        if best_x is not None:
            restart_inits.append(best_x.copy())
        
        # Add random restarts
        while len(restart_inits) < n_restarts:
            restart_inits.append(rng.uniform(-5, 5, dim))
        
        # Diverse initial sigmas
        initial_sigmas = [0.5, 2.0, 1.5, 3.0]
        
        for r in range(n_restarts):
            if call_count[0] >= budget - reserve:
                break
            if time.time() > deadline - 20:
                break
            
            x = np.clip(restart_inits[r].copy(), -5.0, 5.0)
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)
            
            sigma = initial_sigmas[r % len(initial_sigmas)]
            
            # Simple (1+1)-ES loop with 1/5 rule adaptation
            for _ in range(alloc_per_restart):
                if call_count[0] >= budget - reserve:
                    break
                if time.time() > deadline - 20:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                
                # Keep sigma within bounds
                sigma = float(np.clip(sigma, 0.01, 10.0))
        
        # Phase 3: Final local refinement around the best point
        if best_x is not None and call_count[0] < budget - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining_calls = budget - call_count[0] - 5
            
            for _ in range(remaining_calls):
                if call_count[0] >= budget - 5:
                    break
                if time.time() > deadline - 5:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                
                sigma = float(np.clip(sigma, 0.001, 1.0))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
