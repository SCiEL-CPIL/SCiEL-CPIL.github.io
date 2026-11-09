import numpy as np
import time
import random

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to call func safely
        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError
            if calls_made >= budget:
                raise MemoryError # Custom exception to stop loop
            calls_made += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Search with Simulated Annealing
        # We split the budget: ~70% for SA, ~30% for Polish
        # SA is good for exploring the "step" structure and escaping plateaus
        n_restarts = 10
        # Rough allocation: (budget * 0.7) / 10 per restart
        # Each restart uses 1 init + (steps) evaluations
        # Let's aim for ~30-40 steps per restart to keep total < 350
        budget_per_restart = 35 
        
        # SA Parameters
        # Start with a large step to cover the space
        # The function is step ellipsoidal, so large steps can jump between plateaus
        T_initial = 10.0
        T_final = 0.1
        alpha = 0.95 # Cooling rate
        sigma_initial = 2.0
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            x = rng.uniform(-5, 5, n)
            try:
                fx = safe_call(x)
            except (TimeoutError, MemoryError):
                break
            
            # SA Loop
            T = T_initial
            sigma = sigma_initial
            
            # Adaptive sigma: if we get stuck (no improvement in last K steps), reduce sigma or change direction?
            # For SA, we just keep going.
            
            for i in range(budget_per_restart):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 50: # Reserve 50 for polish
                    break
                
                # Generate neighbor
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                
                try:
                    fy = safe_call(y)
                except (TimeoutError, MemoryError):
                    break
                
                delta = fy - fx
                
                # Acceptance probability
                if delta < 0:
                    accept = True
                else:
                    # Avoid overflow in exp
                    if T < 1e-10:
                        accept = False
                    else:
                        accept = rng.random() < np.exp(-delta / T)
                
                if accept:
                    x = y
                    fx = fy
                    # Slightly increase sigma on success to explore more? 
                    # Or keep it constant? Standard SA keeps sigma constant or adapts.
                    # Let's adapt: if accepted, increase sigma slightly to explore further.
                    sigma *= 1.05
                    sigma = min(sigma, 5.0) # Cap at box size
                else:
                    # If rejected, shrink sigma to refine local search
                    sigma *= 0.95
                    sigma = max(sigma, 0.1)
                
                T *= alpha

        # Phase 2: Local Polish (Nelder-Mead or simple coordinate descent)
        # Use remaining budget for a precise local search around best_x
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and best_x is not None and time.time() < deadline:
            # Simple adaptive local search
            # We can use a simple gradient-free method like Powell or just fine-tuning SA
            # Let's do a focused SA with much lower T and smaller sigma
            x = best_x.copy()
            try:
                fx = safe_call(x) # Re-evaluate to ensure fx matches best_f? 
                                  # Actually best_f is already tracked. 
                                  # But x might have drifted if best_x was updated 
                                  # but x was not. Wait, best_x is updated when fx < best_f.
                                  # So if best_x was updated, x is likely best_x or close.
                                  # Let's just start from best_x.
            except (TimeoutError, MemoryError):
                pass
            
            # Polish parameters
            T = 0.1
            alpha = 0.9
            sigma = 0.5
            min_sigma = 1e-5
            
            for i in range(remaining_budget):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                
                try:
                    fy = safe_call(y)
                except (TimeoutError, MemoryError):
                    break
                
                delta = fy - fx
                
                if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
                    x = y
                    fx = fy
                    sigma *= 1.01
                else:
                    sigma *= 0.9
                    if sigma < min_sigma:
                        break # Convergence
                
                T *= alpha

        return best_x, best_f
