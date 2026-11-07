"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+8)-ES with adaptive step size for BBOB f13 sharp ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 320s margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Phase 1: Global multi-start sampling
        n_init = min(50, budget // 10)
        if n_init < 10:
            n_init = min(10, budget)
        
        candidates = []
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x.copy()))
            if time.time() > deadline - 30:
                break

        # Sort by fitness and take top 5
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))
        
        # If we barely have budget, just return best
        remaining = budget - len(candidates)
        if remaining <= 0 or top_k == 0:
            best_x = candidates[0][1] if candidates else np.zeros(dim)
            best_f = candidates[0][0] if candidates else float('inf')
            return best_x, best_f

        # Phase 2: Run (1+8)-ES from each top starting point
        # Allocate remaining budget evenly among the top_k starts
        budget_per_start = max(1, remaining // top_k)
        
        global_best_x = candidates[0][1].copy()
        global_best_f = candidates[0][0]

        for s in range(top_k):
            x = candidates[s][1].copy()
            fx = candidates[s][0]
            sigma = 1.5  # Initial step size for sharp ridge
            
            # Track success rate for adaptive sigma
            successes = 0
            total = 0
            n_offspring = 8
            
            for _ in range(budget_per_start):
                # Check deadline
                if time.time() > deadline:
                    break
                
                # Generate offspring
                best_off_x = None
                best_off_f = float('inf')
                
                for _ in range(n_offspring):
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    if fy < best_off_f:
                        best_off_f = fy
                        best_off_x = y.copy()
                    
                    # Check deadline inside offspring loop
                    if time.time() > deadline:
                        break
                
                if best_off_x is None:
                    break
                
                total += 1
                if best_off_f < fx:
                    successes += 1
                    x = best_off_x
                    fx = best_off_f
                    # Update global best
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                
                # Adaptive sigma based on success rate (target ~20-25%)
                if total >= 4:
                    success_rate = successes / total
                    if success_rate > 0.25:
                        sigma *= 1.2
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    # Keep success window small for responsiveness
                    successes = max(0, successes - 2)
                    total = max(0, total - 4)
                
                # Ensure sigma stays reasonable
                sigma = np.clip(sigma, 0.01, 10.0)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
