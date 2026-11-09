"""Hybrid Phase Search: Random Exploration + Adaptive (1+1)-ES for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Phase 1: Random Exploration
        # Use 20% of the budget to sample random points to find a good basin.
        # This helps avoid starting in a bad local minimum or a region with
        # extreme ill-conditioning that traps a local optimizer.
        exploration_budget = max(10, self.budget // 5)
        
        # Ensure we have at least one evaluation
        if calls < self.budget:
            x0 = np.random.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            calls += 1
            best_x = x0.copy()
            best_f = fx0
            
            # Continue random sampling
            while calls < exploration_budget and time.time() < deadline:
                x_rand = np.random.uniform(-5, 5, self.dim)
                fx_rand = func(x_rand)
                calls += 1
                
                if fx_rand < best_f:
                    best_f = fx_rand
                    best_x = x_rand.copy()
                    
        # Phase 2: Adaptive (1+1)-ES Exploitation
        # Start from the best point found in Phase 1.
        x = best_x.copy()
        fx = best_f
        
        # Initial step size: 
        # If we found a very good point, we might be close to optimum, so start smaller.
        # Otherwise, start moderately large to navigate the landscape.
        # A heuristic: if best_f is still very high (e.g., > 100), we are far away.
        # But we don't know f_opt. Let's just use a robust initial sigma.
        # Given the box is [-5, 5], a sigma of 1.0 is reasonable.
        # We can scale it based on the improvement during exploration, but for simplicity, 
        # let's use a fixed initial value and let adaptation handle it.
        sigma = 1.0
        
        # Adaptation parameters for (1+1)-ES
        # Using a success rate based adaptation with a window
        window_size = 5
        successes = 0
        
        # To make it more robust, we can also use a "restart" mechanism if stagnation is detected.
        # But with only 400 calls left, we should focus on efficient local search.
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            noise = np.random.standard_normal(self.dim)
            y = x + sigma * noise
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            
            # Update local best for ES
            if fy < fx:
                x = y
                fx = fy
                successes += 1
            
            # Adapt sigma
            if calls % window_size == 0:
                success_rate = successes / window_size
                if success_rate > 0.5:
                    sigma *= 1.3
                elif success_rate < 0.5:
                    sigma *= 0.7
                # else: keep same
                successes = 0
                
                # Clip sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-8, 5.0)
                
                # Optional: Stagnation detection
                # If sigma becomes very small and we are not improving, we might be stuck.
                # But with a limited budget, restarting might be risky.
                # We can check if we haven't improved in the last X steps.
                # For now, the adaptive sigma should handle this.
                
        return best_x, best_f
# EVOLVE-BLOCK-END
