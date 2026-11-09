"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma via success-rate window and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        rng = np.random.default_rng(0)
        
        # Initialize at origin (center of box)
        x = np.zeros(self.dim)
        fx = func(x)
        sigma = 3.0
        calls_made = 1
        
        # Adaptive sigma tracking
        window_size = 15
        window = []
        
        # Early progress: quick local scan if budget allows
        # Try a few different initial points to escape bad local minima
        # But keep it simple: just run the (1+1)-ES with good adaptation
        
        for _ in range(self.budget - 1):
            # Check deadline before each evaluation
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                window.append(1)
            else:
                window.append(0)
            
            # Adapt sigma based on success rate in recent window
            if len(window) >= window_size:
                success_rate = sum(window[-window_size:]) / window_size
                # 1/5 rule: aim for ~20% success rate
                if success_rate > 0.25:
                    sigma *= 1.3
                elif success_rate < 0.15:
                    sigma *= 0.75
                else:
                    sigma *= 1.0
                window = window[-window_size:]
            else:
                # Early on, use simpler adaptation
                if len(window) > 5:
                    recent_success = sum(window) / len(window)
                    if recent_success > 0.3:
                        sigma *= 1.15
                    elif recent_success < 0.1:
                        sigma *= 0.85
        
        return x, fx
# EVOLVE-BLOCK-END
