"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with 1/5 success rule for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit

        # Initialize
        rng = np.random.default_rng(42)
        
        # Start at origin, which is a reasonable heuristic for BBOB instances
        # though they have shifts. A uniform start might be safer, but 
        # origin is often close or the shift is small enough that a fast 
        # adaptive step size finds the way. 
        # Actually, BBOB instances have hidden shifts in [-4, 4]. 
        # Starting at 0 is good.
        
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Initial step size
        sigma = 1.0
        # 1/5 rule parameters
        # If success rate > 1/5, increase sigma; else decrease.
        # Standard: sigma *= exp(1/5 * (success_rate - 1/5)) roughly
        # Or simpler: keep a window of last N steps.
        
        # Let's use a simpler adaptive rule:
        # Track success in last 10 steps.
        # If successes > 2, sigma *= 1.05
        # If successes < 2, sigma *= 0.95
        
        window_size = 10
        successes = []

        while calls < budget and time.time() < deadline:
            # Generate candidate
            # Use standard normal perturbation
            dx = rng.standard_normal(dim)
            x_candidate = x + sigma * dx
            
            # Clip to box [-5, 5]
            x_candidate = np.clip(x_candidate, -5, 5)
            
            fc = func(x_candidate)
            calls += 1
            
            # Check for improvement
            is_success = fc < fx
            
            if is_success:
                x = x_candidate
                fx = fc
                if fc < best_f:
                    best_f = fc
                    best_x = x.copy()
                successes.append(1)
            else:
                successes.append(0)
            
            # Keep window size constant
            if len(successes) > window_size:
                successes.pop(0)
            
            # Adapt sigma based on success rate
            if len(successes) == window_size:
                success_rate = sum(successes) / window_size
                # 1/5 rule: target success rate is 1/5 (0.2)
                # If success rate is higher, we are moving too slowly -> increase sigma
                # If success rate is lower, we are moving too fast -> decrease sigma
                # Simple multiplicative update
                if success_rate > 0.2:
                    sigma *= 1.05
                elif success_rate < 0.2:
                    sigma *= 0.95
                
                # Bound sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
