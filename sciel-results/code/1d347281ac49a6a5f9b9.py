"""Combined optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    (1+1)-ES with directional memory to follow ridges.
    This combines the robustness of the simple (1+1)-ES (Inspiration 1)
    with a mechanism to maintain momentum along the ridge,
    addressing the 'Sharp Ridge' topology better than isotropic search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        rng = np.random.default_rng(0)
        dim = self.dim

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = float(func(x))
        
        # Initial step size
        sigma = 2.0
        
        # Directional memory: stores the last successful direction
        # This helps in continuing along the ridge
        last_success_dir = np.zeros(dim)
        
        # Momentum factor: how much to reuse the last direction
        # A value between 0 and 1. 0 is pure random, 1 is pure greedy (bad).
        # 0.5 is a good balance.
        momentum = 0.5
        
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        while calls_made < self.budget:
            if time.time() > deadline:
                break

            # Generate random component
            z = rng.standard_normal(dim)
            
            # Blend random noise with previous successful direction
            # We need to normalize last_success_dir to have unit length before blending
            norm_dir = np.linalg.norm(last_success_dir)
            if norm_dir > 1e-8:
                last_success_dir_norm = last_success_dir / norm_dir
            else:
                last_success_dir_norm = np.zeros(dim)
            
            # The step is a combination of random and directional
            # step = sigma * ( (1-momentum)*z + momentum*last_success_dir_norm )
            # This maintains the shape of the distribution but biases it.
            step_dir = (1 - momentum) * z + momentum * last_success_dir_norm
            step = sigma * step_dir
            
            y = x + step
            y = np.clip(y, -5, 5)
            fy = float(func(y))
            calls_made += 1

            if fy < fx:
                # Success
                # Update directional memory
                # The direction of the step taken
                step_taken = y - x
                # If the step was clipped, the actual direction might be different
                # We use the actual displacement
                last_success_dir = step_taken
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Increase sigma slightly to explore further along the ridge
                sigma *= 1.1
            else:
                # Failure
                # Decrease sigma
                sigma *= 0.9
                # Optionally, decay the directional memory to allow new directions
                # But keeping it might help if we just took a bad step due to noise
                # Let's keep it, but the random component will eventually break it if it's bad.
                # Actually, if we fail, the direction might be wrong.
                # Let's decay it slightly.
                last_success_dir *= 0.9

            # Safety: if sigma becomes too small or too large
            if sigma < 1e-5:
                sigma = 0.5
            if sigma > 10.0:
                sigma = 10.0

        return best_x, best_f
# EVOLVE-BLOCK-END
