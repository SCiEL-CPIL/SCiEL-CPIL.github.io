"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive ridge-follower with directional exploration for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        best_x_global = None
        best_f_global = np.inf
        calls_used = 0

        def eval_point(x):
            nonlocal calls_used, best_x_global, best_f_global
            if calls_used >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()
            return f

        # Strategy: Adaptive ridge-follower
        # f13 is a sharp ridge. The key insight is that the function has a narrow,
        # elongated basin of attraction. We need to:
        # 1. Find the ridge quickly with a diverse initial population
        # 2. Follow the ridge with directional exploration
        # 3. Refine the minimum along the ridge with local search
        
        # Phase 1: Diverse initial sampling to find the ridge
        # Sample 20 points in a grid-like pattern to cover the space
        n_init = 20
        for i in range(n_init):
            if calls_used >= budget or time.time() > deadline:
                break
            # Use a stratified sampling approach
            # Divide [-5, 5] into segments and sample one point per segment
            frac = (i + 0.5) / n_init
            x = np.zeros(dim)
            # Spread across dimensions using a simple pattern
            for d in range(dim):
                # Use different frequencies to cover the space
                phase = (i * (d + 1) + d * 0.3) % 1.0
                x[d] = -5 + 10 * (frac * 0.7 + 0.3 * phase)
            eval_point(x)

        if best_x_global is None:
            best_x_global = np.zeros(dim)
            eval_point(best_x_global)

        # Phase 2: Ridge following with directional exploration
        # The idea: maintain a set of candidate directions.
        # For each direction, try moving along it and see if we improve.
        # Keep the best directions and explore more along them.
        
        # Initialize directions with random unit vectors
        n_dirs = 8
        directions = []
        for i in range(n_dirs):
            d = np.random.randn(dim)
            d /= np.linalg.norm(d)
            directions.append(d)

        # Track the best step size per direction
        step_sizes = [0.5] * n_dirs

        # Main loop: alternate between ridge following and local refinement
        while calls_used < budget and time.time() < deadline:
            # Check if we should do a local refinement instead
            # If we've made good progress recently, refine locally
            if best_f_global < 10 and calls_used > budget * 0.3:
                # Do a local refinement step
                # Use a small random perturbation
                x = best_x_global + 0.1 * np.random.randn(dim)
                eval_point(x)
                continue

            # Try each direction
            improved = False
            for i in range(n_dirs):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Try moving along direction i
                x_new = best_x_global + step_sizes[i] * directions[i]
                f_new = eval_point(x_new)
                if f_new is None:
                    break
                
                if f_new < best_f_global:
                    # Improvement! Keep this direction and increase step size
                    improved = True
                    step_sizes[i] = min(step_sizes[i] * 1.5, 2.0)
                else:
                    # No improvement, try the opposite direction
                    x_new2 = best_x_global - step_sizes[i] * directions[i]
                    f_new2 = eval_point(x_new2)
                    if f_new2 is None:
                        break
                    
                    if f_new2 < best_f_global:
                        # Improvement in opposite direction
                        improved = True
                        directions[i] = -directions[i]
                        step_sizes[i] = min(step_sizes[i] * 1.5, 2.0)
                    else:
                        # No improvement, reduce step size
                        step_sizes[i] = max(step_sizes[i] * 0.7, 0.05)

            # If no direction improved, restart with new random directions
            if not improved:
                # Reset directions to new random ones
                for i in range(n_dirs):
                    d = np.random.randn(dim)
                    d /= np.linalg.norm(d)
                    directions[i] = d
                    step_sizes[i] = 0.5

                # Also try a larger random jump to escape local minima
                if calls_used < budget and time.time() < deadline:
                    x_jump = best_x_global + 2.0 * np.random.randn(dim)
                    eval_point(x_jump)

        # Phase 3: Final local refinement
        # Use the remaining budget for a focused local search
        if calls_used < budget and time.time() < deadline and best_x_global is not None:
            # Adaptive Gaussian perturbations with decreasing step size
            current_step = 0.3
            while calls_used < budget and time.time() < deadline:
                x = best_x_global + current_step * np.random.randn(dim)
                f = eval_point(x)
                if f is None:
                    break
                
                # If we improved, keep the step size; otherwise reduce it
                if f < best_f_global:
                    # Keep step size or increase slightly
                    current_step = min(current_step * 1.1, 1.0)
                else:
                    current_step *= 0.8

                # If step size is too small, we're done
                if current_step < 0.01:
                    break

        if best_x_global is None:
            best_x_global = np.zeros(dim)
            eval_point(best_x_global)

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
