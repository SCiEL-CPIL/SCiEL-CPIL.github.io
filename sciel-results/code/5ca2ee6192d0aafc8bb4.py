import numpy as np
import time

class Optimizer:
    """
    Optimizer using a deterministic lattice search followed by adaptive local refinement.
    1. A 3^5 grid search over [-5, 5]^5 (243 evaluations) ensures global coverage of the
       multimodal Weierstrass landscape.
    2. A variable-step (1+1)-ES refines the best point found, using the remaining budget
       (up to 257 evaluations) to converge locally.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 300s limit, kill at 290s to allow for cleanup/return
        deadline = start_time + 290.0

        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                f_val = float(func(x))
            except:
                f_val = np.inf
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Deterministic Lattice Search ---
        # Create a 3^5 grid over [-5, 5]^5
        # Points: -5, 0, 5
        grid_points = np.array(np.meshgrid([-5.0, 0.0, 5.0] * dim)).T.reshape(-1, dim)

        for x in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)

        # If we ran out of budget during grid search, return best
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # --- Phase 2: Local Refinement using (1+1)-ES ---
        # Start from the best point found in the grid
        if best_x is None:
            x = np.zeros(dim)
        else:
            x = best_x.copy()

        # Adaptive step size
        sigma = 1.0
        # We have budget - calls evaluations left
        # Let's use a simple adaptive rule:
        # If improvement, increase sigma slightly (or keep), else decrease.
        # Standard (1+1)-ES:
        # x_new = x + sigma * N(0, I)
        # If f(x_new) < f(x), x = x_new, sigma *= 1.05 (or similar)
        # Else, sigma /= 1.05

        # To be more robust, we can use a larger initial sigma and shrink it.
        # Given the grid spacing is 5.0, a sigma of 1.0 is reasonable for first step.
        # We will use a fixed schedule or adaptive. Let's use adaptive.

        # Pre-generate random numbers for speed? No, dim is small, budget is small.
        # But we can use np.random.randn which is fast.

        # We will run until budget is exhausted or time limit.
        # We can also stop if sigma becomes very small.
        
        # To ensure we make good use of the remaining budget, we can also try
        # a few restarts if the function seems flat, but (1+1)-ES is greedy.
        # For Weierstrass, local minima are sharp, so greedy should work well from a good start.

        # Let's refine the step size strategy.
        # Initial sigma = 1.0
        # If success: sigma = min(sigma * 1.1, 5.0)
        # If failure: sigma = max(sigma * 0.9, 1e-6)
        # This allows exploration then exploitation.

        # However, a more standard approach is to keep sigma constant or shrink it slowly.
        # Let's try:
        # sigma starts at 2.0 (to cover grid cell)
        # On success: sigma *= 1.0
        # On failure: sigma *= 0.95
        # This ensures we zoom in.

        # Actually, let's use a slightly more aggressive shrink on failure to guarantee convergence.
        # sigma_init = 2.0
        # If we use 257 evaluations, we can shrink significantly.
        # 2.0 * 0.95^257 is very small.

        x_current = best_x.copy()
        f_current = best_f
        sigma = 2.0

        while calls < budget and time.time() < deadline:
            # Generate candidate
            z = np.random.randn(dim)
            x_candidate = x_current + sigma * z

            # Clip to bounds? The task says func clips, but we can clip ourselves to avoid
            # evaluating out-of-bounds if it's cheaper? No, func handles it.
            # But we should keep x_current within bounds for the next step?
            # The task says minimizer is in [-4, 4], so we are safe.
            
            f_candidate = evaluate(x_candidate)

            if f_candidate < f_current:
                # Improvement
                x_current = x_candidate
                f_current = f_candidate
                # Keep sigma or increase slightly?
                # For Weierstrass, staying the same or slightly increasing helps escape shallow local mins.
                # But since we started from a global grid search, we are likely in the correct basin.
                # Let's keep sigma constant on success to maintain progress.
                pass
            else:
                # No improvement
                # Shrink sigma
                sigma *= 0.95
                if sigma < 1e-8:
                    break

        return best_x, best_f
