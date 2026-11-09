# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Coarse-to-Fine Grid Search Optimizer.
    
    This optimizer uses a deterministic Halton sequence for initial coarse coverage,
    followed by iterative refinement around the best point found. The step size
    is halved upon each global improvement, ensuring systematic convergence
    without stochastic local minima traps.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        # Fixed seed for reproducibility
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def time_exceeded():
            return time.time() > deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Halton sequence generator for quasi-random uniform coverage
        def halton(idx, base):
            q = 1.0
            r = 0.0
            while idx > 0:
                q /= base
                r += q * (idx % base)
                idx //= base
            return r

        primes = [2, 3, 5, 7, 11]

        # Phase 1: Coarse Search with Halton Sequence
        # We allocate ~40% of the budget to coarse search
        n_coarse = min(120, budget // 2)
        
        # Generate Halton points in [0, 1)^dim and map to [-5, 5]^dim
        for i in range(1, n_coarse + 1):
            if calls >= budget or time_exceeded():
                break
            pt = np.zeros(dim)
            for j in range(dim):
                pt[j] = halton(i, primes[j])
            pt = pt * 10.0 - 5.0
            pt = np.clip(pt, -5, 5)
            
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # Phase 2: Coarse-to-Fine Refinement
        # We iteratively refine around the best point found so far.
        # Start with a large step size and halve it upon improvement.
        # If no improvement after a fixed number of tries at a given scale, 
        # we reduce the scale anyway to ensure convergence.
        
        # Initial step size for refinement
        # We want to cover the box initially, so step size ~ 2.0 (box width 10, 5 steps)
        current_scale = 2.0
        min_scale = 1e-4
        
        # Number of candidates to try at each scale
        # We use a deterministic grid of points around the best_x
        # In 5D, we can sample 2^5 = 32 corners of the hypercube defined by scale.
        # Plus the center (best_x itself, already evaluated, so skip).
        # Let's use a structured set of points: axes +/- and corners.
        
        while calls < budget and not time_exceeded():
            if current_scale < min_scale:
                break

            improved = False
            points_to_try = []
            
            # Generate candidates around best_x with current_scale
            # 1. Axis-aligned perturbations (10 points: +/- along each axis)
            for j in range(dim):
                for sign in [1, -1]:
                    pt = best_x.copy()
                    pt[j] += sign * current_scale
                    pt = np.clip(pt, -5, 5)
                    points_to_try.append(pt)
            
            # 2. Corner points (2^5 = 32 points)
            # This might be too many if budget is tight, but 42 points is fine.
            # We can reduce this if we want to be more aggressive with scaling down.
            # Let's stick to axes + a few random perturbations to keep it simple and fast.
            # Actually, corners are good for multimodal. Let's add them.
            # But 10 + 32 = 42 points per scale. If we halve scale, we do 42 again.
            # 300 / 42 ~ 7 scales. 2.0 -> 1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.06 -> 0.03 -> 0.015
            # This is a reasonable progression.
            
            for j in range(dim):
                pass # Placeholder, we already added axes above
            
            # Add corner points
            for mask in range(2 ** dim):
                pt = best_x.copy()
                for j in range(dim):
                    if mask & (1 << j):
                        pt[j] += current_scale
                    else:
                        pt[j] -= current_scale
                pt = np.clip(pt, -5, 5)
                points_to_try.append(pt)
                
            # Shuffle points to avoid bias towards axes/corners order
            rng.shuffle(points_to_try)
            
            for pt in points_to_try:
                if calls >= budget or time_exceeded():
                    break
                f = func(pt)
                calls += 1
                if f < best_f:
                    update_best(pt, f)
                    improved = True
            
            # If we improved, keep the same scale or increase slightly?
            # The directive says "refining the step size by half each time a new global minimum is found".
            # This implies: if improved, we found a better point, so we should zoom in (halve scale).
            # If not improved, we should also halve scale to ensure we eventually converge or move to a finer grid.
            # Wait, if we don't improve, maybe we are stuck in a local min and should explore more?
            # But the strategy is "coarse-to-fine". Usually, if you don't improve at a scale, 
            # you might want to try a different location or a larger scale.
            # However, the prompt says: "refining the step size by half each time a new global minimum is found".
            # This suggests: 
            # - If improved: scale /= 2
            # - If not improved: ?
            # If not improved, we might be at a local minimum. 
            # If we just keep halving, we will converge to the local minimum.
            # To escape local minima, we might need to restart or jump.
            # But the prompt specifically asks for a deterministic grid-based coarse-to-fine search 
            # to "avoid the stochastic local minima traps that hinder random-restart evolutionary strategies".
            # This implies the method relies on the grid structure to find the global min, 
            # perhaps by covering the space well enough in the coarse phase.
            
            # Let's interpret "refining... by half each time a new global minimum is found" 
            # as the primary mechanism. 
            # If no improvement, we should probably also halve to ensure we don't waste budget 
            # on a bad scale, OR we should restart from a different coarse point.
            # Given the "deterministic" constraint, restarting from a new Halton point is a good fallback.
            
            if improved:
                current_scale /= 2.0
            else:
                # If no improvement, we might be stuck. 
                # Let's try to restart from a new random (deterministic) point 
                # if we have used a lot of budget without improvement.
                # Or simply halve the scale to ensure termination and finer search.
                # Halving is safer for convergence.
                current_scale /= 2.0
                # If scale becomes very small and we haven't improved, we are stuck.
                # We could break, but let's continue until budget is used.

        return best_x, best_f
# EVOLVE-BLOCK-END
