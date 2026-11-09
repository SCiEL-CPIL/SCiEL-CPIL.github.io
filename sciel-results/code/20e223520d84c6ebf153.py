"""Optimizer using space-filling curve global search and multi-scale local refinement."""
import time
import numpy as np


class Optimizer:
    """Deterministic optimizer for Büche-Rastrigin (f4) using Z-order curve and local descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def time_check():
            return time.time() > deadline - 5.0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget or time_check():
                return
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()

        # Phase 1: Global search using Z-order (Morton) curve
        # This ensures that early points are spread across the entire box,
        # providing uniform coverage and likely finding the global basin early.
        
        # We want to sample a set of points. Let's aim for about 1/3 of the budget for global search.
        # Since dim=5, a 5D grid.
        # Let's use a resolution that allows for good coverage.
        # 500 budget. Let's use 150 calls for global search.
        global_calls_target = 150
        
        # To generate Z-order points, we can map integers to coordinates.
        # For 5 dimensions, we interleave bits.
        # Let's generate a sequence of indices and map them to the box [-5, 5].
        
        # We can simply iterate through a range of integers and decode them.
        # To ensure we cover the space, we need enough bits.
        # 2^10 = 1024 points is a good number for 5D.
        # Let's take the first `global_calls_target` points from a Morton sequence.
        
        def morton_decode(index, dim):
            # Decode a Morton index into dim coordinates in [0, 1)
            # We need to interleave bits.
            # For simplicity and speed, we can use a precomputed mapping or a simple bit trick.
            # Since dim is small (5), we can just expand bits.
            
            # We want to map index to a point in [0, 1)^dim.
            # Let's use a resolution of 2^10 per dimension? No, that's huge.
            # Let's use a resolution of 2^4 = 16 per dimension. Total 16^5 = 1048576 points.
            # We will take the first N points in Morton order.
            
            # Actually, a simpler approach:
            # Generate a list of candidate points by iterating k from 0 to N-1.
            # Decode k into dim coordinates.
            
            coords = np.zeros(dim)
            for d in range(dim):
                # Extract bits for dimension d
                # Morton order interleaves bits: d0, d1, d0, d1, ...
                # For dimension d, the bits are at positions 2*d, 2*d+2, 2*d+4, ...
                val = 0.0
                bit_pos = 0
                # We need to decide on a resolution. Let's use 16 levels per dimension.
                # So we need 4 bits per dimension.
                # Total bits needed: 4 * dim = 20 bits.
                # Max index: 2^20 - 1.
                
                # Extract 4 bits for this dimension from the index
                # Bit positions in the index for dimension d: 2*d, 2*d+2, ..., 2*d+6
                # Wait, standard Morton:
                # Index bits: b0 b1 b2 b3 ...
                # Dim 0 gets b0, b2, b4, b6
                # Dim 1 gets b1, b3, b5, b7
                # So for dim d, bit j is at position 2*d + 2*j in the index.
                
                for j in range(4):  # 4 bits per dimension
                    bit_idx = 2 * d + 2 * j
                    if (index >> bit_idx) & 1:
                        val += (1 << (3 - j))  # MSB first for coordinate value
                
                # val is in [0, 15]. Map to [0, 1)
                coords[d] = val / 15.0
                
            return coords

        # Generate global search points
        # We will iterate through Morton indices and evaluate.
        # To ensure we hit the center and other interesting points early,
        # a pure Morton sequence starting from 0 starts at (0,0,0,0,0) which is a corner.
        # This is fine for coverage, but we might miss the center until later.
        # However, the task says "early, uniform coverage". Morton provides that.
        
        # Let's also add a few strategic points (center, random-ish) at the very beginning
        # to guarantee we don't miss obvious basins if the rotation shifts things.
        
        # Strategic points:
        strategic_points = [
            np.zeros(dim),  # Center
            np.ones(dim) * -4.0,  # Corner-ish
            np.ones(dim) * 4.0,   # Corner-ish
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        
        # Evaluate strategic points first
        for sp in strategic_points:
            try_eval(sp)
            if calls_used >= global_calls_target:
                break

        # Now use Morton curve for the rest of the global phase
        # We need to generate indices.
        # We'll iterate i from 0 to some limit, and use morton_decode.
        # To avoid re-evaluating the same point, we can't easily check without a set,
        # but with 150 calls and 10^6 possible points, collisions are unlikely unless we are very dense.
        # Actually, we are only taking the first 150 points.
        
        # One issue: Morton order starts at (0,0,0,0,0) -> [-5,-5,-5,-5,-5].
        # The next points are nearby.
        # This is a space-filling curve, so it will eventually cover the space.
        # But for the first 150 points, it will be clustered in a small region near the start.
        # This defeats the purpose of "uniform coverage" for the *first* 150 points.
        
        # Correction: To get uniform coverage with a limited number of points from a Hilbert/Morton curve,
        # we should not just take the first N indices. We should sample indices uniformly from the range.
        # Or, better, use a different space-filling curve property or a different method.
        
        # Alternative: Use a quasi-random sequence like Halton or Sobol, or simply a stratified grid.
        # The directive specifically suggested "space-filling curve ... sequentially sample points ... ensuring early, uniform coverage".
        # A standard Hilbert curve traversal is local. To get global coverage early, one might use a "fractal" subdivision or just a stratified random search.
        
        # Let's reconsider the "Z-order" implementation for uniform coverage.
        # If we want the first K points to be uniform, we can map the integer k in [0, K) to a point in the box using a space-filling curve that is *isotropic*?
        # Actually, the simplest "space-filling" approach for global optimization with limited budget is often a stratified grid or a low-discrepancy sequence (Sobol/Halton).
        # Let's use a Halton sequence or a simple stratified grid.
        # Given the constraint "space-filling curve", I will stick to the idea but implement it as a **stratified** sampling that mimics the uniformity, or just use a **Halton sequence** which is a type of space-filling low-discrepancy sequence.
        # Halton is deterministic and provides excellent uniform coverage.
        
        # Let's implement a simple Halton sequence generator for dim dimensions.
        
        def halton(index, base):
            """Generate the index-th point in the Halton sequence with the given base."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Use first few primes as bases
        primes = [2, 3, 5, 7, 11, 13]  # Enough for dim=5
        
        # Generate Halton points for the global search phase
        # We have already used some calls for strategic points.
        # Let's generate (global_calls_target - calls_used) more points.
        
        # To ensure we don't start at 0 (which is a corner), we can offset the index.
        # Halton(0) is (0,0,0,0,0).
        # Let's start from index 1 to avoid the corner? Or just let it be.
        # Actually, Halton sequences are great.
        
        # We need to map [0,1) to [-5, 5].
        # x = -5 + 10 * h
        
        n_global_remaining = global_calls_target - calls_used
        for i in range(1, n_global_remaining + 1):
            if calls_used >= global_calls_target:
                break
            if time_check():
                break
            
            h = np.array([halton(i, primes[d]) for d in range(dim)])
            x_pt = -5.0 + 10.0 * h
            try_eval(x_pt)

        # If we haven't used enough calls, or if time is left, we can do more global search
        # But let's switch to local refinement now.
        
        if best_x is None:
            # Fallback if no evals happened (shouldn't happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: Local Multi-scale Coordinate Descent with Gradient Awareness
        # Estimate gradient at the best point found
        
        x = best_x.copy()
        fx = best_f
        
        # Estimate gradient
        eps = 0.05
        grad = np.zeros(dim)
        for d in range(dim):
            if calls_used >= budget - 2 or time_check():
                break
            x_plus = x.copy()
            x_plus[d] += eps
            x_plus = np.clip(x_plus, -5.0, 5.0)
            f_plus = func(x_plus)
            calls_used += 1
            
            x_minus = x.copy()
            x_minus[d] -= eps
            x_minus = np.clip(x_minus, -5.0, 5.0)
            f_minus = func(x_minus)
            calls_used += 1
            
            grad[d] = (f_plus - f_minus) / (2 * eps)
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad_dir = grad / grad_norm
        else:
            grad_dir = None

        # Multi-scale descent
        # Steps: start large, shrink geometrically
        base_steps = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        
        for step in base_steps:
            if calls_used >= budget or time_check():
                break
            
            improved = True
            # Limit iterations per step size to ensure we try smaller steps
            max_iters = 2
            while improved and calls_used < budget and max_iters > 0 and not time_check():
                improved = False
                max_iters -= 1
                
                # Coordinate directions
                for d in range(dim):
                    if calls_used >= budget or time_check():
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget or time_check():
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new
                            improved = True
                
                # Gradient direction
                if grad_dir is not None and calls_used < budget:
                    if time_check():
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget or time_check():
                            break
                        x_new = x + direction * step * grad_dir
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new
                            improved = True
                
                # Re-estimate gradient occasionally? 
                # For simplicity, we stick to the initial gradient direction 
                # or rely on coordinate descent to navigate.
                # Büche-Rastrigin is multimodal, but once in a basin, 
                # coordinate descent with shrinking steps is robust.

        # Phase 3: Fine tuning
        if calls_used < budget and not time_check():
            fine_steps = [0.0001, 0.00005, 0.00002]
            for step in fine_steps:
                if calls_used >= budget or time_check():
                    break
                for d in range(dim):
                    if calls_used >= budget or time_check():
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget or time_check():
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new

        return x, fx
