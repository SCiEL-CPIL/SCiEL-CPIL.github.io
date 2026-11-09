# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine Halton-based grid search for BBOB-style optimization.
    
    Uses deterministic Halton sequences at decreasing resolutions to guarantee
    uniform coverage, then refines around the best point found at each level.
    This avoids stochastic local minima traps on multimodal functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Halton sequence generator
        def halton_sequence(n, dim):
            """Generate n points in [0,1)^dim using Halton sequence."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for j in range(dim):
                    base = primes[j]
                    f = 1.0
                    r = 0.0
                    idx = i
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    points[i - 1, j] = r
            return points

        # Coarse-to-fine levels: each level has a resolution (number of points per dim)
        # Total points per level = resolution^dim (but we sample a subset for large dims)
        # Level 0: very coarse (2 per dim = 32 points for dim=5)
        # Level 1: coarse (4 per dim = 1024, sample 100)
        # Level 2: medium (8 per dim = 32768, sample 100)
        # Level 3: fine (16 per dim, sample around best)
        # Level 4+: very fine (32+ per dim, sample around best)
        
        # We'll use a budget allocation strategy:
        # - First 50 calls: coarse global coverage
        # - Next 100 calls: medium resolution
        # - Remaining 150 calls: local refinement around best
        
        # Phase 1: Coarse global search using Halton at low resolution
        # Use 50 points spread over the entire box
        n_coarse = 50
        halton_pts = halton_sequence(n_coarse, dim)
        # Map from [0,1)^dim to [-5,5]^dim
        halton_pts = halton_pts * 10.0 - 5.0
        
        for pt in halton_pts:
            if calls >= budget or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            f = func(pt_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt_clipped.copy()
        
        # Phase 2: Medium resolution - sample around the best point with smaller region
        # Use a box of size 4 around best_x (i.e., best_x ± 2)
        n_medium = 100
        halton_pts = halton_sequence(n_medium, dim)
        # Map from [0,1)^dim to a box centered at best_x with half-size 2
        box_half_size = 2.0
        medium_pts = best_x + (halton_pts * 2.0 - 1.0) * box_half_size
        medium_pts = np.clip(medium_pts, -5, 5)
        
        for pt in medium_pts:
            if calls >= budget or time.time() > deadline:
                break
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()
        
        # Phase 3: Local refinement - use a (1+1)-ES with small step size around best
        # This is where we spend the remaining budget
        remaining = budget - calls
        if remaining > 0:
            sigma = 0.1  # Start with small step
            x = best_x.copy()
            fx = best_f
            rng = np.random.default_rng(42)
            
            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Adaptive sigma: decrease as we approach convergence
                progress_in_phase = 1.0 - (budget - calls) / max(remaining, 1)
                sigma = 0.1 * (1.0 - 0.7 * progress_in_phase)
                sigma = max(sigma, 0.01)
                
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Slight increase on success
                    sigma = min(sigma * 1.05, 0.2)
                else:
                    # Slight decrease on failure
                    sigma = max(sigma * 0.9, 0.01)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
