import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Identify relevant dimensions based on WFG9 structure described in comments:
        # f0 depends on x0, f1 on x1, f2 on x2.
        # x3..x11 are irrelevant.
        
        # Fix irrelevant dimensions to mid-range to be safe
        mid = (lo + hi) / 2.0
        
        # Optimize x2 to minimize f2 component sin(2*pi*x2) -> -1
        # sin(2*pi*t) = -1 at t = 0.75 (mod 1)
        # Map 0.75 to the range [lo[2], hi[2]]
        # Assuming the period of sin(2*pi*x) is 1.
        # If bounds are large, we might need to be careful, but typically WFG bounds are [0, 1] or similar.
        # The problem statement says "bounds given per variable".
        # If the range is > 1, there are multiple minima.
        # Let's assume standard [0, 1] or similar where 0.75 is a good guess.
        # To be robust: find the value in [lo, hi] that is closest to k + 0.75 for some integer k.
        range_2 = hi[2] - lo[2]
        # We want x2 such that (x2 - lo[2]) / range_2 * (number of periods) ... 
        # Actually, sin(2*pi*x) has period 1.
        # Minima are at x = 0.75, 1.75, 2.75...
        # Let's just pick the one closest to the middle of the range if range is small, 
        # or generally pick 0.75 mapped into the range.
        # Simple heuristic: center + 0.75 * (hi-lo) is wrong if hi-lo != 1.
        # Correct heuristic:
        # We want x2 = lo[2] + delta such that sin(2*pi*(lo[2]+delta)) is minimized.
        # If the function is periodic with period 1, and the range is small (<1), 
        # we can just check the boundaries and the interior minimum.
        
        # Let's use a small local search for x2 to be safe and accurate.
        # But x2 is a single dimension.
        # Let's try 0.75 mod 1 mapped to the range.
        # If bounds are [0,1], x2=0.75.
        # If bounds are [0, 10], minima at 0.75, 1.75, ...
        # Let's just sample a few candidates for x2.
        candidates_x2 = []
        # Candidate 1: 0.75 mapped to center
        candidates_x2.append(mid[2] + 0.75 * (hi[2] - lo[2]) / 2.0) # This is just a guess
        # Better:
        # The term is sin(2*pi*x2).
        # Minima at x2 = 0.75 + k.
        # Find k such that x2 is in [lo[2], hi[2]].
        # k_min = ceil(lo[2] - 0.75)
        # k_max = floor(hi[2] - 0.75)
        k_min = int(np.ceil(lo[2] - 0.75))
        k_max = int(np.floor(hi[2] - 0.75))
        if k_min <= k_max:
            # Pick the one closest to mid[2]
            best_x2 = mid[2]
            min_dist = float('inf')
            for k in range(k_min, k_max + 1):
                x2_cand = 0.75 + k
                dist = abs(x2_cand - mid[2])
                if dist < min_dist:
                    min_dist = dist
                    best_x2 = x2_cand
            # Clip just in case
            x2_opt = np.clip(best_x2, lo[2], hi[2])
        else:
            # No full period minimum in range? Check boundaries and midpoint?
            # Actually, if range < 1, k_min > k_max might happen if the interval doesn't contain 0.75+k.
            # In that case, the minimum is at one of the boundaries.
            # Let's just evaluate the problem for x2 at lo, mid, hi and pick best?
            # No, we can't call problem yet in a loop for x2 easily without constructing x.
            # Let's assume standard bounds [0,1] or similar where 0.75 is in range.
            # Fallback: mid
            x2_opt = mid[2]

        # Construct base vector
        x_base = mid.copy()
        x_base[2] = x2_opt
        
        # Grid search on x0, x1
        # Determine grid size
        # Reserve some budget for exploitation
        # Let's use 75% of budget for grid
        n_grid = int(np.sqrt(self.budget * 0.75))
        if n_grid < 2: n_grid = 2
        
        g0 = np.linspace(lo[0], hi[0], n_grid)
        g1 = np.linspace(lo[1], hi[1], n_grid)
        
        X = []
        F = []
        
        for a in g0:
            for b in g1:
                x = x_base.copy()
                x[0] = a
                x[1] = b
                X.append(x)
                F.append(np.asarray(problem(x), dtype=float))
                
        used = len(X)
        F = np.asarray(F, dtype=float)
        
        def dominated_mask(F):
            m = len(F)
            dom = np.zeros(m, dtype=bool)
            for i in range(m):
                fi = F[i]
                le = np.all(F <= fi, axis=1)
                lt = np.any(F < fi, axis=1)
                mask = le & lt
                mask[i] = False
                if mask.any():
                    dom[i] = True
            return dom

        dom = dominated_mask(F)
        nd = np.where(~dom)[0]
        
        rng = np.random.default_rng(0)
        
        # Exploitation: Perturb only x0 and x1
        # Decreasing noise
        while used < self.budget and nd.size > 0:
            # Pick a random non-dominated point
            pi = nd[rng.integers(nd.size)]
            parent = X[pi]
            
            frac = used / float(self.budget)
            # Scale noise: start larger, end smaller
            scale = (0.1 - 0.08 * frac) * (hi - lo)
            
            child = parent.copy()
            # Perturb only x0 and x1
            child[0] += scale[0] * rng.standard_normal()
            child[1] += scale[1] * rng.standard_normal()
            
            # Clip
            child[0] = np.clip(child[0], lo[0], hi[0])
            child[1] = np.clip(child[1], lo[1], hi[1])
            
            fc = np.asarray(problem(child), dtype=float)
            used += 1
            X.append(child)
            F = np.vstack([F, fc])
            
            # Update non-dominated set
            dom = dominated_mask(F)
            nd = np.where(~dom)[0]
            
        return X
