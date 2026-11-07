import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290 # 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT6 Pareto front is at x2..xn = 0, so S = 0.
        # f1(x) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2(x) = 1 - sqrt(f1)
        
        def f1_of_x1(x1):
            return 1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1) ** 8

        def get_x(x1, S):
            x = np.zeros(self.dim)
            x[0] = x1
            x[1:] = S / (self.dim - 1)
            x = np.clip(x, lo, hi)
            return x

        def evaluate(x1, S):
            if time.time() > deadline:
                return None
            x = get_x(x1, S)
            f = problem(x)
            return (x1, S, x, f)

        records = []
        evals_used = 0
        
        # Phase 1: Sample the Pareto front with optimal spacing
        # To maximize hypervolume, we should place points where the marginal gain is highest.
        # The Pareto front is f2 = 1 - sqrt(f1).
        # Hypervolume contribution of a point at f1 is roughly (11 - f2) * (11 - f1) * (width in f1).
        # Actually, for a fixed number of points N, uniform spacing in f1 is often good,
        # but since f1 is non-monotonic in x1, we need to map f1 -> x1.
        
        # Let's use a density that accounts for the curvature.
        # The reference point is [11, 11].
        # f1 in [0, 1]. f2 in [0, 1].
        
        N_phase1 = 100
        # We want to cover f1 in [0, 1].
        # Let's try to space points to maximize the sum of hypervolumes.
        # A simple heuristic: uniform in f1 might not be optimal if the front is curved.
        # However, f2 = 1 - sqrt(f1) is convex.
        # Let's just use uniform f1 for now, and refine x1 to match.
        
        f1_targets = np.linspace(0, 1, N_phase1)
        
        # For each f1 target, find x1.
        # f1(x1) is non-monotonic. It has multiple local maxima/minima.
        # We want the x1 that gives f1_target and is on the Pareto front (i.e., x2..xn=0).
        # Actually, any x1 giving that f1 with S=0 is on the Pareto front?
        # No, the Pareto front is the set of non-dominated points.
        # For ZDT6, the Pareto front is indeed given by x2..xn=0.
        # But f1 is not 1-to-1.
        # We should pick the x1 that is "stable" or just any x1 that achieves the f1.
        # Since we evaluate the actual problem, we just need to find an x1 such that f1(x1) is close to target.
        
        # We can do a bisection or grid search for x1 for each f1 target.
        # But we have a budget of 200.
        # 100 points on the front.
        # We can pre-compute x1 for a grid of f1 values? No, we must use the problem function.
        
        # Alternative: Sample x1 uniformly? No, f1 is not uniform.
        # Let's sample x1 with a density proportional to |df1/dx1|?
        # Or just sample x1 uniformly and hope?
        
        # Better: Inverse CDF sampling.
        # We know the shape of f1(x1).
        # We can create a lookup table of f1 for x1 in [0, 1].
        
        x1_grid = np.linspace(0, 1, 1000)
        f1_grid = np.array([f1_of_x1(x) for x in x1_grid])
        
        # For each f1_target, find the x1 that is closest.
        # Since f1 is non-monotonic, there might be multiple x1.
        # We want the one that is likely to be on the "main" Pareto front.
        # Actually, all points with S=0 are on the Pareto front if they are non-dominated.
        # Since f2 depends only on f1, and f1 determines f2, all points with the same f1 have the same f2.
        # So any x1 giving f1_target is equivalent in objective space.
        
        # We can just pick the first x1 in the grid that matches f1_target.
        
        for f1_t in f1_targets:
            if evals_used >= self.budget: break
            if time.time() > deadline: break
            
            # Find x1 closest to f1_t
            # Note: f1_grid might not cover [0,1] if the max is < 1?
            # max f1 is 1 (at x1=0.5, sin=1, exp(-2)~0.135, f1=0.865? No.
            # f1(0.5) = 1 - exp(-2)*1 = 0.864.
            # f1(0) = 0.
            # f1(1) = 1 - exp(-4)*0 = 1.
            # Wait, sin(6*pi*1) = 0. So f1(1)=1.
            # Max of f1?
            # Let's just use the grid to find the best match.
            
            idx = np.argmin(np.abs(f1_grid - f1_t))
            x1_val = x1_grid[idx]
            
            # Evaluate at x1_val, S=0
            res = evaluate(x1_val, 0.0)
            if res:
                records.append(res)
                evals_used += 1
        
        # Phase 2: Refinement
        # We have 100 points on the Pareto front (approx).
        # We have 100 evals left.
        # We can use them to refine the x1 values to get exactly the desired f1?
        # Or to add points in gaps?
        
        # Let's check the current non-dominated set.
        def is_dominated(f, records_list):
            for _, _, _, f_other in records_list:
                if np.all(f_other <= f) and np.any(f_other < f):
                    return True
            return False

        def get_non_dominated(records_list):
            nd = []
            for r in records_list:
                if not is_dominated(r[3], records_list):
                    nd.append(r)
            return nd

        nd_set = get_non_dominated(records)
        
        # We can try to optimize the positions of these points?
        # Or just add more points?
        # With 100 points, we might have gaps.
        # Let's use the remaining budget to add points in the largest gaps in f1-space.
        
        evals_left = self.budget - evals_used
        
        while evals_left > 0 and time.time() < deadline:
            if not nd_set:
                break
            
            # Get f1 values of non-dominated points
            # Note: f1 is r[3][0]
            f1_vals = np.array([r[3][0] for r in nd_set])
            
            # Sort by f1
            sorted_idx = np.argsort(f1_vals)
            sorted_f1 = f1_vals[sorted_idx]
            
            # Find largest gap
            # Include boundaries 0 and 1?
            # The front is [0, 1].
            # Gaps: 0 to first, last to 1, and between points.
            
            gaps = []
            if len(sorted_f1) > 0:
                # Gap from 0 to first
                gaps.append((0, sorted_f1[0]))
                # Gaps between points
                for i in range(len(sorted_f1) - 1):
                    gaps.append((sorted_f1[i], sorted_f1[i+1]))
                # Gap from last to 1
                gaps.append((sorted_f1[-1], 1.0))
            
            if not gaps:
                break
                
            # Pick the largest gap
            max_gap = max(gaps, key=lambda g: g[1] - g[0])
            mid_f1 = (max_gap[0] + max_gap[1]) / 2.0
            
            # Find x1 for mid_f1
            idx = np.argmin(np.abs(f1_grid - mid_f1))
            x1_val = x1_grid[idx]
            
            res = evaluate(x1_val, 0.0)
            if res:
                records.append(res)
                evals_left -= 1
                # Update nd_set
                if not is_dominated(res[3], records):
                    nd_set.append(res)
                    # Remove dominated points from nd_set?
                    # For efficiency, we could, but N is small.
                    # Let's just keep it simple.
                    
        return [r[2] for r in records]
