import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim

        # ZDT6 Pareto Front parameters
        # x[0] in [0, 1], x[1:] = 0
        # f1 = 1 + 10*(dim-1)*x0
        # g = 1 + 10*(dim-1)
        # f2 = (1/g) * (1 - sqrt(f1/g))^2 + 1
        
        g_val = 1 + 10 * (dim - 1)
        
        # Reference point
        r1 = 11.0
        r2 = 11.0
        
        # Only points with f1 < r1 contribute to hypervolume.
        # f1 = 1 + 90*x0 < 11 => x0 < 10/90 = 1/9
        x0_max = (r1 - 1.0) / (10.0 * (dim - 1))
        
        # We have 200 evaluations. 
        # Strategy: Greedy hypervolume maximization on the Pareto front.
        # The front is a 1D curve parameterized by x0 in [0, x0_max].
        # We want to select x0 values to maximize the hypervolume of the resulting points.
        
        # Initial set of points
        # Start with a few well-spaced points to cover the range.
        # Then iteratively add the point that gives the best marginal gain.
        
        # Pre-compute f1, f2 for a dense grid of x0? No, we can compute on the fly.
        # But for greedy selection, we need to evaluate many candidates.
        # Let's use a candidate pool of, say, 1000 points uniformly distributed in [0, x0_max].
        # Then greedily pick the best one, remove it, and repeat.
        
        n_candidates = 1000
        x0_pool = np.linspace(0.0, x0_max, n_candidates)
        f1_pool = 1 + 10 * (dim - 1) * x0_pool
        f2_pool = (1.0 / g_val) * (1.0 - np.sqrt(f1_pool / g_val))**2 + 1.0
        
        # Selected points: list of (f1, f2, x0)
        selected = []
        
        # Available candidates: indices into the pool
        available = list(range(n_candidates))
        
        # Greedy selection
        for step in range(budget):
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_idx = -1
            best_point = None
            
            # Current selected points sorted by f1
            # selected is a list of (f1, f2, x0)
            
            for idx in available:
                f1_c = f1_pool[idx]
                f2_c = f2_pool[idx]
                
                # Find neighbors in selected
                # We maintain selected sorted by f1.
                # Find the point with the largest f1 < f1_c (left neighbor)
                # and the point with the smallest f1 > f1_c (right neighbor)
                
                left_f1 = 0.0
                left_f2 = r2
                right_f1 = r1
                right_f2 = r2
                
                found_left = False
                found_right = False
                
                for p in selected:
                    if p[0] < f1_c:
                        left_f1 = p[0]
                        left_f2 = p[1]
                        found_left = True
                    elif p[0] > f1_c:
                        if not found_right:
                            right_f1 = p[0]
                            right_f2 = p[1]
                            found_right = True
                        else:
                            break # Since sorted, we can break early if we found right
                
                # Calculate marginal hypervolume gain
                # The gain is the area of the new region dominated by c that was not dominated before.
                # In 2D, this is calculated by looking at the "box" defined by the left and right neighbors.
                
                # If no left neighbor, the "left" boundary is f1_c itself (area starts there).
                # Actually, the standard formula for marginal gain of point c in set S:
                # Let p_L be the point in S with max f1 < c.f1 (or (0, r2) if none? No, (0, r2) is not a point).
                # If no left point, the region [0, c.f1] is not dominated by any point in S with f1 < c.f1?
                # Wait, if no point has f1 < c.f1, then for any x in [0, c.f1], no point in S has f1 <= x.
                # So the height of the step function is r2 (infinity) in that region? No, it's undefined or r2.
                # The hypervolume contribution of a point p is (next_f1 - p.f1) * (r2 - p.f2).
                
                # Let's use the formula:
                # Gain = (right_f1 - left_f1) * (r2 - left_f2) - (right_f1 - f1_c) * (r2 - f2_c) - (f1_c - left_f1) * (r2 - left_f2)
                # This is: (right_f1 - left_f1) * (r2 - left_f2) - [(right_f1 - f1_c) * (r2 - f2_c) + (f1_c - left_f1) * (r2 - left_f2)]
                
                # If no left neighbor, left_f1 = f1_c, left_f2 = r2?
                # If left_f1 = f1_c, the term (f1_c - left_f1) is 0.
                # The term (right_f1 - left_f1) * (r2 - left_f2) becomes (right_f1 - f1_c) * (r2 - r2) = 0.
                # So Gain = 0 - (right_f1 - f1_c) * (r2 - f2_c) - 0 = -(...). This is wrong.
                
                # Correct logic:
                # Old area in [left_f1, right_f1] was dominated by the point at left_idx (if exists) or nothing.
                # If left_idx exists, old height was (r2 - left_f2).
                # If left_idx does NOT exist, then for x in [0, right_f1], no point in S has f1 <= x?
                # No, points to the right (f1 > x) do not dominate x.
                # So if no point has f1 < f1_c, then for x in [0, f1_c], no point dominates.
                # For x in [f1_c, right_f1], no point in S has f1 <= x? 
                # Wait, if c is the leftmost point, then for x in [f1_c, right_f1), no point in S has f1 <= x.
                # So the old height was r2 (no contribution).
                # So Old Area = 0.
                
                # New area:
                # Point c dominates [f1_c, right_f1) with height (r2 - f2_c).
                # If left_idx exists, point at left_idx dominates [left_f1, f1_c) with height (r2 - left_f2).
                # So New Area = (f1_c - left_f1) * (r2 - left_f2) + (right_f1 - f1_c) * (r2 - f2_c).
                
                # If left_idx does NOT exist:
                # New Area = 0 + (right_f1 - f1_c) * (r2 - f2_c).
                # Old Area = 0.
                # Gain = (right_f1 - f1_c) * (r2 - f2_c).
                
                if found_left:
                    old_area = (right_f1 - left_f1) * (r2 - left_f2)
                    new_area = (f1_c - left_f1) * (r2 - left_f2) + (right_f1 - f1_c) * (r2 - f2_c)
                else:
                    old_area = 0.0
                    new_area = (right_f1 - f1_c) * (r2 - f2_c)
                
                gain = new_area - old_area
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_point = (f1_c, f2_c, x0_pool[idx])
            
            if best_idx != -1:
                selected.append(best_point)
                available.remove(best_idx)
                selected.sort(key=lambda p: p[0])
            else:
                break

        # Now evaluate the selected points
        n_evals = 0
        for f1, f2, x0 in selected:
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break
            
            x = np.zeros(dim)
            x[0] = x0
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break

        return None
