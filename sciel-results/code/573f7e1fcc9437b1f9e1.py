import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Locate the optimal tail (x_2...x_n)
        # Standard ZDT1: g(x) = 1 + 10(n-1)/sum(x_2..n). Min g -> Max sum -> x_i = 1.
        # We verify this quickly. If it fails, we fall back to random search.
        
        n_search = 10
        best_x_tail = None
        best_f2 = np.inf
        
        x1_fixed = 0.5
        
        for i in range(n_search):
            if time.time() > deadline:
                break
            
            # Try the theoretical optimum first (all 1s)
            if i == 0:
                tail = np.ones(dim - 1)
            else:
                tail = np.random.random(dim - 1)
            
            x = np.concatenate(([x1_fixed], tail))
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                if f[1] < best_f2:
                    best_f2 = f[1]
                    best_x_tail = tail.copy()
            except:
                pass
        
        if best_x_tail is None:
            best_x_tail = np.ones(dim - 1)
        
        # Phase 2: Dynamic Sampling of x_1
        # We have budget - n_search evaluations left.
        n_sample = budget - n_search
        
        # The Pareto front for ZDT1 is f2 = 11 * h(f1).
        # h(t) = 1 - sqrt(t/11) - (t/11)sin(10*pi*t)
        # We want to maximize hypervolume with respect to reference point (11, 11).
        # Hypervolume contribution of a point (f1, f2) in 2D with ref (r1, r2)
        # when points are sorted by f1 is sum (r1 - f1_i) * (f2_{i-1} - f2_i) ... 
        # Actually, simpler: sort by f1. HV = sum_{i} (r1 - f1_i) * (f2_{i-1} - f2_i) is not quite right for non-uniform.
        # HV = \int_{min_f1}^{r1} (r2 - f2(x)) dx approx.
        
        # The "control" is the next x1 to sample.
        # We will use a greedy approach: maintain a list of sampled f1 values and their f2.
        # At each step, pick the x1 that maximizes the marginal hypervolume increase.
        # Since n_sample is up to 190, and we have 320s, we can afford O(N^2) or even O(N^3) per step if N is small, but 190^3 is 6.8M ops, which is fast in Python/NumPy if vectorized.
        
        # However, calculating exact HV for every candidate is expensive.
        # Let's approximate the marginal gain.
        # The marginal gain of adding a point (x, y) to a set of points sorted by x is roughly:
        # (r1 - x) * (y_prev - y) + (x - x_next) * (y - y_next) ... 
        # Actually, the standard way to calculate marginal HV gain in 2D:
        # If we insert (x, y) into the sorted list of non-dominated points:
        # It replaces the segment between its neighbors in the "skyline".
        
        # Let's implement a simpler heuristic that mimics optimal control:
        # The density of points should be proportional to the curvature or the "value" of the region.
        # For ZDT1, the front is concave. 
        # A known optimal strategy for hypervolume on a concave front is to sample more densely where the function is "higher" (farther from the axes) or where the curvature is higher?
        # Actually, for a concave function f(x), the optimal spacing for minimizing the error of the integral (which correlates with HV loss) is proportional to sqrt(|f''(x)|).
        
        # Let's compute the curvature of the ZDT1 front.
        # f2(t) = 11 * (1 - sqrt(t/11) - (t/11)sin(10*pi*t))
        # Let's just use the "Pontryagin" style feedback:
        # Start with a uniform grid.
        # Then, iteratively, add a point where the "gap" in hypervolume is largest.
        
        # To be efficient and robust:
        # 1. Generate a coarse grid of 20 points.
        # 2. Evaluate them.
        # 3. For the remaining budget, at each step, evaluate the midpoints of the largest "HV gaps" in the current set of points.
        
        # This is equivalent to an adaptive quadrature scheme for the hypervolume integral.
        
        # Initialize with a few points to establish the shape
        n_init = 20
        x1_init = np.linspace(lo[0], hi[0], n_init)
        
        sampled_x = []
        sampled_f = []
        
        for x1 in x1_init:
            if time.time() > deadline:
                break
            x = np.concatenate(([x1], best_x_tail))
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                sampled_x.append(x1)
                sampled_f.append(f)
            except:
                pass
        
        # If we ran out of budget, return
        if len(sampled_x) >= budget:
            return np.zeros((1, dim))
        
        # Data structures for adaptive sampling
        # We need to track the current "skyline" or just the set of points.
        # In 2D, the hypervolume is determined by the non-dominated points.
        # Let's maintain a sorted list of non-dominated points.
        
        # Helper to calculate HV of a set of points
        def calc_hv(points, ref):
            # points: list of (f1, f2)
            # ref: (r1, r2)
            if not points:
                return 0.0
            # Sort by f1
            pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_f2 = ref[1]
            for f1, f2 in pts:
                if f2 >= ref[1] or f1 >= ref[1]:
                    continue
                # The area added by this point is (ref[1] - f2) * (width of its dominance region in f1)
                # But we need to be careful. 
                # Standard 2D HV calculation:
                # Sort by f1 ascending.
                # Keep track of the minimum f2 seen so far (from the right? No, from the left).
                # Actually, iterate from right to left (high f1 to low f1).
                # Maintain best_f2 (min f2 seen so far).
                # If current f2 < best_f2, it contributes.
                # Contribution = (best_f2 - f2) * (current_f1 - next_f1) ... no.
                
                # Let's use the standard algorithm:
                # Sort points by f1 ascending.
                # Filter out points where f1 >= r1 or f2 >= r2.
                # Sort by f1 ascending.
                # Iterate. Keep track of the "skyline".
                # A point (x, y) is non-dominated if there is no other point (x', y') with x' <= x and y' <= y.
                # In 2D, if sorted by x, a point is non-dominated if its y is strictly less than all previous y's? No.
                # If sorted by x ascending, a point is non-dominated if its y is strictly less than the minimum y of all points with smaller x?
                # Wait. (x1, y1) dominates (x2, y2) if x1<=x2 and y1<=y2.
                # So if we sort by x ascending, a point is non-dominated if its y is smaller than all y's to its left?
                # No. (0, 10) and (10, 0). (0,10) doesn't dominate (10,0).
                # If we have (0, 10) and (1, 1). (1,1) dominates (0,10)? No. (0,10) has smaller x.
                # (1,1) dominates (0,10) if 1<=0 (False).
                # So neither dominates the other?
                # (0,10) vs (1,1): 0<1, 10>1. No dominance.
                # (0,10) vs (5, 5): 0<5, 10>5. No dominance.
                # (0,10) vs (5, 20): 0<5, 10<20. (0,10) dominates (5,20).
                
                # Algorithm:
                # 1. Filter valid points.
                # 2. Sort by f1 ascending.
                # 3. Iterate from right to left (largest f1 to smallest).
                # 4. Maintain min_f2_so_far (initialized to ref[1]).
                # 5. If current f2 < min_f2_so_far:
                #    It is on the skyline.
                #    Contribution = (min_f2_so_far - f2) * (f1 - next_f1) ... 
                #    Wait, the width is determined by the next point to the right (larger f1).
                #    Let's store the skyline points.
                
                skyline = []
                min_f2 = ref[1]
                # Sort by f1 descending for this logic?
                # Let's sort by f1 ascending.
                # Iterate from right to left.
                # If f2 < min_f2:
                #   Add to skyline.
                #   min_f2 = f2.
                
                # Then calculate HV from skyline.
                # Skyline points are (x_i, y_i) with x increasing and y increasing?
                # No, if we go right to left and keep min_f2, the y values will be increasing as we go left?
                # Let's trace:
                # Points: (0, 10), (5, 5), (10, 0). Ref (11, 11).
                # Sorted asc: (0,10), (5,5), (10,0).
                # Right to left:
                # (10, 0): 0 < 11. Skyline: [(10,0)]. min_f2=0.
                # (5, 5): 5 < 0? No.
                # (0, 10): 10 < 0? No.
                # Skyline: [(10,0)].
                # HV = (11-10)*(11-0) = 11.
                # Correct?
                # Area under (10,0) to (11,0) is 1*11=11.
                # But (0,10) and (5,5) are dominated by (10,0)?
                # (10,0) dominates (5,5)? 10<=5 False.
                # So (5,5) is NOT dominated by (10,0).
                # My logic "if f2 < min_f2" is wrong for "non-dominated".
                
                # Correct Logic for 2D Non-Dominated Set (sorted by f1 asc):
                # A point is non-dominated if its f2 is smaller than ALL points to its left?
                # No.
                # (0, 10) and (10, 0).
                # (0, 10) is non-dominated.
                # (10, 0) is non-dominated.
                # (5, 5) is non-dominated.
                # (5, 15) is dominated by (0, 10)? 0<=5, 10<=15. Yes.
                
                # So, sort by f1 ascending.
                # Keep track of min_f2_seen_so_far (from the left).
                # If current f2 < min_f2_seen_so_far, it is NOT dominated by any point to its left.
                # But it could be dominated by a point to its right?
                # If a point to its right (larger f1) has smaller f2, it dominates it.
                # So we must filter from the right as well?
                # Actually, the set of non-dominated points in 2D, when sorted by f1, has strictly decreasing f2.
                
                # Algorithm to get skyline:
                # 1. Sort by f1 ascending.
                # 2. Remove duplicates (keep min f2 for same f1).
                # 3. Iterate from right to left. Keep min_f2.
                #    If current f2 < min_f2:
                #       Add to list.
                #       min_f2 = current f2.
                # This yields points that are not dominated by any point with LARGER f1.
                # Are they dominated by points with SMALLER f1?
                # If (x1, y1) has x1 < x2 and y1 < y2, then (x1, y1) dominates (x2, y2).
                # In the right-to-left pass, if we encounter (x1, y1) after (x2, y2), and y1 < y2, we would have kept (x2, y2) only if y2 was smaller than everything to its right.
                # But if y1 < y2, and x1 < x2, then (x1, y1) dominates (x2, y2).
                # So (x2, y2) should be removed.
                # My right-to-left pass keeps (x2, y2) if y2 < min_f2_right.
                # It does NOT check left.
                
                # Correct Standard Algorithm:
                # 1. Sort by f1 ascending.
                # 2. Iterate left to right. Keep min_f2.
                #    If current f2 < min_f2:
                #       It is not dominated by left.
                #       Add to candidate list.
                #       min_f2 = current f2.
                # This list has decreasing f2? No, f2 is decreasing?
                # If f2 < min_f2, then f2 is getting smaller. So f2 is decreasing.
                # So we have points with increasing f1 and decreasing f2.
                # These are the candidates.
                # Do any of them dominate each other?
                # (x1, y1) and (x2, y2) with x1<x2 and y1>y2.
                # Neither dominates the other.
                # So this IS the skyline.
                
                # So:
                # 1. Sort by f1 asc.
                # 2. Filter: keep point if f2 < min_f2_so_far. Update min_f2.
                # 3. Calculate HV from this skyline.
                
                # HV Calculation from Skyline (sorted by f1 asc, f2 desc):
                # hv = 0
                # prev_f2 = ref[1]
                # for (f1, f2) in skyline:
                #    # The width of this segment is (next_f1 - f1)? No.
                #    # The segment extends from f1 to the next point's f1 (or ref[1] for the last one? No, ref[1] is f1 ref).
                #    # The height is (prev_f2 - f2)? No.
                #    # The area is sum of rectangles.
                #    # Rectangle for point i: width = (f1_{i+1} - f1_i) for i < last, (ref[1] - f1_last) for last?
                #    # Height = (ref[1] - f2_i)? No.
                #    # The hypervolume is the area between the skyline and the reference point.
                #    # It is the sum of areas of rectangles defined by each point on the skyline.
                #    # For a point (x_i, y_i) on the skyline, it contributes a rectangle of width (x_{i+1} - x_i) and height (y_i - y_{i+1})?
                #    # No, it's easier to integrate.
                #    # HV = \sum (ref[1] - x_i) * (y_{i-1} - y_i) ... no.
                #    
                #    # Let's use the "shoelace" like area or simple summation.
                #    # HV = \sum_{i=0}^{N-1} (x_{i+1} - x_i) * (ref[1] - y_i) + (ref[1] - x_N) * (ref[1] - y_N) ?
                #    # Let's test with (0, 10) and (10, 0). Ref (11, 11).
                #    # Skyline: (0, 10), (10, 0).
                #    # i=0: (10-0)*(11-10) = 10.
                #    # i=1 (last): (11-10)*(11-0) = 11.
                #    # Total 21.
                #    # Manual: Area from 0 to 10 under line connecting (0,10) and (10,0)?
                #    # No, the skyline is the lower envelope.
                #    # The area is the integral of (ref[1] - f2(x)) dx.
                #    # From 0 to 10, f2 goes from 10 to 0.
                #    # Approximation by rectangles:
                #    # If we use right Riemann sum: (10-0)*(11-0) = 110? No.
                #    # The standard HV calculation for a set of points is the volume of the union of boxes.
                #    # Box for (x,y) is [x, ref[1]] x [y, ref[1]].
                #    # Union of boxes.
                #    # In 2D, sorted by x, the union area is sum of (x_{i+1} - x_i) * (ref[1] - y_i) where x_{N} = ref[1].
                #    # Wait, if y is decreasing, the box for (x_i, y_i) covers x in [x_i, x_{i+1}] and y in [y_i, ref[1]].
                #    # But (x_{i+1}, y_{i+1}) has y_{i+1} < y_i.
                #    # So the box for i+1 covers a LOWER y range.
                #    # So the union is indeed the sum of rectangles:
                #    # Width_i = x_{i+1} - x_i (with x_N = ref[1])
                #    # Height_i = ref[1] - y_i
                #    # Let's re-test:
                #    # (0, 10): Width = 10-0=10. Height = 11-10=1. Area=10.
                #    # (10, 0): Width = 11-10=1. Height = 11-0=11. Area=11.
                #    # Total 21.
                #    # This matches.
                
                pts = sorted(points, key=lambda p: p[0])
                skyline = []
                min_f2 = ref[1]
                for f1, f2 in pts:
                    if f1 >= ref[0] or f2 >= ref[1]:
                        continue
                    if f2 < min_f2:
                        skyline.append((f1, f2))
                        min_f2 = f2
                
                hv = 0.0
                for i in range(len(skyline)):
                    f1, f2 = skyline[i]
                    next_f1 = skyline[i+1][0] if i + 1 < len(skyline) else ref[0]
                    width = next_f1 - f1
                    height = ref[1] - f2
                    hv += width * height
                return hv

        ref = (hi[0] + 10, hi[1] + 10) # Approximate ref, actually task says [11, 11]
        # Task says ref [11.0, 11.0]
        ref = (11.0, 11.0)
        
        current_points = list(zip(sampled_x, sampled_f))
        
        # Adaptive loop
        remaining = budget - len(sampled_x)
        
        while remaining > 0:
            if time.time() > deadline:
                break
            
            # Find the best next point to sample
            # Candidates: midpoints between existing skyline points?
            # Or just a dense grid search for the best marginal gain?
            # With ~190 points, a grid of 100 candidates is 190*100 = 19k HV calcs.
            # Each HV calc is O(N). Total 19k * 190 = 3.6M ops. Fast.
            
            # Generate candidates: midpoints of intervals in the current skyline
            # We need the current skyline to know the intervals.
            # Re-calc skyline each time? O(N log N). 190 * 190 log 190 ~ 2.5M. Fine.
            
            pts = sorted(current_points, key=lambda p: p[0])
            skyline = []
            min_f2 = ref[1]
            for f1, f2 in pts:
                if f1 >= ref[0] or f2 >= ref[1]:
                    continue
                if f2 < min_f2:
                    skyline.append((f1, f2))
                    min_f2 = f2
            
            # If skyline is empty, just sample random
            if not skyline:
                x1 = np.random.uniform(lo[0], hi[0])
            else:
                # Intervals: (0, skyline[0]), (skyline[i], skyline[i+1]), (skyline[-1], 1)
                # We want to sample where the marginal HV gain is highest.
                # Approximation: The gain is proportional to the area of the "gap".
                # For a concave function, the gap is largest in the middle.
                # Let's just pick the midpoint of the interval with the largest "height" (ref[1] - min_f2_in_interval).
                # Or simply, pick the midpoint of the widest interval?
                # Widest interval in x is a good proxy.
                
                candidates = []
                # Add endpoints if not covered
                if skyline[0][0] > lo[0]:
                    candidates.append((lo[0] + skyline[0][0]) / 2)
                
                for i in range(len(skyline) - 1):
                    x_start = skyline[i][0]
                    x_end = skyline[i+1][0]
                    candidates.append((x_start + x_end) / 2)
                
                if skyline[-1][0] < hi[0]:
                    candidates.append((skyline[-1][0] + hi[0]) / 2)
                
                # Also, to be safe, add a few random points
                for _ in range(5):
                    candidates.append(np.random.uniform(lo[0], hi[0]))
                
                # Evaluate marginal gain for each candidate
                # To speed up, we can approximate the gain without full HV recalculation?
                # No, full HV is safer.
                
                best_gain = -1
                best_x1 = None
                
                for cx in candidates:
                    # Create new set
                    new_points = current_points + [(cx, 0)] # Placeholder f2
                    # We don't know f2 yet!
                    # We must estimate f2.
                    # We can interpolate from the skyline.
                    # Find the two skyline points surrounding cx.
                    # Linear interpolation of f2.
                    
                    # Find interval
                    left_idx = -1
                    right_idx = -1
                    for i in range(len(skyline)):
                        if skyline[i][0] <= cx:
                            left_idx = i
                        else:
                            right_idx = i
                            break
                    if left_idx == -1:
                        left_idx = 0
                    if right_idx == -1:
                        right_idx = len(skyline) - 1
                    
                    x_l, y_l = skyline[left_idx]
                    x_r, y_r = skyline[right_idx]
                    
                    if x_r == x_l:
                        est_f2 = y_l
                    else:
                        t = (cx - x_l) / (x_r - x_l)
                        est_f2 = y_l + t * (y_r - y_l)
                    
                    # Now calculate HV with estimated point
                    # Replace placeholder
                    new_points_real = current_points + [(cx, est_f2)]
                    
                    # Optimization: Instead of full HV, just calculate the change.
                    # But full HV is O(N).
                    hv_new = calc_hv(new_points_real, ref)
                    # We need hv_old.
                    # We can cache hv_old outside the loop.
                    
                    if hv_new > best_gain:
                        best_gain = hv_new
                        best_x1 = cx
                
                if best_x1 is None:
                    best_x1 = np.random.uniform(lo[0], hi[0])
                
                x1 = best_x1
            
            # Evaluate the chosen x1
            x = np.concatenate(([x1], best_x_tail))
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                current_points.append((x1, f))
                remaining -= 1
            except:
                remaining -= 1
                continue
        
        return np.zeros((1, dim))
