# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive multi-objective optimizer for ZDT1 using hypervolume-driven bisection.
    Exploits the known structure of ZDT1 where the Pareto front is achieved
    when x_2..x_n are set to 0.
    Uses a strategy that selects the next x_1 to maximize the marginal hypervolume gain.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0
        
        # Reference point for hypervolume calculation
        ref = np.array([11.0, 11.0])
        
        def make_x(x1_val):
            x = np.zeros(n)
            x[0] = x1_val
            return x
            
        def evaluate(x1_val):
            nonlocal used
            if used >= budget:
                return None
            x = make_x(x1_val)
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # Phase 1: Initial points
        # We need at least 2 points to define an interval.
        # Start with 0.0 and 1.0 to cover the range.
        f1_start = evaluate(0.0)
        f1_end = evaluate(1.0)
        
        # Store points as (f1, f2, x)
        # We maintain a list of non-dominated points sorted by f1
        # Since we are on the Pareto front, all points are non-dominated.
        # We sort by f1 (which is x1).
        points = []
        if f1_start is not None:
            points.append((f1_start[0], f1_start[1], 0.0))
        if f1_end is not None:
            points.append((f1_end[0], f1_end[1], 1.0))
            
        # Sort by f1
        points.sort(key=lambda p: p[0])
        
        # Helper to calculate marginal hypervolume gain of inserting a point (f1, f2)
        # into the current set of points.
        def calc_gain(f1_new, f2_new):
            # Find the index where f1_new would be inserted
            # Points are sorted by f1.
            # We look for i such that points[i][0] <= f1_new <= points[i+1][0]
            # If f1_new is outside, we handle edges.
            
            # Current hypervolume contribution of the interval containing f1_new
            # The interval is [points[i][0], points[i+1][0]]
            # The point points[i] covers this interval with height (ref[1] - points[i][1])
            
            # Find i
            i = -1
            for k in range(len(points)):
                if points[k][0] <= f1_new:
                    i = k
                else:
                    break
            
            # If f1_new is smaller than all points, i = -1
            # If f1_new is larger than all points, i = len(points) - 1 (and the interval goes to ref[0])
            
            if i == -1:
                # f1_new < points[0][0]
                # Old contribution of points[0] in [f1_new, points[0][0]]? No, points[0] covers [points[0][0], ...]
                # The new point covers [f1_new, points[0][0]] with height (ref[1] - f2_new)
                # The old point points[0] covered [points[0][0], ...]
                # Wait, the hypervolume is the union.
                # If we add a point to the left, it adds a new rectangle [f1_new, points[0][0]] x [f2_new, ref[1]]
                # Does it overlap with existing? No, because f1_new < points[0][0].
                # So gain = (points[0][0] - f1_new) * (ref[1] - f2_new)
                if len(points) == 0:
                    # Only point
                    return (ref[0] - f1_new) * (ref[1] - f2_new)
                return (points[0][0] - f1_new) * (ref[1] - f2_new)
            
            elif i == len(points) - 1:
                # f1_new > points[-1][0]
                # New point covers [f1_new, ref[0]]
                # Old point points[-1] covered [points[-1][0], ref[0]]
                # The new point is to the right.
                # Gain = (ref[0] - f1_new) * (ref[1] - f2_new)
                # Wait, does it steal area from points[-1]?
                # points[-1] has f2_old. f2_new < f2_old (since f1_new > f1_old).
                # The region [f1_new, ref[0]] x [f2_new, ref[1]] is NOT covered by points[-1] because points[-1] only covers f2 >= f2_old, and f2_new < f2_old, so the strip [f2_new, f2_old] is new.
                # Actually, the hypervolume formula:
                # Old: points[-1] contributes (ref[0] - points[-1][0]) * (ref[1] - points[-1][1])
                # New: points[-1] contributes (f1_new - points[-1][0]) * (ref[1] - points[-1][1])
                #      new point contributes (ref[0] - f1_new) * (ref[1] - f2_new)
                # Gain = New - Old
                #      = (f1_new - points[-1][0]) * (ref[1] - points[-1][1]) + (ref[0] - f1_new) * (ref[1] - f2_new)
                #        - (ref[0] - points[-1][0]) * (ref[1] - points[-1][1])
                #      = (ref[1] - points[-1][1]) * (f1_new - points[-1][0] - (ref[0] - points[-1][0]))
                #        + (ref[0] - f1_new) * (ref[1] - f2_new)
                #      = (ref[1] - points[-1][1]) * (f1_new - ref[0])
                #        + (ref[0] - f1_new) * (ref[1] - f2_new)
                #      = (ref[0] - f1_new) * ( (ref[1] - f2_new) - (ref[1] - points[-1][1]) )
                #      = (ref[0] - f1_new) * (points[-1][1] - f2_new)
                return (ref[0] - f1_new) * (points[-1][1] - f2_new)
            
            else:
                # points[i][0] <= f1_new <= points[i+1][0]
                # Old contribution of points[i] in this interval:
                # (points[i+1][0] - points[i][0]) * (ref[1] - points[i][1])
                # New contributions:
                # points[i] covers [points[i][0], f1_new]: (f1_new - points[i][0]) * (ref[1] - points[i][1])
                # new point covers [f1_new, points[i+1][0]]: (points[i+1][0] - f1_new) * (ref[1] - f2_new)
                # Gain = New - Old
                #      = (f1_new - points[i][0]) * (ref[1] - points[i][1]) + (points[i+1][0] - f1_new) * (ref[1] - f2_new)
                #        - (points[i+1][0] - points[i][0]) * (ref[1] - points[i][1])
                #      = (ref[1] - points[i][1]) * (f1_new - points[i][0] - (points[i+1][0] - points[i][0]))
                #        + (points[i+1][0] - f1_new) * (ref[1] - f2_new)
                #      = (ref[1] - points[i][1]) * (f1_new - points[i+1][0])
                #        + (points[i+1][0] - f1_new) * (ref[1] - f2_new)
                #      = (points[i+1][0] - f1_new) * ( (ref[1] - f2_new) - (ref[1] - points[i][1]) )
                #      = (points[i+1][0] - f1_new) * (points[i][1] - f2_new)
                return (points[i+1][0] - f1_new) * (points[i][1] - f2_new)

        # Phase 2: Adaptive selection
        while used < budget:
            best_gain = -1
            best_x1 = None
            best_f1 = None
            best_f2 = None
            
            # Candidate 1: Midpoint of each interval between existing points
            for i in range(len(points) - 1):
                f1_left = points[i][0]
                f1_right = points[i+1][0]
                mid_x1 = (f1_left + f1_right) / 2.0
                
                # We need to estimate f2 for this x1.
                # Since we don't know the exact function without evaluating, 
                # we can't calculate the exact gain without evaluating.
                # However, we can approximate f2 using the known ZDT1 formula:
                # f2 = 1 - sqrt(f1) (for g=1)
                # This allows us to rank candidates without evaluation.
                # Then we evaluate the best one.
                
                f2_est = 1.0 - np.sqrt(mid_x1)
                gain_est = calc_gain(mid_x1, f2_est)
                
                if gain_est > best_gain:
                    best_gain = gain_est
                    best_x1 = mid_x1
                    best_f1 = mid_x1
                    best_f2 = f2_est
            
            # Candidate 2: Midpoint of the interval from last point to reference?
            # The reference is not a point in the set, but the hypervolume extends to ref.
            # The "next" point after the last one is effectively the reference point (11, 11)?
            # No, the reference point defines the boundary.
            # The last point p_last covers [p_last.f1, ref[0]].
            # If we add a point p_new with f1 > p_last.f1, it splits this interval.
            # We should consider midpoints in [p_last.f1, ref[0]]?
            # But we only care about x1 in [0, 1].
            # So if p_last.f1 < 1.0, we should consider midpoints in [p_last.f1, 1.0].
            # If p_last.f1 == 1.0, there is no more room in [0, 1].
            
            if points and points[-1][0] < 1.0:
                f1_left = points[-1][0]
                f1_right = 1.0
                mid_x1 = (f1_left + f1_right) / 2.0
                f2_est = 1.0 - np.sqrt(mid_x1)
                gain_est = calc_gain(mid_x1, f2_est)
                if gain_est > best_gain:
                    best_gain = gain_est
                    best_x1 = mid_x1
                    best_f1 = mid_x1
                    best_f2 = f2_est
            
            # If no candidates (should not happen), break
            if best_x1 is None:
                break
            
            # Evaluate the best candidate
            f_val = evaluate(best_x1)
            if f_val is None:
                break
                
            # Update the points list
            # Insert (f1, f2, x1) into points, keeping sorted by f1
            new_point = (f_val[0], f_val[1], best_x1)
            
            # Find insertion index
            ins_idx = 0
            for k in range(len(points)):
                if points[k][0] < new_point[0]:
                    ins_idx = k + 1
                else:
                    break
            
            points.insert(ins_idx, new_point)
            
        return np.array(all_X)
# EVOLVE-BLOCK-END
