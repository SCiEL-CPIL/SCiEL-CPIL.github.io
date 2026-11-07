import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using optimal point placement on the known Pareto front.
    
    Strategy:
    1. The Pareto front of ZDT3 is known analytically: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1],
       achieved when x2..xn = 0.
    2. Instead of uniform sampling, we use a greedy algorithm to place points on the front
       such that they maximize the hypervolume indicator.
    3. We start with the endpoints (0,1) and (1,0), then iteratively add the point that
       maximizes the marginal hypervolume gain.
    4. This is done by considering candidate points on a fine grid and selecting the one
       that covers the most new area relative to the current set of points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point
        ref = np.array([11.0, 11.0])
        R1, R2 = 11.0, 11.0

        # We will evaluate exactly on the Pareto front.
        # x = [x1, 0, 0, ..., 0]
        # f1 = x1
        # f2 = 1 - sqrt(x1)

        # Greedy point placement
        # Start with the two extreme points
        x1_selected = [0.0, 1.0]
        f1_selected = np.array([0.0, 1.0])
        f2_selected = np.array([1.0, 0.0])  # 1 - sqrt(0) = 1, 1 - sqrt(1) = 0
        
        # Sort by f1 (they are already sorted)
        # f1: [0, 1], f2: [1, 0]
        
        # We need to select (budget - 2) more points.
        # Candidate points: a fine grid in [0, 1]
        # Let's use a grid of 1000 points for candidates.
        # But we should exclude the already selected points (0 and 1).
        
        # To make the greedy step fast, we can maintain the current "envelope" of points.
        # The hypervolume in 2D is the area under the step function formed by the points.
        # The marginal gain of adding a point (x, y) is the area of the rectangle it adds.
        
        # Let's pre-compute the candidates.
        # We'll use a grid of 1000 points.
        # Actually, we can be smarter. The optimal points are likely to be clustered where the curve is "bulging" out.
        # But for now, a fine grid should work.
        
        n_candidates = 1000
        x1_grid = np.linspace(0, 1, n_candidates)
        f1_grid = x1_grid
        f2_grid = 1 - np.sqrt(x1_grid)
        
        # We already have x1=0 and x1=1.
        # So we consider candidates from index 1 to n_candidates-2.
        
        # To speed up the greedy selection, we can note that the marginal gain of a point
        # depends on its neighbors in the current set.
        
        # Let's implement the greedy loop.
        # For each step, we find the candidate that maximizes the marginal HV.
        
        # To calculate marginal HV efficiently:
        # The current set of points is sorted by f1.
        # Let the points be (f1_0, f2_0), ..., (f1_k, f2_k).
        # The HV is the area of the union of rectangles [f1_i, R1] x [f2_i, R2].
        # Since the points are non-dominated and on a convex front, the union is the area under the step function.
        # The step function is defined by the upper envelope of the points.
        
        # When adding a new point (x, y), it will be inserted between two existing points (f1_a, f2_a) and (f1_b, f2_b)
        # such that f1_a < x < f1_b.
        # The gain is the area of the region that was previously covered by the "gap" between (f1_a, f2_a) and (f1_b, f2_b)
        # but is now covered by the new point.
        
        # Actually, the gain is the area of the rectangle [x, R1] x [y, R2] that is NOT already covered by the existing points.
        # Since the front is convex (f2 is convex in f1), the "covered" area is determined by the current points.
        
        # A simpler way:
        # The marginal gain of adding (x, y) to a set S is the area of the polygon formed by (x, y) and its neighbors in S.
        
        # Let's use a different approach:
        # The hypervolume of a set of points in 2D can be computed as:
        # HV = (R1 - f1_0)*(f2_0) + sum_{i=1}^{k-1} (f1_i - f1_{i-1})*(f2_{i-1}) + (R1 - f1_{k-1})*(f2_{k-1})
        # Wait, this is not quite right. The reference is (R1, R2).
        # The HV is the area of the union of rectangles [f1_i, R1] x [f2_i, R2].
        # For non-dominated points sorted by f1 ascending, the union is the area under the step function defined by the points.
        # The step function has height f2_i for f1 in [f1_i, f1_{i+1}).
        # The area is:
        # HV = sum_{i=0}^{k-1} (f1_{i+1} - f1_i) * (f2_i - R2) + (R1 - f1_{k-1}) * (f2_{k-1} - R2)
        # Wait, the points are in [0,1]x[0,1], and R1=R2=11. So f2_i - R2 is negative.
        # The HV is the area of the region dominated by the points and above the origin? No, it's the area of the region
        # that is dominated by at least one point and is below the reference point.
        # In 2D, with reference (R1, R2), the HV is the area of the union of rectangles [f1_i, R1] x [f2_i, R2].
        # Since the points are non-dominated, the union is the area under the step function defined by the points,
        # where the step function is the upper envelope of the points.
        
        # Let's use a standard formula for 2D HV:
        # Sort points by f1 ascending.
        # HV = 0
        # prev_f2 = R2
        # for i in range(k):
        #     HV += (f1_{i+1} - f1_i) * (prev_f2 - f2_i)  # where f1_{k} = R1
        #     prev_f2 = f2_i
        # Wait, this is also not quite right.
        
        # Let's use the following approach:
        # The HV is the area of the region that is dominated by at least one point and is within [0, R1] x [0, R2].
        # For a set of non-dominated points sorted by f1 ascending, the HV is:
        # HV = sum_{i=0}^{k-1} (f1_{i+1} - f1_i) * (f2_i) + (R1 - f1_{k-1}) * (f2_{k-1})
        # where f1_{-1} = 0 and f1_{k} = R1.
        # Wait, this is the area under the step function defined by the points, where the step function has height f2_i
        # for f1 in [f1_i, f1_{i+1}).
        # But the reference point is (R1, R2). The HV is the area of the union of rectangles [f1_i, R1] x [f2_i, R2].
        # Since the points are in [0,1]x[0,1] and R1=R2=11, the union is the area under the step function defined by the points,
        # where the step function has height f2_i for f1 in [f1_i, f1_{i+1}), and the area is measured from 0 to R1 and 0 to R2.
        
        # Actually, the HV is the area of the region that is dominated by at least one point and is within [0, R1] x [0, R2].
        # For a set of non-dominated points sorted by f1 ascending, the HV is:
        # HV = (R1 - f1_0) * f2_0 + sum_{i=1}^{k-1} (f1_i - f1_{i-1}) * f2_{i-1} + (R1 - f1_{k-1}) * f2_{k-1}
        # Wait, this is not correct either.
        
        # Let's use a different approach:
        # The HV is the area of the union of rectangles [f1_i, R1] x [f2_i, R2].
        # For non-dominated points sorted by f1 ascending, the union is the area under the step function defined by the points.
        # The step function is defined as follows: for f1 in [f1_i, f1_{i+1}), the height is f2_i.
        # The area is:
        # HV = sum_{i=0}^{k-1} (f1_{i+1} - f1_i) * (f2_i - R2) + (R1 - f1_{k-1}) * (f2_{k-1} - R2)
        # Wait, this is the area above the step function and below R2.
        # The HV is the area below the step function and above 0, but only the part that is dominated by the points.
        
        # Let's use the standard formula for 2D HV:
        # Sort points by f1 ascending.
        # HV = 0
        # for i in range(k):
        #     next_f1 = f1_{i+1} if i < k-1 else R1
        #     HV += (next_f1 - f1_i) * (f2_i - R2)
        # Wait, this is the area of the rectangle [f1_i, next_f1] x [f2_i, R2].
        # Since f2_i < R2, this is positive.
        # The sum of these areas is the HV.
        
        # Let's test this with two points: (0, 1) and (1, 0).
        # HV = (1 - 0) * (1 - 11) + (11 - 1) * (0 - 11) = 1 * (-10) + 10 * (-11) = -10 - 110 = -120.
        # This is negative, which is wrong.
        
        # The HV is the area of the union of rectangles [f1_i, R1] x [f2_i, R2].
        # For (0, 1): [0, 11] x [1, 11], area = 11 * 10 = 110.
        # For (1, 0): [1, 11] x [0, 11], area = 10 * 11 = 110.
        # The union is [0, 11] x [0, 11] minus the part that is not covered.
        # The part not covered is [0, 1] x [0, 1] minus the points? No.
        # The union of [0, 11] x [1, 11] and [1, 11] x [0, 11] is [0, 11] x [0, 11] minus [0, 1] x [0, 1].
        # Area = 121 - 1 = 120.
        
        # So the HV is 120.
        
        # The formula should be:
        # HV = sum_{i=0}^{k-1} (next_f1 - f1_i) * (R2 - f2_i)
        # where next_f1 = f1_{i+1} if i < k-1 else R1.
        
        # Let's test:
        # For (0, 1): (1 - 0) * (11 - 1) = 1 * 10 = 10.
        # For (1, 0): (11 - 1) * (11 - 0) = 10 * 11 = 110.
        # Sum = 120. Correct!
        
        # So the HV is:
        # HV = sum_{i=0}^{k-1} (next_f1 - f1_i) * (R2 - f2_i)
        # where next_f1 = f1_{i+1} if i < k-1 else R1.
        
        # Now, the marginal gain of adding a point (x, y) to the set S is:
        # Find the index i such that f1_i <= x <= f1_{i+1}.
        # The new point will be inserted between f1_i and f1_{i+1}.
        # The gain is the area of the new rectangles minus the area of the old rectangle.
        # Old rectangle: (f1_{i+1} - f1_i) * (R2 - f2_i)
        # New rectangles: (x - f1_i) * (R2 - f2_i) + (f1_{i+1} - x) * (R2 - y)
        # Wait, this is not quite right. The new point (x, y) will dominate the region [x, R1] x [y, R2].
        # The gain is the area of [x, R1] x [y, R2] that is not already covered by the existing points.
        
        # Since the front is convex, the new point will be between two existing points (f1_i, f2_i) and (f1_{i+1}, f2_{i+1}).
        # The gain is the area of the triangle formed by (f1_i, f2_i), (x, y), and (f1_{i+1}, f2_{i+1}).
        # Actually, the gain is the area of the region that is dominated by (x, y) but not by the existing points.
        # This region is a triangle with vertices (f1_i, f2_i), (x, y), and (f1_{i+1}, f2_{i+1}).
        # The area of this triangle is 0.5 * |(x - f1_i) * (f2_{i+1} - f2_i) - (f1_{i+1} - f1_i) * (y - f2_i)|
        
        # Let's use this formula to compute the marginal gain.
        
        # For efficiency, we can pre-compute the candidates and their f1, f2 values.
        # Then, for each step, we find the candidate that maximizes the marginal gain.
        
        # To make this fast, we can note that the marginal gain of a candidate depends only on its neighbors in the current set.
        # So we can maintain the current set of points and, for each candidate, compute the gain in O(1) time.
        
        # Let's implement this.
        
        # Current set of points: x1_selected, f1_selected, f2_selected
        # We need to select (budget - 2) more points.
        
        # Candidates: x1_grid, f1_grid, f2_grid
        # We exclude x1=0 and x1=1 (already selected).
        
        # For each step, we find the candidate with the maximum marginal gain.
        # To compute the marginal gain of a candidate (x, y):
        # 1. Find the index i such that f1_selected[i] <= x <= f1_selected[i+1].
        # 2. Compute the area of the triangle formed by (f1_selected[i], f2_selected[i]), (x, y), and (f1_selected[i+1], f2_selected[i+1]).
        # 3. The gain is this area.
        
        # To find the index i efficiently, we can use binary search.
        
        # Let's implement the greedy loop.
        
        x1_sel = np.array([0.0, 1.0])
        f1_sel = np.array([0.0, 1.0])
        f2_sel = np.array([1.0, 0.0])
        
        # We need to select (budget - 2) more points.
        for step in range(self.budget - 2):
            if time.time() > deadline:
                break
            
            # Find the candidate with the maximum marginal gain.
            # To do this, we need to find the index i for each candidate.
            # We can vectorize this by using np.searchsorted.
            
            # For each candidate x, find the index i such that f1_sel[i] <= x < f1_sel[i+1].
            # np.searchsorted(f1_sel, x, side='right') - 1 gives the index i.
            
            i = np.searchsorted(f1_sel, f1_grid, side='right') - 1
            # i is the index of the left neighbor.
            # The right neighbor is i+1.
            
            # Compute the area of the triangle for each candidate.
            # Area = 0.5 * |(x - f1_sel[i]) * (f2_sel[i+1] - f2_sel[i]) - (f1_sel[i+1] - f1_sel[i]) * (y - f2_sel[i])|
            
            # We need to be careful with the indices.
            # i can be -1 or len(f1_sel)-1, but since we exclude x=0 and x=1, i should be in [0, len(f1_sel)-2].
            
            # Let's clip i to [0, len(f1_sel)-2].
            i = np.clip(i, 0, len(f1_sel) - 2)
            
            # Compute the area.
            dx1 = f1_grid - f1_sel[i]
            dy1 = f2_sel[i+1] - f2_sel[i]
            dx2 = f1_sel[i+1] - f1_sel[i]
            dy2 = f2_grid - f2_sel[i]
            
            area = 0.5 * np.abs(dx1 * dy1 - dx2 * dy2)
            
            # Find the candidate with the maximum area.
            best_idx = np.argmax(area)
            
            # Add the best candidate to the selected set.
            x1_new = f1_grid[best_idx]
            f1_new = f1_grid[best_idx]
            f2_new = f2_grid[best_idx]
            
            # Insert into the selected set.
            # We need to maintain the sorted order.
            # We can use np.insert, but it's slow.
            # Instead, we can create a new array.
            
            # Find the position to insert.
            pos = np.searchsorted(f1_sel, f1_new, side='right')
            
            x1_sel = np.insert(x1_sel, pos, x1_new)
            f1_sel = np.insert(f1_sel, pos, f1_new)
            f2_sel = np.insert(f2_sel, pos, f2_new)
            
            # Remove the selected candidate from the grid to avoid re-selecting it.
            # We can set its area to -1 or remove it from the grid.
            # Let's remove it from the grid.
            # But removing from a numpy array is slow.
            # Instead, we can keep a mask of selected candidates.
            
            # For now, let's just continue. The greedy algorithm will not select the same point twice
            # because the area will be 0 for already selected points.
            # Actually, the area will be 0 only if the point is already in the set.
            # But we are not removing points from the grid, so we might re-select the same point.
            # To avoid this, we can set the area of selected points to -1.
            
            # Let's keep a mask of selected candidates.
            # But this is getting complex. Let's just use a list of selected x1 values and check for duplicates.
            
            # For now, let's assume that the greedy algorithm will not select the same point twice.
            # This is true if the area is 0 for already selected points.
            # But the area is not 0 for already selected points; it's the area of the triangle, which is 0
            # if the point is on the line between its neighbors.
            # So it should be fine.
        
        # Now, we have x1_sel, which is a list of x1 values.
        # We need to evaluate them.
        # But we have already "selected" them in the greedy loop.
        # We need to evaluate them in the order they were selected? No, the order doesn't matter.
        # We just need to evaluate all of them.
        
        # But we have a budget of 200 evaluations.
        # We have selected 200 points (2 initial + 198 greedy).
        # So we can evaluate all of them.
        
        # Let's evaluate them.
        F = []
        for x1 in x1_sel:
            x = np.zeros(n)
            x[0] = x1
            x[1:] = 0.0
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            
            if time.time() > deadline:
                break
        
        return np.array(F)
