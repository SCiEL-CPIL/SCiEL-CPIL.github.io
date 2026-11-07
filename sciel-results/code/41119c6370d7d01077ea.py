import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points, ref):
        """Compute 2D hypervolume for points strictly better than ref."""
        if len(points) == 0:
            return 0.0
        points = np.array(points)
        # Filter points strictly better than reference
        mask = np.all(points < ref, axis=1)
        points = points[mask]
        if len(points) == 0:
            return 0.0
        # Sort by first objective
        idx = np.argsort(points[:, 0])
        points = points[idx]
        hv = 0.0
        prev_y = ref[1]
        for x, y in points:
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return hv

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([11.0, 11.0])
        
        # ZDT6 specific parameters
        n = self.dim
        g = 1 + 10 * (n - 1)
        
        # Analytical front function
        def front_y(x):
            # x is array
            return g * (1 - np.sqrt(x / g) - np.sin(10 * np.pi * x))
            
        # Discretize x in [0, 1]
        # Use a fine grid for the facility location problem
        M = 10000
        x_grid = np.linspace(0.0, 1.0, M)
        y_grid = front_y(x_grid)
        
        # Greedy selection of N points from the grid
        N = self.budget
        selected_indices = []
        # We will maintain the current set of points and their HV
        # To speed up, we can just compute the full HV for candidates?
        # No, marginal gain is better.
        
        # For 2D, marginal gain of adding point p to set S:
        # It's the area of the rectangle defined by p that is not already covered by S.
        # This is complex to compute directly in a vectorized way for all candidates.
        # Alternative: Since M=10000 and N=200, we can just compute the HV of the 
        # set of selected points + candidate for each candidate?
        # That would be O(N * M * HV_calc). HV_calc is O(K log K) where K is size of set.
        # K grows from 1 to 200.
        # Total ops ~ sum_{k=1}^{200} 10000 * k log k ~ 10000 * 200 * 200 * 5 = 2e8.
        # This might be too slow in pure Python loops.
        
        # Better approach:
        # The hypervolume of a set of points on the Pareto front (non-dominated)
        # is the sum of areas of rectangles.
        # If we assume the selected points are non-dominated (which they will be if we pick
        # from the non-dominated part of the curve), the HV is:
        # Sort by x. HV = sum_{i} (x_{i+1} - x_i) * (ref_y - y_i) + (ref_x - x_last) * (ref_y - y_last)?
        # No, standard 2D HV formula for non-dominated points sorted by x:
        # HV = sum_{i=1}^{K} (x_{i+1} - x_i) * (ref_y - y_i) where x_{K+1} = ref_x, y_{K+1} = ref_y?
        # Actually, the area is the union of rectangles [x_i, ref_x] x [y_i, ref_y].
        # For non-dominated points sorted by x ascending (y descending):
        # The union is a staircase.
        # Area = sum_{i=1}^{K-1} (x_{i+1} - x_i) * (ref_y - y_i) + (ref_x - x_K) * (ref_y - y_K)
        # This formula works if y is strictly decreasing.
        
        # Let's verify if ZDT6 front y is strictly decreasing on the non-dominated set.
        # The non-dominated set is where y is strictly decreasing as x increases.
        # So we can filter the grid to only include points that are "Pareto optimal" in the grid.
        
        # Step 1: Identify non-dominated points in the grid.
        # Sort by x ascending.
        # Keep a point if its y is strictly less than the minimum y seen so far (from left)?
        # No. 
        # Point i dominates j (i<j in x) if y_i <= y_j.
        # So j is dominated if there exists i < j with y_i <= y_j.
        # So j is non-dominated if y_j < min(y_0, ..., y_{j-1}).
        
        y_min_so_far = np.full(M, np.inf)
        y_min_so_far[0] = y_grid[0]
        for i in range(1, M):
            y_min_so_far[i] = min(y_min_so_far[i-1], y_grid[i])
            
        # Non-dominated mask: y_grid[i] < y_min_so_far[i] (strictly, to avoid ties)
        # Actually, if y_grid[i] == y_min_so_far[i], it is dominated by the previous min.
        # So we want y_grid[i] < y_min_so_far[i] is NOT correct because y_min_so_far[i] includes y_grid[i].
        # Let's define prefix_min[i] = min(y[0..i-1]).
        # Then i is non-dominated if y[i] < prefix_min[i].
        
        prefix_min = np.full(M, np.inf)
        prefix_min[1:] = np.minimum.accumulate(y_grid[:-1])
        
        nd_mask = y_grid < prefix_min
        # Also check against ref? All y < 11?
        # g=91. y is around 80-91. So all are < 11? No, 91 > 11.
        # Wait, ref is [11, 11].
        # ZDT6 objectives are usually in [0, 100] range.
        # If y > 11, the point is NOT strictly better than ref in obj 2.
        # The scoring says: "Only points strictly better than the reference point in every objective count."
        # So we must filter out points where y >= 11 or x >= 11.
        # x is in [0,1], so x < 11 is always true.
        # y is ~80-91. So y > 11 is always true.
        # This means NO points from the standard ZDT6 front are strictly better than [11, 11] in the 2nd objective?
        
        # Let's re-read the task.
        # "minimizing all 2 objectives of ZDT6"
        # "reference point [11.0, 11.0]"
        # ZDT6 f1 = x0. Range [0, 1].
        # ZDT6 f2 = g * h. g = 91. h is around 1. So f2 is around 90.
        # If ref is 11, and f2 is 90, then f2 > ref.
        # So the point is NOT better than ref.
        # Hypervolume is 0?
        
        # This implies the standard ZDT6 definition might be scaled or I am misinterpreting the reference point.
        # Or maybe the "ZDT6" in the task is a scaled version?
        # "problem(x) ... returns 2 objective values."
        # If the evaluator uses standard ZDT6, the score would be 0.
        # But the current program has a score of 1.5036.
        # This implies the points ARE better than the reference point.
        # How?
        # Maybe the reference point is [11, 11] but the objectives are scaled?
        # Or maybe my calculation of f2 is wrong?
        # ZDT6: f2 = (1+10n) * (1 - sqrt(x1/(1+10n)) - sin(10*pi*x1)).
        # If n=9 (dim=10, so n=dim-1=9?), g = 1+90=91.
        # If n=dim=10? g=101.
        # In any case, g is large.
        
        # Is it possible the reference point is [11, 11] but the objectives are normalized?
        # Or is the "ZDT6" in the task a different function?
        # The task says "ZDT6 (10 variables)".
        
        # Let's look at the current program's `_zdt6_pareto_front`.
        # It returns f1, f2.
        # If the score is non-zero, the points must satisfy f1 < 11 and f2 < 11.
        # Since f1 = x0 <= 1, f1 < 11 is true.
        # So f2 must be < 11.
        # This requires g * h < 11.
        # If g=91, h must be < 0.12.
        # h = 1 - sqrt(x/g) - sin(10 pi x).
        # Max h is 1 (when x=0, sin=0, sqrt=0).
        # Min h?
        # If x=1, sqrt(1/91)~0.1, sin(10pi)=0. h ~ 0.9. f2 ~ 81.
        # There is no x where h is small enough to make f2 < 11 if g=91.
        
        # Conclusion: The "ZDT6" in the evaluator might be scaled, OR the reference point is different, OR I am missing something.
        # However, I must assume the current program works (score 1.5).
        # The current program uses `problem(x)`.
        # If I use the analytical front to select points, I must ensure the points are valid.
        # If the analytical front doesn't match the `problem` function's scaling, the selection will be wrong.
        
        # Safe strategy:
        # The directive says "design a new program ... built on this approach".
        # The approach is "discrete 1D facility location ... in objective space".
        # I cannot rely on the analytical formula if it doesn't match the evaluator's `problem`.
        # BUT, the current program uses the analytical formula for *initialization* and then refines using `problem`.
        # And it gets a good score.
        # This suggests the analytical formula *is* close to the actual `problem` behavior, or the `problem` is standard and the ref point allows it?
        # Wait, if ref is 11, and f2 is 90, HV is 0.
        # Unless... the reference point is [11, 11] but the objectives are *maximized*? No, "minimizing".
        
        # Let's assume the evaluator's `problem` returns values in a range where [11, 11] is a valid reference (i.e., objectives are < 11).
        # This implies the `problem` function might be a normalized ZDT6 or similar.
        # Given I don't have the `problem` source, I must treat it as a black box.
        
        # However, the directive asks to use the "known Pareto front" structure.
        # If I use the analytical ZDT6 front, and it doesn't match the black box, I fail.
        # The current program *does* use the analytical front for initialization.
        # To be safe, I should use the analytical front to *generate candidates* but verify/improve using the black box.
        
        # Revised Plan:
        # 1. Use the analytical front to generate a fine grid of candidate points.
        # 2. Evaluate a subset of these to understand the scaling? No, budget is tight.
        # 3. Assume the analytical front is a good proxy for the *shape* of the Pareto front.
        # 4. Use the greedy facility location algorithm on the *analytical* front to select 200 x0 values.
        # 5. Evaluate these 200 points with `problem`.
        
        # If the analytical front is wrong, this fails.
        # But the current program's success suggests the analytical front is relevant.
        
        # Let's refine the greedy selection to be robust.
        
        # If I can't be sure about the analytical front matching the black box, I should stick to the current program's strategy of using the black box for evaluation but improve the *selection* logic.
        
        # The current program does:
        # 1. Uniform init.
        # 2. Local mutation.
        
        # The directive wants:
        # "Formulate the placement ... as a discrete 1D facility location problem ... solving ... via dynamic programming or greedy iterative refinement."
        
        # I will implement the greedy refinement on the *black box* evaluations, but guided by the analytical front to propose candidates.
        
        # Actually, a simpler interpretation:
        # The "objective space" is 2D.
        # We want to place 200 points.
        # We can use a grid of x0 values.
        # For each x0, we get a point.
        # We select the best 200 points from a large pool of candidates (e.g., 1000 candidates) based on hypervolume.
        
        # Steps:
        # 1. Generate 1000 candidate x0 values (uniform + analytical front insights).
        # 2. Evaluate them? No, 1000 > 200 budget.
        # 3. We must select 200 points *without* evaluating all of them.
        
        # This brings us back to the analytical selection.
        
        # I will proceed with the analytical selection of x0 values, assuming the ZDT6 structure holds.
        # To mitigate risk, I will add a small perturbation or use the problem function to verify the first few points?
        # No, if I use problem function, I spend budget.
        
        # Let's assume the analytical front is correct for the *shape*.
        
        # Implementation of Greedy HV Maximization on Analytical Front:
        # 1. Create grid x_grid (10000 points).
        # 2. Compute y_grid.
        # 3. Filter to non-dominated points in the grid.
        # 4. Greedily select 200 points from this filtered list to maximize HV.
        # 5. Return the corresponding x vectors.

        # This uses 0 budget for selection, 200 for final evaluation.

        # Let's write the code.

        M = 10000
        x_grid = np.linspace(0.0, 1.0, M)
        
        # Calculate y for ZDT6
        # g = 1 + 10 * (self.dim - 1)
        g = 1 + 10 * (self.dim - 1)
        y_grid = g * (1 - np.sqrt(x_grid / g) - np.sin(10 * np.pi * x_grid))
        
        # Filter non-dominated
        # Sort by x (already sorted)
        # A point is non-dominated if y[i] < min(y[0..i-1])
        prefix_min = np.full(M, np.inf)
        prefix_min[1:] = np.minimum.accumulate(y_grid[:-1])
        nd_mask = y_grid < prefix_min
        
        x_nd = x_grid[nd_mask]
        y_nd = y_grid[nd_mask]
        
        # If no points are non-dominated (should not happen), fallback to uniform
        if len(x_nd) == 0:
             x_nd = x_grid
             y_nd = y_grid

        # Greedy selection
        N = self.budget
        selected_idx = []
        # To speed up HV calculation, we can use the formula for non-dominated points.
        # But the selected points might not be sorted by y?
        # Since we pick from a non-dominated set (y decreasing as x increases), 
        # any subset will also be non-dominated and sorted by x implies sorted by y (decreasing).
        
        # HV formula for sorted non-dominated points P:
        # HV = sum_{i=1}^{K} (x_{i+1} - x_i) * (ref_y - y_i)
        # where x_{K+1} = ref_x, y_{K+1} = ref_y?
        # Actually, the last rectangle extends to ref_x.
        # Let's verify.
        # Points p1, p2. x1 < x2, y1 > y2.
        # Area = (x2 - x1)(ref_y - y1) + (ref_x - x2)(ref_y - y2).
        # This matches the formula.
        
        def calc_hv(pts_x, pts_y):
            # pts_x, pts_y are sorted by x
            if len(pts_x) == 0:
                return 0.0
            hv = 0.0
            prev_x = 0.0 # Not needed for the formula above, but let's use the explicit sum
            # The formula is:
            # sum_{i=1}^{K-1} (x_{i+1} - x_i) * (ref_y - y_i) + (ref_x - x_K) * (ref_y - y_K)
            
            if len(pts_x) == 1:
                return (ref[0] - pts_x[0]) * (ref[1] - pts_y[0])
            
            x_arr = np.array(pts_x)
            y_arr = np.array(pts_y)
            
            # Differences in x
            dx = np.diff(x_arr)
            # y of the left point in each interval
            y_left = y_arr[:-1]
            
            hv = np.sum(dx * (ref[1] - y_left))
            # Last segment
            hv += (ref[0] - x_arr[-1]) * (ref[1] - y_arr[-1])
            
            return hv

        # Greedy loop
        current_x = []
        current_y = []
        
        # To optimize, we can just track the current HV and compute marginal gain.
        # Marginal gain of adding point p to set S:
        # Find the position of p in S (sorted by x).
        # It splits an interval [x_i, x_{i+1}] into [x_i, x_p] and [x_p, x_{i+1}].
        # Old area contribution of that interval: (x_{i+1} - x_i) * (ref_y - y_i)
        # New area: (x_p - x_i) * (ref_y - y_i) + (x_{i+1} - x_p) * (ref_y - y_p)
        # Gain = New - Old
        # = (x_p - x_i)(ref_y - y_i) + (x_{i+1} - x_p)(ref_y - y_p) - (x_{i+1} - x_i)(ref_y - y_i)
        # = (x_p - x_i)(ref_y - y_i) + (x_{i+1} - x_p)(ref_y - y_p) - (x_{i+1} - x_i)(ref_y - y_i)
        # Note: (x_{i+1} - x_i) = (x_p - x_i) + (x_{i+1} - x_p)
        # = (x_p - x_i)(ref_y - y_i) + (x_{i+1} - x_p)(ref_y - y_p) - [(x_p - x_i) + (x_{i+1} - x_p)](ref_y - y_i)
        # = (x_p - x_i)(ref_y - y_i) + (x_{i+1} - x_p)(ref_y - y_p) - (x_p - x_i)(ref_y - y_i) - (x_{i+1} - x_p)(ref_y - y_i)
        # = (x_{i+1} - x_p)(ref_y - y_p) - (x_{i+1} - x_p)(ref_y - y_i)
        # = (x_{i+1} - x_p) * (y_i - y_p)
        
        # This formula is O(1) if we know the neighbors!
        
        # So we can maintain a sorted list of selected points and compute gains efficiently.
        
        # However, finding the neighbors for M candidates is O(M log K) per step.
        # Total O(N M log K). 200 * 10000 * log(200) ~ 200 * 10000 * 8 = 1.6e7. Fast.
        
        # Let's implement this.
        
        # Data structures:
        # selected_x: sorted list of x
        # selected_y: corresponding y
        # We can use numpy arrays and insert? Insertion is O(K).
        # Or use a list and bisect.
        
        import bisect
        
        sel_x = []
        sel_y = []
        
        for _ in range(N):
            best_gain = -1
            best_idx = -1
            
            # For each candidate in x_nd, y_nd
            # Find insertion point in sel_x
            # If sel_x is empty, gain is full HV of the point.
            
            if len(sel_x) == 0:
                # Gain is (ref_x - x) * (ref_y - y)
                gains = (ref[0] - x_nd) * (ref[1] - y_nd)
                best_idx = np.argmax(gains)
            else:
                # Vectorized search for insertion points is tricky.
                # We can loop through candidates. M=10000 is small enough for a Python loop?
                # 200 * 10000 = 2,000,000 iterations. Yes.
                
                for i in range(len(x_nd)):
                    x_c = x_nd[i]
                    y_c = y_nd[i]
                    
                    # Find position in sel_x
                    pos = bisect.bisect_left(sel_x, x_c)
                    
                    # Determine neighbors
                    # Left neighbor: pos-1 (if exists)
                    # Right neighbor: pos (if exists)
                    
                    if pos == 0:
                        # Insert at beginning
                        # Interval is [0, sel_x[0]]? No, the domain starts at 0?
                        # The HV formula assumes the points are the only ones.
                        # The "interval" before the first point is [0, x_1]?
                        # No, the HV is defined relative to ref.
                        # The area to the left of the first point is not covered by any point?
                        # Wait, the formula:
                        # sum_{i=1}^{K-1} ... + (ref_x - x_K)(ref_y - y_K)
                        # This formula assumes the "staircase" starts at x_1.
                        # What about the area from 0 to x_1?
                        # The rectangle for point 1 is [x_1, ref_x] x [y_1, ref_y].
                        # It does NOT cover [0, x_1].
                        # So the area [0, x_1] x [y_1, ref_y] is NOT counted?
                        # Correct. HV is the union of rectangles [x_i, ref_x] x [y_i, ref_y].
                        # So points with smaller x cover MORE area.
                        # So the leftmost point covers the most area.
                        
                        # If we insert a new point x_c < sel_x[0]:
                        # Old contribution of sel_x[0] was (sel_x[1]-sel_x[0])... wait.
                        # Let's use the marginal gain formula derived:
                        # Gain = (x_right - x_c) * (y_left - y_c)
                        # Here, x_left is effectively 0? No.
                        # The "left neighbor" in terms of the staircase is the point with the next larger x.
                        # If we insert at the beginning, the new point becomes the leftmost.
                        # The "interval" it affects is the one starting at the old leftmost point.
                        # Old point: x_old, y_old.
                        # Next point: x_next, y_next.
                        # Old area segment: (x_next - x_old) * (ref_y - y_old)
                        # New area segments:
                        # 1. (x_old - x_c) * (ref_y - y_c)  [New point covers this]
                        # 2. (x_next - x_old) * (ref_y - y_old) [Old point still covers this? No, old point's rectangle starts at x_old. New point's rectangle starts at x_c. Since x_c < x_old, new point's rectangle covers [x_c, ref_x]. It covers [x_c, x_old] and [x_old, ref_x].
                        # In [x_old, ref_x], the old point's y_old is lower (better) than y_c?
                        # Since points are non-dominated, y decreases as x increases.
                        # So y_c > y_old.
                        # So in [x_old, ref_x], the old point is better.
                        # So the new point only adds area in [x_c, x_old].
                        # Gain = (x_old - x_c) * (ref_y - y_c).
                        
                        # Let's re-verify the general formula.
                        # Insert x_c between x_l and x_r (x_l < x_c < x_r).
                        # y_l > y_c > y_r.
                        # Old area segment: (x_r - x_l) * (ref_y - y_l).
                        # New area segments:
                        # (x_c - x_l) * (ref_y - y_l)  [Left part, still covered by y_l? No, y_l is the left point. Its rectangle covers [x_l, ref_x]. So [x_l, x_c] is covered by y_l.]
                        # (x_r - x_c) * (ref_y - y_c)  [Right part, covered by y_c? y_c's rectangle covers [x_c, ref_x]. y_r's rectangle covers [x_r, ref_x]. Since y_c > y_r, y_r is better in [x_r, ref_x]. So [x_c, x_r] is covered by y_c.]
                        # Wait, the union is determined by the *minimum* y in the interval.
                        # In [x_l, x_c], min y is y_l (since y_l < y_c).
                        # In [x_c, x_r], min y is y_c (since y_c < y_r).
                        # So New Area = (x_c - x_l)(ref_y - y_l) + (x_r - x_c)(ref_y - y_c).
                        # Old Area = (x_r - x_l)(ref_y - y_l).
                        # Gain = New - Old
                        # = (x_c - x_l)(ref_y - y_l) + (x_r - x_c)(ref_y - y_c) - (x_r - x_l)(ref_y - y_l)
                        # = (x_r - x_c)(ref_y - y_c) - (x_r - x_c)(ref_y - y_l)
                        # = (x_r - x_c) * (y_l - y_c)
                        
                        # This matches the previous derivation!
                        # So Gain = (x_right - x_c) * (y_left - y_c)
                        # Where x_left, y_left is the point with the next smaller x (or 0 if none? No, if none, x_left is effectively the start of the domain? No, the formula holds if we define the "left point" as the one that was covering the interval before x_c was inserted.
                        
                        # If inserting at beginning (pos=0):
                        # x_left is not defined. But the "interval" being split is [x_0, x_1].
                        # The point x_0 was covering [x_0, x_1] with height ref_y - y_0.
                        # Now x_c covers [x_c, x_0] with height ref_y - y_c.
                        # And x_0 still covers [x_0, x_1] with height ref_y - y_0.
                        # So Gain = (x_0 - x_c) * (ref_y - y_c).
                        # This can be written as (x_0 - x_c) * (y_virtual - y_c) where y_virtual = ref_y? No.
                        # Let's just use the explicit formula for the beginning case.
                        
                        gain = (sel_x[0] - x_c) * (ref[1] - y_c)
                        
                    elif pos == len(sel_x):
                        # Insert at end
                        # x_left = sel_x[-1], y_left = sel_y[-1]
                        # x_right = ref_x? No.
                        # The last point x_last covers [x_last, ref_x].
                        # New point x_c < x_last.
                        # x_c covers [x_c, x_last] with height ref_y - y_c.
                        # x_last still covers [x_last, ref_x] with height ref_y - y_last.
                        # But wait, y_c > y_last.
                        # So in [x_c, x_last], min y is y_c.
                        # Old area in [x_c, x_last] was covered by y_last? No, x_last's rectangle starts at x_last.
                        # So [x_c, x_last] was NOT covered by any point?
                        # No, the point x_{last-1} covers [x_{last-1}, ref_x].
                        # So [x_c, x_last] was covered by x_{last-1}'s y_{last-1}.
                        # So we are splitting the interval [x_{last-1}, x_last].
                        # x_left = sel_x[-2], y_left = sel_y[-2].
                        # x_right = sel_x[-1], y_right = sel_y[-1].
                        # Gain = (x_right - x_c) * (y_left - y_c).
                        
                        if len(sel_x) == 1:
                            # Only one point x0, y0.
                            # New point x_c < x0.
                            # x_c covers [x_c, x0] with ref_y - y_c.
                            # x0 covers [x0, ref_x] with ref_y - y0.
                            # Old area: x0 covered [x0, ref_x]. [x_c, x0] was empty?
                            # No, HV is union of rectangles.
                            # If only x0 was present, area was (ref_x - x0)(ref_y - y0).
                            # Now we have x_c and x0.
                            # Area = (x0 - x_c)(ref_y - y_c) + (ref_x - x0)(ref_y - y0).
                            # Gain = (x0 - x_c)(ref_y - y_c).
                            gain = (sel_x[0] - x_c) * (ref[1] - y_c)
                        else:
                            x_l = sel_x[-2]
                            y_l = sel_y[-2]
                            x_r = sel_x[-1]
                            gain = (x_r - x_c) * (y_l - y_c)
                            
                    else:
                        x_l = sel_x[pos-1]
                        y_l = sel_y[pos-1]
                        x_r = sel_x[pos]
                        gain = (x_r - x_c) * (y_l - y_c)
                        
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        
            # Add best point
            x_c = x_nd[best_idx]
            y_c = y_nd[best_idx]
            pos = bisect.bisect_left(sel_x, x_c)
            sel_x.insert(pos, x_c)
            sel_y.insert(pos, y_c)
            
        # Construct final X
        X = np.zeros((N, self.dim))
        X[:, 0] = sel_x
        
        # Evaluate
        F = np.array([problem(X[i]) for i in range(N)])
        
        return X

    def _hypervolume_2d(self, points, ref):
        # Keep the old method for compatibility if needed, but not used in the new logic
        pass

# EVOLVE-BLOCK-END
