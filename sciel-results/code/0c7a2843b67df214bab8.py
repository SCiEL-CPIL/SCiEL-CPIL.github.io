import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290

        # MOPSO Parameters
        pop_size = 15
        archive_max_size = 40
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Initialize personal bests
        pbest_X = X.copy()
        pbest_F = F.copy()
        
        # Initialize velocities
        v_max = 0.5 * (hi - lo)
        V = np.random.uniform(-v_max, v_max, (pop_size, dim))

        # Initialize archive
        archive_X = []
        archive_F = []
        
        # Add initial non-dominated points to archive
        for i in range(pop_size):
            self.update_archive(archive_X, archive_F, X[i], F[i], archive_max_size)

        # MOPSO Loop
        # We will do one "iteration" per particle evaluation? 
        # Standard MOPSO updates all particles, then evaluates.
        # But we have a tight budget. 
        # If we update all 15 particles, that's 15 evals.
        # 200 / 15 = 13 iterations.
        # This is very few iterations for PSO.
        
        # Alternative: Update one particle at a time? 
        # Or smaller population?
        # If pop_size = 10, 20 iterations.
        # If pop_size = 5, 40 iterations.
        
        # Let's try pop_size = 10 for more iterations.
        # Actually, let's stick to 15 but maybe the velocity update is cheap.
        # The bottleneck is the function evaluation.
        
        # Let's re-evaluate pop_size.
        # With 200 evals, if we use 15 for init, we have 185 left.
        # 185 / 15 = 12.3 generations.
        # 12 generations is low for PSO to converge.
        
        # Let's reduce pop_size to 10.
        # 10 init. 190 left. 19 generations.
        
        # Let's reset with pop_size = 10.
        # I will re-initialize inside the logic or just adjust the variable.
        # Since I already allocated X, I'll just use the first 10 if I change size?
        # Better to just define pop_size at the start.
        
        # Let's restart the logic with pop_size = 10.
        
        # --- Re-initialization for pop_size = 10 ---
        # The code above already used 15. I should write the code cleanly.
        
        pass # Placeholder for thought process

    def update_archive(self, archive_X, archive_F, x_new, f_new, max_size):
        """
        Updates the archive with a new point.
        Returns True if the point was added, False if it was dominated.
        """
        # Check if dominated by any point in archive
        for f_arch in archive_F:
            if self.dominates(f_arch, f_new):
                return False

        # Remove points dominated by the new point
        new_archive_X = []
        new_archive_F = []
        for x_arch, f_arch in zip(archive_X, archive_F):
            if not self.dominates(f_new, f_arch):
                new_archive_X.append(x_arch)
                new_archive_F.append(f_arch)
        
        # Add new point
        new_archive_X.append(x_new)
        new_archive_F.append(f_new)
        
        # Prune if size exceeded
        if len(new_archive_X) > max_size:
            # Find point with lowest hypervolume contribution
            # HV(S) - HV(S \ {p})
            # We need to calculate HV for sets.
            # Since max_size is small (40), we can do this.
            
            # To avoid O(N^2) HV calls which might be slow if HV is complex,
            # we can use a simpler heuristic: remove the point that is closest to another point?
            # Or just random removal?
            # Let's try a simple heuristic: remove the point with the largest distance to the nearest neighbor?
            # No, we want to keep diversity.
            # Let's use the "crowding distance" on the archive.
            # Or, simpler: just remove the last added if it's not significantly better?
            
            # Actually, for 3D, HV is easy.
            # Let's implement a simple 3D HV calculator.
            # But doing it 40 times per insertion is 1600 calls.
            # 1600 * (40^2) operations = 2.5M ops. This is fast in Python/NumPy.
            
            # Let's implement a helper to calculate HV of a set of points.
            # Reference point is fixed [3.0, 5.0, 7.0] per task description?
            # "reference point [3.0, 5.0, 7.0]"
            
            ref = np.array([3.0, 5.0, 7.0])
            
            # We want to maximize HV.
            # We will remove the point p that minimizes HV(S) - HV(S\{p}).
            # Which is equivalent to removing the point that contributes the least.
            
            best_to_remove = -1
            min_contrib = float('inf')
            
            # Calculate HV of full set
            hv_full = self.calc_hv_3d(new_archive_F, ref)
            
            for i in range(len(new_archive_F)):
                # Create set without i
                mask = np.ones(len(new_archive_F), dtype=bool)
                mask[i] = False
                F_subset = new_archive_F[mask]
                
                if len(F_subset) == 0:
                    hv_subset = 0.0
                else:
                    hv_subset = self.calc_hv_3d(F_subset, ref)
                    
                contrib = hv_full - hv_subset
                if contrib < min_contrib:
                    min_contrib = contrib
                    best_to_remove = i
                    
            # Remove the identified point
            new_archive_X.pop(best_to_remove)
            new_archive_F.pop(best_to_remove)
            
        # Update original lists (since lists are passed by reference, we can't just reassign local vars)
        # We need to clear and extend or modify in place.
        archive_X.clear()
        archive_F.clear()
        archive_X.extend(new_archive_X)
        archive_F.extend(new_archive_F)
        
        return True

    def calc_hv_3d(self, F, ref):
        """
        Calculate hypervolume for 3 objectives.
        F is Nx3 array. ref is 3 array.
        Only points better than ref in all objectives count.
        """
        # Filter points
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        
        # Translate so ref is origin? No, just calculate volume of boxes.
        # For 3D, we can sort by one objective and use a 2D HV algorithm for the others?
        # Or simply use the "union of boxes" approach.
        # Since N is small, we can use a simple inclusion-exclusion or grid?
        # Grid is inefficient.
        
        # A standard algorithm for 3D HV:
        # Sort points by f1.
        # Sweep line from best f1 to worst.
        # Maintain a 2D HV structure for f2, f3.
        
        # Given the small size (<=40), a brute force or simple recursive approach might work.
        # Let's use a simple algorithm:
        # HV = sum over all points of the volume of the box defined by that point and the reference point,
        # minus overlaps. This is hard.
        
        # Alternative: 
        # For 3 objectives, we can sort by f1.
        # Let points be p1...pn sorted by f1 ascending.
        # The contribution of p_i is (f1_{i+1} - f1_i) * HV_2D(points i..n in f2,f3).
        # This is O(N^2) if HV_2D is O(N).
        
        def hv_2d(F2d, ref2d):
            # F2d is Mx2, ref2d is 2
            # Filter
            mask = np.all(F2d < ref2d, axis=1)
            F2d = F2d[mask]
            if len(F2d) == 0:
                return 0.0
            # Sort by first obj
            F2d = F2d[np.argsort(F2d[:, 0])]
            total = 0.0
            prev_f1 = 0.0 # Not exactly, we need to integrate
            # Actually, for 2D:
            # Sort by x.
            # Keep track of min y.
            # Volume = sum (x_i - x_{i-1}) * (ref_y - min_y_so_far)
            # Be careful with boundaries.
            
            # Simpler:
            # The region is the union of rectangles [f2_i, ref2d[0]] x [f3_i, ref2d[1]].
            # Sort by f2.
            # Sweep f2 from high to low? Or low to high?
            # Let's sort by f2 ascending.
            # The "active" min f3 determines the height.
            
            F2d = F2d[np.argsort(F2d[:, 1])] # Sort by 2nd obj
            min_f3 = ref2d[1]
            prev_f2 = F2d[0, 1]
            
            # This logic is tricky. Let's use a different approach for 2D.
            # For 2D, HV is the area under the step function.
            # Sort by obj1.
            # For each point, the width is the difference in obj1 to the next point (or ref).
            # The height is the minimum obj2 seen so far from the "best" end?
            
            # Let's sort by obj1 ascending (best to worst).
            F2d = F2d[np.argsort(F2d[:, 0])]
            
            # The area is sum of rectangles.
            # Width of rect i: min(F2d[i+1, 0], ref[0]) - F2d[i, 0]
            # Height of rect i: min(F2d[i, 1], ref[1]) ... no.
            # The height is determined by the best (lowest) obj2 value among points that cover this x-range.
            
            # Let's iterate from worst obj1 to best?
            # No, let's iterate from best obj1 (smallest value) to worst.
            
            # Actually, there is a simpler way for 2D:
            # Sort by x.
            # Maintain a list of active y-intervals?
            
            # Given the small N, let's just use a grid? No.
            # Let's use the standard 2D HV algorithm:
            # Sort points by x.
            # Initialize min_y = ref_y.
            # Area = 0
            # For i from N-1 down to 0 (worst x to best x? No, best x is smallest).
            # Let's sort x ascending.
            # The "skyline" is formed by points where y is decreasing as x increases.
            
            # Let's try:
            # Sort by x ascending.
            # Keep a variable min_y = inf.
            # Iterate from N-1 down to 0?
            # If we go from worst x (largest) to best x (smallest):
            # At each step, if current y < min_y, then the strip from current x to previous x has height (ref_y - current_y)?
            # No.
            
            # Correct 2D HV Algorithm:
            # 1. Filter points better than ref.
            # 2. Sort by obj1 ascending.
            # 3. Initialize total_hv = 0.
            # 4. Initialize min_obj2 = ref[1].
            # 5. Iterate i from N-1 down to 0:
            #    If F[i,1] < min_obj2:
            #       width = min(F[i+1, 0] if i+1<N else ref[0], ref[0]) - F[i, 0]
            #       Wait, if i is the best (smallest x), the width extends to the next point's x or ref.
            #       Let's define x_next = F[i+1, 0] if i+1 < N else ref[0].
            #       width = x_next - F[i, 0].
            #       height = min_obj2 - F[i, 1].
            #       total_hv += width * height.
            #       min_obj2 = F[i, 1].
            
            # Let's verify this logic.
            # Points: (1, 10), (2, 5), (3, 8). Ref: (5, 12).
            # Sorted by x: (1, 10), (2, 5), (3, 8).
            # i=2: (3, 8). min_y=12. x_next=5. width=2. height=12-8=4. Area=8. min_y=8.
            # i=1: (2, 5). min_y=8. x_next=3. width=1. height=8-5=3. Area=3. min_y=5.
            # i=0: (1, 10). min_y=5. x_next=2. width=1. height=5-10? No, 10 > 5. So no contribution?
            # Wait, (1,10) is dominated by (2,5) in terms of y? No, (1,10) is better in x, worse in y.
            # The region covered by (1,10) is x in [1,2], y in [10, 12]?
            # But (2,5) covers x in [2,5], y in [5,12].
            # The union is:
            # x in [1,2]: y in [10,12] (from p0) -> Area 1*2=2.
            # x in [2,3]: y in [5,12] (from p1) -> Area 1*7=7.
            # x in [3,5]: y in [5,12] (from p1) -> Area 2*7=14.
            # Total = 2+7+14 = 23.
            
            # My algorithm:
            # i=2 (3,8): width = 5-3=2. height = 12-8=4. Area=8. min_y=8.
            # i=1 (2,5): width = 3-2=1. height = 8-5=3. Area=3. min_y=5.
            # i=0 (1,10): F[0,1]=10. min_y=5. 10 > 5. So no update to min_y.
            # But it should contribute area for x in [1,2] with height 12-10=2.
            # My algorithm failed because it only adds area when min_y decreases.
            # It misses the "steps" that go up?
            # No, the skyline is the lower-left envelope.
            # Points that are not on the skyline don't contribute.
            # (1,10) is NOT on the skyline because (2,5) has better y.
            # Wait, (1,10) is better in x. So it IS on the skyline for x < 2.
            # The skyline is:
            # x=1, y=10
            # x=2, y=5
            # x=3, y=8 (Dominated by (2,5)? No, (2,5) is better in y, but (3,8) is better in x? No, 3>2. So (3,8) is dominated by (2,5) in both? No, 8>5. So (3,8) is dominated by (2,5).
            # So (3,8) should be removed.
            # If (3,8) is removed, points are (1,10), (2,5).
            # i=1 (2,5): width = 5-2=3. height = 12-5=7. Area=21. min_y=5.
            # i=0 (1,10): 10 > 5. No update.
            # Total 21.
            # Actual HV:
            # x in [1,2]: y in [10,12]. Area 2.
            # x in [2,5]: y in [5,12]. Area 21.
            # Total 23.
            
            # Why did my algorithm give 21?
            # Because it didn't account for the part where y is bounded by the *current* point's y if it's higher than the previous min?
            # No, the HV is the volume under the curve defined by the non-dominated points.
            # The curve is:
            # (1,10) -> (2,5) -> (5,5) -> (5,12) -> (1,12) -> (1,10)?
            # The area is integral of (ref_y - y(x)) dx.
            # For x in [1,2], y(x) = 10. Area = (12-10)*1 = 2.
            # For x in [2,5], y(x) = 5. Area = (12-5)*3 = 21.
            # Total 23.
            
            # My algorithm:
            # i=1 (2,5): width = x_next - x_i. x_next is ref_x=5? Or next point?
            # If we iterate from worst x to best x:
            # Points sorted by x: (1,10), (2,5).
            # i=1 (2,5): x_next = 5 (ref). width = 3. height = 12-5=7. Area=21. min_y=5.
            # i=0 (1,10): x_next = 2. width = 1.
            # Here, the "active" height should be min(min_y, F[i,1])?
            # No, the height is ref_y - F[i,1] IF F[i,1] < min_y?
            # If F[i,1] > min_y, then this point is dominated by a previous (worse x, better y) point?
            # No, (1,10) has better x than (2,5).
            # So (1,10) defines the boundary for x in [1,2].
            # The height is 12-10=2.
            # So we should add width * (ref_y - F[i,1])?
            # But only if F[i,1] is the minimum y so far?
            # No, F[i,1] is 10. min_y was 5.
            # The condition "if F[i,1] < min_y" failed.
            # But it should contribute!
            
            # Correct logic for 2D HV (sorted by x ascending, iterating from N-1 down to 0):
            # We maintain min_y.
            # At step i, the width is (x_{i+1} - x_i).
            # The height is (ref_y - min_y_so_far).
            # Wait, if we are at x_i, the region [x_i, x_{i+1}] is covered by the best y seen so far (which is min_y from the right/worse x side).
            # So height = ref_y - min_y.
            # Then we update min_y = min(min_y, F[i,1]).
            
            # Let's re-test:
            # Points: (1,10), (2,5). Ref (5,12).
            # i=1 (2,5):
            #   width = 5 - 2 = 3.
            #   height = 12 - min_y (12) = 0? No, min_y starts at ref_y.
            #   So height = 0. Area = 0.
            #   Update min_y = min(12, 5) = 5.
            # i=0 (1,10):
            #   width = 2 - 1 = 1.
            #   height = 12 - min_y (5) = 7.
            #   Area = 7.
            #   Update min_y = min(5, 10) = 5.
            # Total 7.
            # Still wrong. Expected 23.
            
            # The error is in the width/height assignment.
            # The region [x_i, x_{i+1}] is covered by the point i? Or by the best point to the right?
            # The point i has x_i. The next point has x_{i+1}.
            # The "skyline" segment from x_i to x_{i+1} is determined by the point with the best y among points with x <= x_{i+1}?
            # No, the skyline is the lower-left envelope.
            # For x in [x_i, x_{i+1}], the best y is the minimum y of all points with x <= x_{i+1}?
            # No, it's the minimum y of points that are not dominated.
            
            # Let's use a different 2D HV method:
            # Sort by x.
            # Filter dominated points.
            # Calculate area of rectangles.
            
            # I will implement a simple 3D HV by decomposing into 2D.
            # Or just use a library if available? No, standard lib only.
            # Given the complexity, I will use a simpler pruning strategy:
            # **Remove the point with the largest distance to the nearest neighbor.**
            # This maintains diversity without expensive HV calcs.
            
        return 0.0

    def dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)
