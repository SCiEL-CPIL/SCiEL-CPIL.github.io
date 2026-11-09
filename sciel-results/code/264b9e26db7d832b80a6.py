"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HypE (Hypervolume-based Epsilon) for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use a fixed seed for reproducibility within the run, but distinct from the evaluator's seed
        # The evaluator sets np.random.seed(seed), so we should use that or a derived one.
        # However, the contract says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # So we can use np.random directly or default_rng. Let's use default_rng for better stats if available,
        # but to be safe and consistent with "standard library, NumPy", we'll use np.random if we want to leverage the global seed,
        # or just use a fixed seed for the optimizer's internal logic to ensure determinism regardless of global state?
        # Actually, using the global state is usually better for evolutionary algorithms to explore different regions per seed.
        # But the contract says "For seeds 0 and 1...". It implies the global seed is set.
        # Let's use np.random to inherit the seed.
        
        # Stratified Initialization (Sobol-like or Halton)
        # For 4 dimensions, we can use a simple Halton sequence or just stratified random.
        # Let's use a simple stratified sampling: divide each dimension into k bins, pick one point per bin.
        # With budget 200, let's start with a population of size 50.
        
        pop_size = 50
        
        # Generate initial population using stratified sampling
        # We will generate pop_size points.
        # A simple way: use Latin Hypercube Sampling (LHS)
        def lhs_sample(n, dim, lo, hi):
            # n points in dim dimensions
            # For each dimension, divide [0,1] into n intervals, pick one point in each
            samples = np.zeros((n, dim))
            for d in range(dim):
                # Random permutation of intervals
                perm = np.random.permutation(n)
                # Uniform point within each interval
                u = np.random.rand(n)
                # Map to interval
                samples[:, d] = (perm + u) / n
            # Scale to bounds
            return lo + (hi - lo) * samples

        X = lhs_sample(pop_size, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume calculation
        # The scoring uses [3000.0, 0.06]. We should use this or a slightly worse point for internal HV.
        # Using the exact reference point is good.
        ref_point = np.array([3000.0, 0.06])
        
        # Filter out points that are not strictly better than ref point?
        # The scoring only counts points strictly better. 
        # For internal HV, we can just calculate HV of the set.
        
        def calculate_hypervolume_2d(points, ref):
            """
            Calculate hypervolume for 2D points.
            points: (N, 2) array
            ref: (2,) array
            Returns scalar HV.
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points that are dominated by ref (i.e., worse in at least one objective)
            # A point is valid if p[0] < ref[0] and p[1] < ref[1]
            mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
            pts = points[mask]
            
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(pts)):
                x1, y1 = pts[i]
                # The width of this "slice" is x1 - prev_x? No, we integrate.
                # Actually, for 2D HV with ref (rx, ry):
                # Sort by x ascending.
                # For each point i, the contribution is (x_i - x_{i-1}) * (ry - y_i) ? No.
                # Let's use the standard sweep line.
                # Sort by x.
                # Maintain the minimum y seen so far? No, we want the area under the curve formed by the pareto front.
                
                # Better approach: 
                # 1. Sort by x ascending.
                # 2. Iterate. Keep track of the current "maximal" y? No, minimal y is best.
                # Let's just compute the union of rectangles.
                # Since N is small (<=200), O(N^2) is fine.
                
                # Simple O(N^2) method:
                # For each point p, if it is not dominated by any other point in the set,
                # it contributes to the HV.
                # Contribution of a non-dominated point p in 2D:
                # Width = (next_x - p_x) where next_x is the x of the next point in sorted order, or ref_x if it's the last.
                # Height = (ref_y - p_y)
                
                # Let's re-sort pts by x
                # We already sorted.
                # We need to ensure we only consider the Pareto front.
                # A point is on the Pareto front if no other point has x <= p_x and y <= p_y.
                # Since sorted by x, we can just track the min_y so far.
                # If p_y < min_y_so_far, it's on the front.
                
                min_y = ref[1]
                prev_x = pts[0, 0] # This is not quite right for the first point's width
                
                # Let's do it properly:
                # Sort by x.
                # Traverse from left to right.
                # If current y < min_y_so_far:
                #    This point extends the front.
                #    The area contributed by this "step" is (next_x - current_x) * (min_y_so_far - current_y)?
                #    No.
                
                # Let's use the standard algorithm:
                # 1. Sort points by x ascending.
                # 2. Initialize min_y = ref[1].
                # 3. For each point (x, y) in sorted order:
                #    If y < min_y:
                #       The segment from prev_x to x at height min_y is part of the HV?
                #       Actually, the HV is the area of the union of rectangles [x_i, ref_x] x [y_i, ref_y].
                #       We can compute it by summing the area of the "staircase".
                
                # Let's just use a simple O(N^2) dominance check to get the Pareto front, then integrate.
                pareto_front = []
                for i in range(len(pts)):
                    dominated = False
                    for j in range(len(pts)):
                        if i == j: continue
                        if pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1] and (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1]):
                            dominated = True
                            break
                    if not dominated:
                        pareto_front.append(pts[i])
                
                if not pareto_front:
                    return 0.0
                
                # Sort pareto front by x
                pf = np.array(pareto_front)
                pf = pf[np.argsort(pf[:, 0])]
                
                hv = 0.0
                prev_x = pf[0, 0]
                prev_y = ref[1]
                
                # The HV is the sum of rectangles.
                # For each point in the front (sorted by x):
                # The "width" of the contribution of this point's y-level is determined by the next point's x.
                # Let's iterate from right to left? Or left to right?
                # Left to right:
                # Point 0: x0, y0. It dominates the region [x0, x1] x [y0, ref_y]? No.
                # The region [x0, x1] is covered by y0 if y0 is the minimum y in that range.
                # Since it's a Pareto front, x increases, y decreases.
                # So for the interval [x_i, x_{i+1}], the best y is y_i.
                # The area is (x_{i+1} - x_i) * (ref_y - y_i).
                # For the last point, the interval is [x_last, ref_x].
                
                for i in range(len(pf)):
                    x_i = pf[i, 0]
                    y_i = pf[i, 1]
                    if i < len(pf) - 1:
                        x_next = pf[i+1, 0]
                    else:
                        x_next = ref[0]
                    
                    width = x_next - x_i
                    height = ref[1] - y_i
                    hv += width * height
                    
                return hv

        def marginal_hypervolume_contribution(points, ref, point_idx):
            """
            Calculate the marginal HV contribution of point_idx in the set.
            HV(all) - HV(all \ point_idx)
            """
            if len(points) == 0:
                return 0.0
            
            hv_all = calculate_hypervolume_2d(points, ref)
            
            # Remove point
            mask = np.ones(len(points), dtype=bool)
            mask[point_idx] = False
            pts_reduced = points[mask]
            
            hv_reduced = calculate_hypervolume_2d(pts_reduced, ref)
            
            return hv_all - hv_reduced

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Calculate marginal HV contributions for all current points
            # To save time, we can approximate or use a faster method.
            # With pop_size 50, O(N^2) HV calc is 2500 ops. Marginal is 2x that.
            # 50 * 2500 = 125,000 ops per generation. Very fast.
            
            contribs = np.zeros(len(F))
            for i in range(len(F)):
                contribs[i] = marginal_hypervolume_contribution(F, ref_point, i)
            
            # Normalize contributions for selection
            # If all are 0, pick random
            if np.sum(contribs) > 1e-12:
                probs = contribs / np.sum(contribs)
            else:
                probs = np.ones(len(F)) / len(F)
            
            # Select parent based on marginal HV contribution
            parent_idx = np.random.choice(len(F), p=probs)
            parent = X[parent_idx]
            
            # Generate child via mutation
            # Adaptive sigma: start wide, decay over time
            # t is fraction of budget used
            t = used / self.budget
            sigma = 0.5 * (1 - t) + 0.05 # Decay from 0.5 to 0.05
            
            child = parent.copy()
            for j in range(self.dim):
                if np.random.random() < 0.9:
                    # Gaussian mutation
                    delta = np.random.normal(0, sigma)
                    child[j] += delta
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            used += 1
            
            # Add to population
            X = np.vstack([X, child])
            F = np.vstack([F, [f_child]])
            
            # Environmental selection: keep best pop_size points based on HV
            # We want to keep the set of points that maximizes HV.
            # This is the HypE selection.
            # Greedy selection: 
            # 1. Start with empty set.
            # 2. At each step, add the point that gives the largest marginal HV increase to the current set.
            # 3. Repeat until pop_size points are selected.
            
            if len(F) > pop_size:
                # Select best pop_size points
                # We have X, F of size > pop_size
                # We need to select a subset of size pop_size that maximizes HV.
                
                # Greedy HypE selection
                selected_indices = []
                remaining_indices = list(range(len(F)))
                
                for _ in range(pop_size):
                    if not remaining_indices:
                        break
                    
                    # Find point in remaining that maximizes marginal HV when added to selected
                    best_idx = -1
                    best_gain = -1
                    
                    # Current HV of selected set
                    if selected_indices:
                        F_sel = F[selected_indices]
                        hv_sel = calculate_hypervolume_2d(F_sel, ref_point)
                    else:
                        hv_sel = 0.0
                    
                    for idx in remaining_indices:
                        # Try adding idx
                        # New set is selected + [idx]
                        # New HV is HV(selected U {idx})
                        # Gain = New HV - hv_sel
                        
                        # To avoid recalculating full HV for each candidate (O(N^2) * N candidates),
                        # we can just calculate the marginal contribution of idx to the *current* selected set?
                        # No, that's not quite right. The marginal contribution depends on the set.
                        # However, for small N, we can just calculate.
                        
                        # Optimization: 
                        # If we just want the point that contributes most to the *current* front,
                        # we can approximate. But let's be accurate.
                        
                        # Calculate HV of selected + [idx]
                        # Since selected is small (<=50), and we are adding 1, we can just compute.
                        
                        # To speed up, we can pre-compute dominance?
                        # Let's just do the O(N^2) HV calc.
                        # 50 candidates * 50 HV calcs * 50^2 ops = 625,000 ops.
                        # 50 steps in selection = 31,250,000 ops.
                        # This might be slow in Python if done naively.
                        
                        # Let's optimize:
                        # We can just pick the point with the highest "potential" based on the previous marginal HV?
                        # Or use a simpler heuristic: pick the point that is not dominated by the current selected set?
                        
                        # Let's try a faster greedy:
                        # 1. Compute HV of current selected set.
                        # 2. For each remaining point, compute its marginal contribution to the *current* selected set.
                        #    This is HV(selected U {p}) - HV(selected).
                        #    This is exactly the marginal contribution.
                        #    We can compute this faster?
                        #    HV(selected U {p}) is the HV of the set.
                        
                        # Let's stick to the accurate method but optimize the HV calc.
                        # My HV calc is O(N^2).
                        # For N=50, it's fast.
                        # 50 candidates * (50^2) = 125,000 ops per selection step.
                        # 50 steps = 6,250,000 ops.
                        # This is acceptable in Python if vectorized or simple loops.
                        
                        # Let's implement a faster HV calc for 2D.
                        # Sort by x, sweep line. O(N log N).
                        
                        def hv_2d_fast(pts, ref):
                            if len(pts) == 0:
                                return 0.0
                            mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
                            pts = pts[mask]
                            if len(pts) == 0:
                                return 0.0
                            pts = pts[np.argsort(pts[:, 0])]
                            
                            # Remove dominated points in O(N)
                            # Since sorted by x, we keep track of min_y.
                            # If y < min_y, it's on the front.
                            front_x = []
                            front_y = []
                            min_y = ref[1]
                            for i in range(len(pts)):
                                if pts[i, 1] < min_y:
                                    front_x.append(pts[i, 0])
                                    front_y.append(pts[i, 1])
                                    min_y = pts[i, 1]
                            
                            if not front_x:
                                return 0.0
                            
                            hv = 0.0
                            for i in range(len(front_x)):
                                x_i = front_x[i]
                                y_i = front_y[i]
                                if i < len(front_x) - 1:
                                    x_next = front_x[i+1]
                                else:
                                    x_next = ref[0]
                                hv += (x_next - x_i) * (ref[1] - y_i)
                            return hv

                        # Use fast HV in the selection loop
                        best_gain = -1
                        best_candidate = -1
                        for idx in remaining_indices:
                            # Calculate gain
                            # HV(selected U {idx}) - hv_sel
                            # We can just calculate HV of the combined set
                            # To avoid vstack overhead, we can just pass a list?
                            # Or just use the fast HV on the combined array.
                            
                            # For efficiency, let's just pick the point with the highest marginal HV contribution
                            # calculated against the *current* selected set.
                            
                            # Actually, there's a simpler heuristic:
                            # Just pick the point that is non-dominated by the selected set and has the best "spread"?
                            # Let's stick to the greedy HV maximization.
                            
                            # Combine selected and candidate
                            # Since we are doing this 50*50 times, let's be careful.
                            
                            # Let's just use the marginal contribution function but with the fast HV.
                            # We need to modify marginal_hypervolume_contribution to use hv_2d_fast.
                            
                            # Let's refactor:
                            # We will compute the HV of the new set directly.
                            
                            # Optimization: 
                            # If we just want to select the best point, we can compute the marginal contribution
                            # of each remaining point to the *current* selected set.
                            
                            # Let's do that.
                            
                            # We need a function: marginal_contribution(point, set, ref)
                            # This is HV(set U {point}) - HV(set)
                            
                            # We already have hv_sel.
                            # We just need HV(set U {point}).
                            
                            # Let's create a helper inside the loop.
                            
                            pass # Placeholder

                        # This inner loop is getting complex. Let's simplify.
                        # Instead of full greedy HypE selection which is O(K * N * HV_cost),
                        # we can just keep the top K points based on their *current* marginal HV contribution
                        # to the *entire* population? No, that's what we did for selection.
                        
                        # Alternative: 
                        # Just keep the non-dominated points? 
                        # If the number of non-dominated points is > pop_size, then use crowding distance?
                        # That's NSGA-II.
                        
                        # Let's go back to the simple HypE idea:
                        # 1. Calculate marginal HV contribution of each point in the *current* population.
                        # 2. Keep the top pop_size points.
                        # This is O(N * HV_cost).
                        # HV_cost is O(N log N) with fast HV.
                        # So O(N^2 log N).
                        # For N=100 (max pop size during selection), 100 * 100 * log(100) ~ 70,000 ops.
                        # This is very fast.
                        
                        # Let's implement this simpler version.
                        
                        break # Break out of the complex selection loop and use the simpler one below
                
                # Simpler Selection:
                # Calculate marginal HV contribution of each point in the current population (size M > pop_size)
                # Keep the top pop_size.
                
                contribs = np.zeros(len(F))
                for i in range(len(F)):
                    # Marginal contribution of i to the rest
                    # HV(all) - HV(all \ i)
                    # We can compute HV(all) once.
                    hv_all = hv_2d_fast(F, ref_point)
                    
                    # Compute HV(all \ i)
                    mask = np.ones(len(F), dtype=bool)
                    mask[i] = False
                    F_reduced = F[mask]
                    hv_reduced = hv_2d_fast(F_reduced, ref_point)
                    
                    contribs[i] = hv_all - hv_reduced
                
                # Select top pop_size
                indices = np.argsort(-contribs)[:pop_size]
                X = X[indices]
                F = F[indices]

        return X

# EVOLVE-BLOCK-END
