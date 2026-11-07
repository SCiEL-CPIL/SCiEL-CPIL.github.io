"""Multi-objective optimizer for ZDT1 using hypervolume-driven adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # ZDT1 specific: x1 controls f1, x2..xn control f2
        # f1 = x1
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/n))
        # For 2 objectives, the Pareto front is defined by varying x1 in [0,1]
        
        # Strategy: 
        # 1. Initialize with a spread of x1 values (and random x2..xn)
        # 2. Use adaptive refinement: focus evaluations on regions of the Pareto front 
        #    that maximize marginal hypervolume gain.
        
        # Initial population: uniform spread in x1
        n_init = min(20, self.budget // 4)
        x1_values = np.linspace(0.0, 1.0, n_init)
        rng = np.random.default_rng(42)
        
        X = np.zeros((n_init, dim))
        X[:, 0] = x1_values
        X[:, 1:] = rng.random((n_init, dim - 1))
        
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Keep track of non-dominated points for hypervolume calculation
        # Since ZDT1 has a known structure, we can be smarter.
        # f1 = x1, f2 = g(x2..xn) * (1 - sqrt(x1/dim))
        # To minimize f2 for a given x1, we need g(x2..xn) -> 0
        # g(x) = 1 + 10*(n-1)*sum(x_i^2) for i=2..n
        # So we want x2..xn to be close to 0.
        
        # However, we don't know the problem function exactly, so we must use evaluations.
        # But we can exploit the structure: for each x1, the best f2 is achieved when x2..xn are minimal.
        # So a good strategy is to fix x2..xn to 0 (or very small) and vary x1.
        # But the problem might have noise or different bounds? The contract says bounds [0,1].
        # Let's try a hybrid: 
        # 1. Explore with random x2..xn to find good g values.
        # 2. Exploit by fixing x2..xn to their best found values and varying x1.
        
        # Actually, since the task is ZDT1, and we know the structure, 
        # the most efficient way to get high hypervolume is to find points with small g.
        # Let's use a simple approach: 
        # - Maintain a set of candidate x1 values.
        # - For each x1, try to minimize f2 by adjusting x2..xn.
        # - Use the evaluation budget to refine x1 distribution based on hypervolume contribution.
        
        # Simpler robust approach: 
        # 1. Initial spread.
        # 2. For each iteration, select a parent from non-dominated set.
        # 3. Mutate x1 primarily (since it controls f1 and affects f2), and slightly adjust x2..xn.
        # 4. Use hypervolume indicator to decide which point to replace.
        
        # Let's implement a lightweight hypervolume-guided search.
        
        # Helper to calculate hypervolume for 2D
        def hypervolume_2d(points, ref):
            # points: (n, 2), ref: (2,)
            # Only count points strictly better than ref
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by f1
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            for i in range(len(pts)):
                f1, f2 = pts[i]
                # Width in f1
                w1 = ref[0] - f1
                # Height in f2: from f2[i] to min(ref[1], f2[i-1] if i>0 else ref[1])
                # Actually, standard way: sort by f1 ascending.
                # The contribution of point i is (ref[0] - f1[i]) * (min(ref[1], f2[i-1]) - f2[i]) if f2[i-1] < ref[1]
                # But we need to be careful with non-dominated points.
                # Let's use a simpler incremental approach or just compute the union of rectangles.
                pass
            
            # More robust 2D HV calculation
            # Sort by f1 ascending
            pts = pts[np.argsort(pts[:, 0])]
            # Filter out dominated points within the set? 
            # Actually, for HV, we only need the non-dominated front.
            # Let's filter non-dominated first.
            nd_mask = np.ones(len(pts), dtype=bool)
            for i in range(len(pts)):
                if not nd_mask[i]:
                    continue
                for j in range(len(pts)):
                    if i == j or not nd_mask[j]:
                        continue
                    if pts[i][0] <= pts[j][0] and pts[i][1] <= pts[j][1] and (pts[i][0] < pts[j][0] or pts[i][1] < pts[j][1]):
                        nd_mask[j] = False
            
            pts = pts[nd_mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by f1 ascending
            pts = pts[np.argsort(pts[:, 0])]
            
            hv = 0.0
            prev_f2 = ref[1]
            for i in range(len(pts)):
                f1, f2 = pts[i]
                if f2 >= ref[1]:
                    continue
                # The rectangle for this point extends from f1 to next_f1 (or ref[0]) in f1 dimension
                # and from f2 to prev_f2 in f2 dimension
                # Wait, this is not quite right for non-sorted f2.
                
                # Correct 2D HV:
                # Sort points by f1 ascending.
                # Keep track of the minimum f2 seen so far (from the right, i.e., from higher f1 to lower f1)
                # Actually, sort by f1 ascending. 
                # For each point, the contribution is (next_f1 - f1) * (min_f2_so_far - f2) where min_f2_so_far is the min f2 of points with higher f1.
                
                # Let's sort by f1 descending to make it easier?
                # Or just use the standard algorithm:
                # Sort by f1 ascending.
                # Traverse from right to left (highest f1 to lowest).
                # Keep min_f2 = ref[1]
                # For each point (f1, f2):
                #   if f2 < min_f2:
                #     hv += (f1 - next_f1) * (min_f2 - f2)  -- Wait, next_f1 is the f1 of the point to the right.
                #     min_f2 = f2
                #   next_f1 = f1
                # At the end, add the last segment to ref[0].
                
                pts_desc = pts[np.argsort(pts[:, 0])[::-1]]
                hv = 0.0
                min_f2 = ref[1]
                next_f1 = ref[0]
                for f1, f2 in pts_desc:
                    if f2 < min_f2:
                        hv += (next_f1 - f1) * (min_f2 - f2)
                        min_f2 = f2
                    next_f1 = f1
                return hv

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select parent from non-dominated set
            # Calculate non-dominated mask
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not nd_mask[j]:
                        continue
                    if F[i][0] <= F[j][0] and F[i][1] <= F[j][1] and (F[i][0] < F[j][0] or F[i][1] < F[j][1]):
                        nd_mask[j] = False
            
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                parent_idx = np.random.randint(len(F))
            else:
                # Prefer parents with better hypervolume contribution? 
                # For simplicity, pick randomly from non-dominated, or the one with best f2.
                # Let's pick the one with the smallest f2 among non-dominated to encourage exploitation.
                parent_idx = nd_indices[np.argmin(F[nd_indices, 1])]
            
            # Generate child
            # Mutate x1 significantly to explore the front, x2..xn slightly to improve g.
            child = X[parent_idx].copy()
            
            # x1 mutation: large step to cover the front
            child[0] += rng.uniform(-0.1, 0.1)
            child[0] = np.clip(child[0], 0.0, 1.0)
            
            # x2..xn mutation: small steps towards 0 (since we want g small)
            # But we don't know if 0 is optimal without evaluating.
            # Let's use a random mutation but with a bias towards lower values.
            child[1:] += rng.normal(0, 0.05, dim - 1)
            child[1:] = np.clip(child[1:], 0.0, 1.0)
            
            fc = problem(child)
            used += 1
            
            # Replace the worst point in terms of hypervolume contribution
            # For simplicity, replace the point with the largest f2 among dominated points,
            # or the one with the largest sum of objectives.
            # Let's find the point that contributes least to hypervolume.
            # This is expensive. Approximate by finding the point that is dominated by the most others.
            
            # Simpler: replace the point with the worst f2 if it's dominated, or worst sum.
            # Let's use the sum of objectives as a proxy.
            worst_idx = np.argmax(np.sum(F, axis=1))
            
            # But this might discard good spread. 
            # Let's use a simple crowding distance or just replace the worst non-dominated point if the child is non-dominated.
            
            # Check if child is non-dominated with respect to current population
            child_dominated = False
            for i in range(len(F)):
                if F[i][0] <= fc[0] and F[i][1] <= fc[1] and (F[i][0] < fc[0] or F[i][1] < fc[1]):
                    child_dominated = True
                    break
            
            if not child_dominated:
                # Child is non-dominated. Replace the point with the largest distance to the ideal point (0,0)
                # Or replace the point that is dominated by the child.
                dominated_by_child = np.where((F[:, 0] >= fc[0]) & (F[:, 1] >= fc[1]) & ((F[:, 0] > fc[0]) | (F[:, 1] > fc[1])))[0]
                if len(dominated_by_child) > 0:
                    # Replace the one with the worst f2 among those dominated
                    worst_in_dominated = dominated_by_child[np.argmax(F[dominated_by_child, 1])]
                    X[worst_in_dominated] = child
                    F[worst_in_dominated] = fc
                else:
                    # Replace the point with the largest f2 (worst in second objective)
                    worst_idx = np.argmax(F[:, 1])
                    X[worst_idx] = child
                    F[worst_idx] = fc
            else:
                # Child is dominated. Only replace if it is better than the worst point in some way?
                # For now, skip replacement to maintain diversity, or replace if it improves the worst point.
                if fc[1] < F[worst_idx, 1]:
                    X[worst_idx] = child
                    F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
