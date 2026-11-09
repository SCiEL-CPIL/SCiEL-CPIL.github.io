import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        def eval_point(x):
            return np.array(problem(x))
        
        # Grid-based decomposition strategy:
        # 1. Use structured grid sampling to cover the search space systematically
        # 2. Identify promising regions based on objective values
        # 3. Refine with adaptive local search in those regions
        
        # Phase 1: Structured grid sampling
        # Use a hypercube grid with varying resolution per dimension
        # For 12 dims, we can't do full factorial, so use stratified sampling
        
        # Allocate 40% of budget to initial structured exploration
        n_explore = max(40, budget // 5)
        
        # Generate stratified samples: divide each dimension into k bins,
        # sample one point from each bin, then combine across dimensions
        # Use a smaller number of bins to keep it tractable
        n_bins = 4  # 4 bins per dimension
        
        # Generate all combinations of bin indices (4^12 is too many, so use random selection)
        # Instead, use a space-filling approach: sample n_explore points with stratification
        
        X_explore = np.zeros((n_explore, dim))
        for i in range(dim):
            # Divide dimension i into n_bins equal bins
            bin_width = (hi[i] - lo[i]) / n_bins
            # For each point, assign it to a bin, ensuring coverage
            # Use a cyclic pattern with randomization
            bin_assignments = np.random.randint(0, n_bins, n_explore)
            # Ensure each bin gets roughly equal representation
            # Shuffle to mix
            np.random.shuffle(bin_assignments)
            # Assign points to bins
            X_explore[:, i] = lo[i] + (bin_assignments + np.random.rand(n_explore)) * bin_width
            X_explore[:, i] = np.clip(X_explore[:, i], lo[i], hi[i])
        
        # Evaluate exploration points
        F_explore = np.array([eval_point(x) for x in X_explore])
        
        # Check time
        if time.time() > deadline - 10.0:
            return X_explore
        
        # Phase 2: Identify promising regions
        # Find non-dominated points and cluster them
        fronts = self._fast_non_dominated_sort(F_explore)
        
        if not fronts:
            return X_explore
        
        # Get the best front
        best_front_idx = list(fronts[0])
        
        # If we have multiple fronts, also consider the second front for diversity
        all_promise_idx = list(fronts[0])
        if len(fronts) > 1:
            # Add some from second front for diversity
            second_front = list(fronts[1])
            n_second = min(5, len(second_front))
            if n_second > 0:
                all_promise_idx.extend(np.random.choice(second_front, n_second, replace=False).tolist())
        
        # Phase 3: Adaptive local refinement
        remaining_budget = budget - n_explore
        
        if remaining_budget <= 0:
            return X_explore
        
        # Select up to 8 centers for refinement from promising points
        max_centers = min(8, len(all_promise_idx))
        if max_centers == 0:
            return X_explore
        
        # Select centers with diversity (spread across the Pareto front)
        selected_centers = self._select_diverse_centers(
            F_explore[all_promise_idx], 
            X_explore[all_promise_idx],
            max_centers,
            lo, hi
        )
        
        # Distribute remaining budget among centers
        evals_per_center = remaining_budget // max_centers
        if evals_per_center < 3:
            evals_per_center = max(3, remaining_budget // max(1, max_centers))
        
        # Adjust to stay within budget
        total_refine = evals_per_center * max_centers
        if total_refine > remaining_budget:
            evals_per_center = remaining_budget // max_centers
            if evals_per_center < 1:
                evals_per_center = 1
            total_refine = evals_per_center * max_centers
        
        # Perform adaptive local search around each center
        X_refine = []
        F_refine = []
        
        for center in selected_centers:
            # Start with a moderate step size and adapt
            step_size = 0.1 * (hi - lo)  # 10% of range initially
            
            for eval_i in range(evals_per_center):
                if time.time() > deadline - 2.0:
                    break
                
                # Generate candidate points with decreasing step sizes
                # Use a mix of directions for better coverage
                n_candidates = 3
                best_x = None
                best_f = None
                
                for _ in range(n_candidates):
                    # Random direction with varying magnitude
                    direction = np.random.randn(dim)
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    
                    # Vary the step size: sometimes large, sometimes small
                    scale = np.random.choice([0.5, 1.0, 2.0])
                    x_candidate = center + direction * step_size * scale
                    x_candidate = np.clip(x_candidate, lo, hi)
                    
                    f_candidate = eval_point(x_candidate)
                    
                    if best_f is None or self._is_better(f_candidate, best_f, F_explore):
                        best_x = x_candidate
                        best_f = f_candidate
                
                if best_x is not None:
                    X_refine.append(best_x)
                    F_refine.append(best_f)
                    
                    # Adapt step size: if we found improvement, keep going; else, shrink
                    # Simple adaptation: shrink step size over iterations
                    step_size *= 0.8
        
        if X_refine:
            X_refine = np.array(X_refine)
            F_refine = np.array(F_refine)
        else:
            X_refine = np.empty((0, dim))
            F_refine = np.empty((0, n_obj))
        
        # Phase 4: Final global refinement
        # Use any remaining budget for a few random global samples to catch missed regions
        used_budget = n_explore + len(X_refine)
        remaining_final = budget - used_budget
        
        if remaining_final > 0 and time.time() < deadline - 5.0:
            X_final = []
            F_final = []
            for _ in range(min(remaining_final, 20)):
                if time.time() > deadline - 1.0:
                    break
                x_rand = lo + np.random.rand(dim) * (hi - lo)
                f_rand = eval_point(x_rand)
                X_final.append(x_rand)
                F_final.append(f_rand)
            
            if X_final:
                X_final = np.array(X_final)
                F_final = np.array(F_final)
            else:
                X_final = np.empty((0, dim))
                F_final = np.empty((0, n_obj))
        else:
            X_final = np.empty((0, dim))
            F_final = np.empty((0, n_obj))
        
        # Combine all results
        X_all = X_explore
        F_all = F_explore
        
        if len(X_refine) > 0:
            X_all = np.vstack([X_all, X_refine])
            F_all = np.vstack([F_all, F_refine])
        
        if len(X_final) > 0:
            X_all = np.vstack([X_all, X_final])
            F_all = np.vstack([F_all, F_final])
        
        return X_all

    def _is_better(self, f_new, f_ref, all_F):
        """Check if f_new is non-dominated and improves coverage"""
        # Check if f_new dominates any existing point
        for i in range(len(all_F)):
            if np.all(f_new <= all_F[i]) and np.any(f_new < all_F[i]):
                return True
        # Check if f_new is non-dominated by all existing points
        for i in range(len(all_F)):
            if np.all(all_F[i] <= f_new) and np.any(all_F[i] < f_new):
                return False
        # Non-dominated: accept for diversity
        return True

    def _select_diverse_centers(self, F, X, n_select, lo, hi):
        """Select diverse centers from candidate points"""
        n = len(F)
        if n <= n_select:
            return X
        
        # Use a greedy selection based on distance in objective space
        selected = []
        remaining = list(range(n))
        
        # Start with the point that is most "central" in objective space
        # Normalize objectives to [0,1] for distance calculation
        F_min = np.min(F, axis=0)
        F_max = np.max(F, axis=0)
        F_range = F_max - F_min + 1e-10
        F_norm = (F - F_min) / F_range
        
        # Select first center: closest to center of normalized space
        center_target = np.ones(n_select) * 0.5
        dists = np.linalg.norm(F_norm - center_target, axis=1)
        first_idx = np.argmin(dists)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Greedily select next points that maximize minimum distance to already selected
        while len(selected) < n_select and remaining:
            # For each remaining point, compute min distance to selected points
            min_dists = []
            for idx in remaining:
                min_d = min(np.linalg.norm(F_norm[idx] - F_norm[sel]) for sel in selected)
                min_dists.append(min_d)
            
            # Select the one with maximum minimum distance
            best_idx = remaining[np.argmax(min_dists)]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return X[selected]

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
            
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                # i dominates j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                    
        front = set()
        for i in range(n):
            if n_dom[i] == 0:
                front.add(i)
        if front:
            fronts.append(front)
                
        k = 0
        while k < len(fronts):
            next_front = set()
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.add(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts
