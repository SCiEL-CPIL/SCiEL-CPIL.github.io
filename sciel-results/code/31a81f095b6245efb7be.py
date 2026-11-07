"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Differential Evolution with Hypervolume Contribution Selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # Reference point for hypervolume calculation
        # Based on task description: [3000.0, 0.06]
        # But we should estimate a safe reference point if the actual bounds are tighter or if objectives are larger.
        # The task says "reference point [3000.0, 0.06]". We will use this.
        ref_point = np.array([3000.0, 0.06])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = np.array(problem(x), dtype=np.float64)
            except Exception:
                return None
            if not np.all(np.isfinite(f)):
                f = np.array([1e10, 1e10])
            all_X.append(x)
            all_F.append(f)
            used += 1
            return f

        # 1. Initialization
        # Use a population size that allows for several generations.
        # Budget 200. Let's use population size 20.
        # 200 / 20 = 10 generations.
        pop_size = 20
        
        # Initialize population with random points
        # Ensure we don't exceed budget
        init_count = min(pop_size, budget)
        pop_X = []
        pop_F = []
        
        for _ in range(init_count):
            x = lo + span * np.random.random(dim)
            f = eval_point(x)
            if f is None:
                break
            pop_X.append(x)
            pop_F.append(f)

        if len(pop_X) == 0:
            return all_X

        pop_X = np.array(pop_X)
        pop_F = np.array(pop_F)

        # Helper: Calculate Hypervolume Contribution of a point given a set of non-dominated points
        # For 2D, the contribution of a point p to the hypervolume of set S (with reference r)
        # is the area of the box defined by p and r that is not covered by any other point in S.
        # Specifically, for 2D, if we sort non-dominated points by f1 ascending, 
        # the contribution of point i is (f_{i+1,1} - f_{i,1}) * (r_2 - f_{i,2}) 
        # with f_{n,1} = r_1.
        
        def calculate_hv_contributions(F_nd, ref):
            """
            F_nd: (N, 2) array of non-dominated points.
            ref: (2,) array of reference point.
            Returns: (N,) array of hypervolume contributions.
            """
            if len(F_nd) == 0:
                return np.array([])
            
            # Sort by first objective ascending
            idx = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[idx]
            
            contributions = np.zeros(len(F_sorted))
            
            # The "next" first objective value for the last point is the reference
            f1_next = np.empty(len(F_sorted))
            f1_next[:-1] = F_sorted[1:, 0]
            f1_next[-1] = ref[0]
            
            # Width in f1 direction
            widths = f1_next - F_sorted[:, 0]
            # Height in f2 direction
            heights = ref[1] - F_sorted[:, 1]
            
            # Area
            contributions = widths * heights
            
            # Map back to original order
            inv_idx = np.empty(len(idx), dtype=int)
            inv_idx[idx] = np.arange(len(idx))
            contrib_original = np.zeros(len(F_nd))
            contrib_original[inv_idx] = contributions
            
            return contrib_original

        def get_non_dominated_indices(F):
            if len(F) == 0:
                return np.array([], dtype=int)
            n = len(F)
            le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
            lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
            dominated = np.any(le & lt, axis=0)
            return np.where(~dominated)[0]

        # 2. Main DE Loop
        # Adaptive mutation factor F
        # Standard DE uses F in [0.5, 1.0]. We can shrink it over time.
        
        while used < budget and time.time() < deadline:
            # Get current non-dominated set from population
            nd_idx = get_non_dominated_indices(pop_F)
            
            if len(nd_idx) == 0:
                # Fallback if something went wrong
                nd_idx = np.arange(len(pop_F))
            
            # Calculate contributions for selection
            F_nd = pop_F[nd_idx]
            contribs = calculate_hv_contributions(F_nd, ref_point)
            
            # Select parents based on contribution
            # Points with higher contribution are more valuable to the front
            # We want to explore around them.
            
            # Normalize contributions for probability
            total_contrib = np.sum(contribs)
            if total_contrib < 1e-10:
                # All contributions are zero? Use uniform
                probs = np.ones(len(nd_idx)) / len(nd_idx)
            else:
                probs = contribs / total_contrib
            
            # Select two distinct parents for mutation
            # We need at least 3 points for DE (v = x_r1 + F*(x_r2 - x_r3))
            # If we have fewer, we just pick random distinct indices.
            
            n_gen = (budget - used) // pop_size
            # Shrink mutation factor
            # Start with 0.8, end with 0.2
            progress = 1.0 - (budget - used) / max(1, (budget - init_count))
            F_mut = 0.8 - 0.6 * progress
            
            # Generate offspring
            new_X = []
            new_F = []
            
            for i in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random parents from non-dominated set
                # If nd_idx has < 3, sample with replacement or from full pop
                if len(nd_idx) >= 3:
                    # Sample 3 distinct indices from nd_idx
                    r_idx = np.random.choice(len(nd_idx), size=3, replace=False)
                    r1, r2, r3 = nd_idx[r_idx[0]], nd_idx[r_idx[1]], nd_idx[r_idx[2]]
                else:
                    # Sample from full population with replacement
                    r_idx = np.random.choice(len(pop_F), size=3, replace=True)
                    r1, r2, r3 = r_idx[0], r_idx[1], r_idx[2]
                    
                # Ensure r1, r2, r3 are distinct if possible
                # If len(nd_idx) >= 3, they are distinct.
                # If not, we might have duplicates. That's okay for a fallback.
                
                v = pop_X[r1] + F_mut * (pop_X[r2] - pop_X[r3])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Evaluate
                f_v = eval_point(v)
                if f_v is None:
                    continue
                
                new_X.append(v)
                new_F.append(f_v)
            
            if len(new_X) == 0:
                break
            
            new_X = np.array(new_X)
            new_F = np.array(new_F)
            
            # Selection: Tournament or 1:1 replacement?
            # Standard DE: Compare offspring with parent. If offspring dominates parent, replace.
            # If non-dominated, keep both? No, population size must stay constant.
            # Let's do 1:1 replacement based on dominance, but with a twist:
            # If offspring and parent are non-dominated to each other, keep the one with better HV contribution?
            # Or just keep the one that is better in terms of some metric?
            # Simpler: If offspring dominates parent, replace.
            # Else if parent dominates offspring, keep parent.
            # Else (non-dominated), keep the one with better "distance to ideal" or random?
            # To maintain diversity, let's use a simple rule:
            # If non-dominated, replace with probability 0.5? Or keep parent to ensure convergence?
            # For hypervolume, we want diversity. Let's replace if offspring has better HV contribution 
            # in the context of the current front? That's expensive.
            # Let's try: Replace if offspring dominates parent. Otherwise, keep parent.
            # This might lose diversity.
            
            # Alternative: Combine population and offspring, select top pop_size based on 
            # non-dominance + crowding distance. This is NSGA-II style.
            # Given the small budget, let's stick to a simpler 1:1 but slightly biased towards diversity.
            
            # Let's implement a simple tournament:
            # For each i, compare new_X[i] with pop_X[i].
            # If new dominates old: replace.
            # If old dominates new: keep.
            # If non-dominated: 
            #   Calculate HV contribution of new point assuming it's added to the current non-dominated set.
            #   Calculate HV contribution of old point.
            #   Replace if new has higher contribution?
            #   Actually, let's just keep the one that is closer to the reference point in a normalized sense?
            #   Or simpler: Keep the one that is better in the weighted sum with random weights?
            
            # Let's use the "Crowding Distance" logic from NSGA-II for ties, but simplified.
            # Actually, with 20 points, we can just do a full front update.
            
            # Let's combine and select top 20.
            combined_X = np.vstack([pop_X, new_X])
            combined_F = np.vstack([pop_F, new_F])
            
            # Non-dominated sort
            nd_idx_all = get_non_dominated_indices(combined_F)
            
            # If we have more than pop_size non-dominated points, we need to select.
            # If we have fewer, we fill with dominated points closest to the front.
            
            if len(nd_idx_all) > pop_size:
                # Select top pop_size from non-dominated based on crowding distance
                F_nd_all = combined_F[nd_idx_all]
                
                # Calculate crowding distance
                # For 2D, crowding distance is (f1_max - f1_min) + (f2_max - f2_min) normalized?
                # Standard: Sum of distances to neighbors in each objective.
                # For 2D, it's just the area of the box defined by neighbors.
                
                # Sort by f1
                order_f1 = np.argsort(F_nd_all[:, 0])
                F_sorted_f1 = F_nd_all[order_f1]
                
                cd = np.zeros(len(F_nd_all))
                
                # For each objective, calculate distances
                for j in range(n_obj):
                    order = np.argsort(F_nd_all[:, j])
                    F_sorted_j = F_nd_all[order, j]
                    dists = np.zeros(len(F_nd_all))
                    dists[0] = np.inf
                    dists[-1] = np.inf
                    for k in range(1, len(F_nd_all) - 1):
                        dists[k] = F_sorted_j[k+1] - F_sorted_j[k-1]
                    cd[order] += dists
                
                # Select top pop_size by crowding distance
                top_idx_in_nd = np.argsort(cd)[::-1][:pop_size]
                final_idx = nd_idx_all[top_idx_in_nd]
                
            elif len(nd_idx_all) == pop_size:
                final_idx = nd_idx_all
            else:
                # Fill with dominated points
                # Sort dominated points by their "distance" to the non-dominated front?
                # Simple: Just take the first pop_size - len(nd_idx_all) dominated points?
                # Better: Rank all points. Non-dominated are rank 0. Dominated are rank 1, etc.
                # For simplicity, just pick random dominated points to fill.
                dominated_idx = np.setdiff1d(np.arange(len(combined_F)), nd_idx_all)
                n_fill = pop_size - len(nd_idx_all)
                if len(dominated_idx) >= n_fill:
                    fill_idx = np.random.choice(dominated_idx, size=n_fill, replace=False)
                else:
                    fill_idx = dominated_idx
                
                final_idx = np.concatenate([nd_idx_all, fill_idx])
            
            pop_X = combined_X[final_idx]
            pop_F = combined_F[final_idx]
            
        return all_X
# EVOLVE-BLOCK-END
