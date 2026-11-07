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
        # Use a larger population for better coverage in early generations
        pop_size = 30
        
        # Initialize population with random points
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
        def calculate_hv_contributions(F_nd, ref):
            if len(F_nd) == 0:
                return np.array([])
            
            idx = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[idx]
            
            contributions = np.zeros(len(F_sorted))
            
            f1_next = np.empty(len(F_sorted))
            f1_next[:-1] = F_sorted[1:, 0]
            f1_next[-1] = ref[0]
            
            widths = f1_next - F_sorted[:, 0]
            heights = ref[1] - F_sorted[:, 1]
            
            contributions = widths * heights
            
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
        while used < budget and time.time() < deadline:
            # Get current non-dominated set from population
            nd_idx = get_non_dominated_indices(pop_F)
            
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(pop_F))
            
            # Calculate contributions for selection
            F_nd = pop_F[nd_idx]
            contribs = calculate_hv_contributions(F_nd, ref_point)
            
            # Normalize contributions for probability
            total_contrib = np.sum(contribs)
            if total_contrib < 1e-10:
                probs = np.ones(len(nd_idx)) / len(nd_idx)
            else:
                probs = contribs / total_contrib
            
            # Adaptive mutation factor: start high for exploration, decrease for exploitation
            progress = 1.0 - (budget - used) / max(1, (budget - init_count))
            F_mut = 0.9 - 0.7 * progress  # Range [0.2, 0.9]
            
            # Generate offspring
            new_X = []
            new_F = []
            
            for i in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random parents from non-dominated set
                if len(nd_idx) >= 3:
                    r_idx = np.random.choice(len(nd_idx), size=3, replace=False)
                    r1, r2, r3 = nd_idx[r_idx[0]], nd_idx[r_idx[1]], nd_idx[r_idx[2]]
                else:
                    r_idx = np.random.choice(len(pop_F), size=3, replace=True)
                    r1, r2, r3 = r_idx[0], r_idx[1], r_idx[2]
                
                v = pop_X[r1] + F_mut * (pop_X[r2] - pop_X[r3])
                v = np.clip(v, lo, hi)
                
                f_v = eval_point(v)
                if f_v is None:
                    continue
                
                new_X.append(v)
                new_F.append(f_v)
            
            if len(new_X) == 0:
                break
            
            new_X = np.array(new_X)
            new_F = np.array(new_F)
            
            # Combine population and offspring
            combined_X = np.vstack([pop_X, new_X])
            combined_F = np.vstack([pop_F, new_F])
            
            # Non-dominated sort
            nd_idx_all = get_non_dominated_indices(combined_F)
            
            if len(nd_idx_all) > pop_size:
                # Select top pop_size from non-dominated based on crowding distance
                F_nd_all = combined_F[nd_idx_all]
                
                cd = np.zeros(len(F_nd_all))
                
                for j in range(n_obj):
                    order = np.argsort(F_nd_all[:, j])
                    F_sorted_j = F_nd_all[order, j]
                    dists = np.zeros(len(F_nd_all))
                    dists[0] = np.inf
                    dists[-1] = np.inf
                    for k in range(1, len(F_nd_all) - 1):
                        dists[k] = F_sorted_j[k+1] - F_sorted_j[k-1]
                    cd[order] += dists
                
                top_idx_in_nd = np.argsort(cd)[::-1][:pop_size]
                final_idx = nd_idx_all[top_idx_in_nd]
                
            elif len(nd_idx_all) == pop_size:
                final_idx = nd_idx_all
            else:
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
