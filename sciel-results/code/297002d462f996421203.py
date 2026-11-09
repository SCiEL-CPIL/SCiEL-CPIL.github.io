"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for RE37 (ZDT4-like) using hypervolume-aware
    parent selection and adaptive local refinement.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # Reference point for hypervolume calculation
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Storage for all evaluated points
        all_X = []
        all_F = []
        evals_used = 0

        def get_non_dominated_indices(F_list):
            """
            Returns indices of non-dominated points in F_list.
            F_list is a list of numpy arrays.
            """
            n = len(F_list)
            if n == 0:
                return np.array([], dtype=int)
            if n == 1:
                return np.array([0], dtype=int)
            
            F_arr = np.array(F_list)
            # Filter out points that are not strictly better than reference point
            # The prompt says "Only points strictly better than the reference point in every objective count"
            # However, for dominance sorting, we usually consider all points.
            # But for hypervolume, only those inside the box matter.
            
            # Dominance check: i dominates j if F[i] <= F[j] and F[i] != F[j]
            # We want indices that are NOT dominated by any other point in the list.
            
            # Vectorized dominance check
            # F_i: (N, 1, M), F_j: (1, N, M)
            F_i = F_arr[:, np.newaxis, :] 
            F_j = F_arr[np.newaxis, :, :]
            
            # le[i, j] is True if F_j <= F_i (j is better or equal in all objectives)
            le = np.all(F_j <= F_i, axis=2) 
            # lt[i, j] is True if F_j < F_i in at least one objective
            lt = np.any(F_j < F_i, axis=2)
            
            # dominated[i] is True if there exists a j != i such that j dominates i
            # We need to exclude self-comparison
            # le & lt includes self if F_i == F_i? No, lt is strict. 
            # If F_i == F_j, le is True, lt is False. So self does not dominate self.
            # If F_j < F_i, le is True, lt is True.
            
            dominated_mask = np.any(le & lt, axis=1)
            nd_indices = np.where(~dominated_mask)[0]
            return nd_indices

        def calculate_hv_contributions(nd_indices, F_list):
            """
            Calculate hypervolume contribution of each point in the non-dominated set.
            This is an approximation using a simple grid or sorting method for 3D.
            For small N, we can use a more precise method.
            """
            if len(nd_indices) == 0:
                return np.array([])
            
            F_nd = np.array([F_list[i] for i in nd_indices])
            n_nd = len(F_nd)
            
            # Filter points that are actually within the reference box
            valid_mask = np.all(F_nd < ref_point, axis=1)
            if not np.any(valid_mask):
                return np.zeros(n_nd)
            
            F_valid = F_nd[valid_mask]
            idx_valid = np.where(valid_mask)[0]
            
            # For 3D, we can compute HV contribution by sorting and integrating.
            # A simpler heuristic for "importance" is the volume of the box defined by the point and the reference point,
            # minus the volume covered by points that dominate it in the sorted order.
            # Given the small size, let's use a standard 3D HV algorithm.
            
            # Sort by f1
            sort_idx = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[sort_idx]
            
            # This is getting complex. Let's use a simpler metric: 
            # Contribution is proportional to the "empty space" it covers.
            # Approximation: For each point, calculate the volume of the hyper-rectangle 
            # from the point to the ref point, weighted by how many other points are "outside" it.
            
            contribs = np.zeros(n_nd)
            
            # Efficient 3D HV calculation for small sets
            # We can iterate through points and for each, calculate the volume of the box 
            # that is NOT covered by any other point with smaller f1 (assuming sorted).
            
            # Let's stick to a robust, simpler metric if HV is too slow/complex to implement correctly in a few lines.
            # Metric: Isolation distance in objective space.
            # Points that are far from others in the ND set are more valuable for diversity.
            # Points that are "extreme" (min f1, min f2, min f3) are valuable for extent.
            
            # Combine: 
            # 1. Extent score: how close to the min of each objective.
            # 2. Diversity score: distance to nearest neighbor.
            
            min_f = np.min(F_nd, axis=0)
            max_f = np.max(F_nd, axis=0)
            
            # Normalize F_nd to [0, 1] range based on current ND set
            if np.all(max_f == min_f):
                norm_F = np.zeros_like(F_nd)
            else:
                norm_F = (F_nd - min_f) / (max_f - min_f)
            
            # Distance to nearest neighbor
            min_dists = np.zeros(n_nd)
            for k in range(n_nd):
                dists = np.linalg.norm(F_nd - F_nd[k], axis=1)
                dists[k] = np.inf
                min_dists[k] = np.min(dists)
            
            # Normalize distances
            max_dist = np.max(min_dists)
            if max_dist > 0:
                norm_dists = min_dists / max_dist
            else:
                norm_dists = np.zeros(n_nd)
            
            # Score = 0.5 * (1 - norm_F.sum(axis=1)) + 0.5 * norm_dists
            # (1 - norm_F.sum) favors points with lower objective values (closer to ideal)
            # norm_dists favors isolated points
            
            scores = 0.5 * (1 - norm_F.sum(axis=1)) + 0.5 * norm_dists
            
            contribs[idx_valid] = scores[idx_valid]
            return contribs

        # Phase 1: Initial random population
        init_pop = 40
        X_init = lo + (hi - lo) * np.random.random((init_pop, dim))
        for x in X_init:
            all_X.append(x.copy())
            all_F.append(np.array(problem(x), dtype=np.float64))
            evals_used += 1
            if time.time() > deadline:
                return None

        # Phase 2: Evolutionary Refinement
        mutation_scale_start = 0.3
        mutation_scale_end = 0.01
        polish_threshold = int(budget * 0.8) # Start polish in last 20%
        
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated set
            nd_idx = get_non_dominated_indices(all_F)
            
            if len(nd_idx) == 0:
                # Fallback to random if no ND points (should not happen)
                parent_local = np.random.randint(len(all_F))
            else:
                # Calculate selection scores
                contribs = calculate_hv_contributions(nd_idx, all_F)
                
                # Normalize contribs to probabilities
                if np.sum(contribs) > 0:
                    probs = contribs / np.sum(contribs)
                    # Add small epsilon to avoid zero probs
                    probs += 1e-9
                    probs /= np.sum(probs)
                    
                    # Select parent based on score
                    parent_local_nd = np.random.choice(len(nd_idx), p=probs)
                    parent_local = nd_idx[parent_local_nd]
                else:
                    parent_local = nd_idx[np.random.randint(len(nd_idx))]
            
            parent_X = all_X[parent_local]
            
            # Determine step size
            if evals_used < polish_threshold:
                progress = (evals_used - init_pop) / max(1, polish_threshold - init_pop)
                scale = mutation_scale_start + (mutation_scale_end - mutation_scale_start) * progress
                scale = max(scale, mutation_scale_end)
            else:
                # Polish phase: very small steps
                scale = 0.005
            
            # Mutation
            # 5% chance of global restart to escape local optima
            if np.random.random() < 0.05:
                child_X = lo + (hi - lo) * np.random.random(dim)
            else:
                # Adaptive mutation: larger steps for dimensions that have more room
                # Simple Gaussian
                mutation = scale * (hi - lo) * np.random.standard_normal(dim)
                child_X = np.clip(parent_X + mutation, lo, hi)
                
                # In polish phase, try a "coordinate descent" style step
                if evals_used >= polish_threshold and np.random.random() < 0.5:
                    # Pick one dimension to perturb
                    dim_to_change = np.random.randint(dim)
                    step = scale * (hi[dim_to_change] - lo[dim_to_change]) * np.random.standard_normal()
                    child_X = parent_X.copy()
                    child_X[dim_to_change] = np.clip(child_X[dim_to_change] + step, lo[dim_to_change], hi[dim_to_change])

            child_F = np.array(problem(child_X), dtype=np.float64)
            evals_used += 1
            
            all_X.append(child_X)
            all_F.append(child_F)

        return None
# EVOLVE-BLOCK-END
