"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Direct-structure optimizer for WFG7 with LHS initialization and adaptive local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Helper: Non-dominated indices
        def get_nd_indices(F):
            n = len(F)
            if n == 0:
                return []
            nd_idx = []
            for i in range(n):
                is_nd = True
                for j in range(n):
                    if i == j:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd_idx.append(i)
            return nd_idx

        # Phase 1: Latin Hypercube Sampling for diverse initialization
        n_init = int(budget * 0.6)
        # Use LHS for better spread than uniform random
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        sample = sampler.random(n=n_init)
        X_init = lo + sample * (hi - lo)
        
        X_all = []
        F_all = []
        
        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            X_all.append(x)
            F_all.append(f)
            
        used = len(F_all)
        
        if used == 0:
            return np.array([])
            
        X_all = np.array(X_all)
        F_all = np.array(F_all)

        # Phase 2: Adaptive Local Search
        # Use remaining budget for refinement
        # Step size decays over time to allow fine-tuning
        initial_step = 0.2 * (hi - lo)
        final_step = 0.02 * (hi - lo)
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Determine progress (0 to 1) to decay step size
            progress = used / budget
            current_step = initial_step * (1 - progress) + final_step * progress
            
            # Get current non-dominated set
            nd_idx = get_nd_indices(F_all)
            
            if not nd_idx:
                # Fallback: pick random parent
                parent_idx = np.random.randint(len(F_all))
            else:
                # Select parent from non-dominated set
                # Weighted selection: prefer points that are "better" (closer to origin)
                # Simple heuristic: normalize objectives and sum
                F_nd = F_all[nd_idx]
                # Normalize by max for comparison
                F_max = np.max(F_all, axis=0)
                F_norm = F_nd / (F_max + 1e-10)
                scores = np.sum(F_norm, axis=1)
                
                # Pick one of the top 3 best non-dominated points randomly
                k = min(3, len(scores))
                top_k_indices = np.argsort(scores)[:k]
                parent_local_idx = top_k_indices[np.random.randint(k)]
                parent_idx = nd_idx[parent_local_idx]
                
            parent = X_all[parent_idx]
            
            # Generate child: Gaussian perturbation
            noise = np.random.normal(0, 1, dim)
            child = parent + noise * current_step
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            
            X_all = np.vstack([X_all, child])
            F_all = np.vstack([F_all, f_child])
            used += 1

        return X_all

# EVOLVE-BLOCK-END
