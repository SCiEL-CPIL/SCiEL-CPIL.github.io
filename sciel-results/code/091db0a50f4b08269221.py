import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 260.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget

        # WFG9 Pareto front is near the origin
        # Focus sampling heavily near lo (origin)
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_front(F_mat, n_total):
            """Return indices of non-dominated points only (faster than full sort)"""
            front_indices = []
            for i in range(n_total):
                dominated = False
                for j in range(n_total):
                    if i == j:
                        continue
                    if dominates(F_mat[j], F_mat[i]):
                        dominated = True
                        break
                if not dominated:
                    front_indices.append(i)
            return front_indices

        # Store all evaluated points and their objective values
        all_X = []
        all_F = []
        evals_used = 0

        # Phase 1: Broad exploration near origin (60% of budget)
        # Phase 2: Refined search around best points (30% of budget)
        # Phase 3: Final polish (10% of budget)
        
        phase1_budget = int(max_evals * 0.6)
        phase2_budget = int(max_evals * 0.3)
        phase3_budget = max_evals - phase1_budget - phase2_budget

        # Phase 1: Sample in a small region near origin
        # Use exponential decay to focus even more on very small values
        for _ in range(phase1_budget):
            if evals_used >= max_evals or time.time() > deadline:
                break
            
            # Sample with strong bias towards origin
            # Use power law: x = lo + (hi-lo) * u^k where k > 1 biases towards lo
            u = np.random.rand(self.dim)
            x = lo + (hi - lo) * (u ** 3.0)  # Strong bias towards lo
            
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            evals_used += 1

        # Prune to keep only non-dominated front
        if len(all_F) > 0:
            F_arr = np.array(all_F)
            front_idx = fast_non_dominated_front(F_arr, len(all_F))
            if len(front_idx) > 0:
                all_X = [all_X[i] for i in front_idx]
                all_F = [all_F[i] for i in front_idx]

        # Phase 2: Local search around best points
        if len(all_F) > 0 and evals_used < max_evals:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Pick a few diverse points from the front
            n_parents = min(5, len(all_X))
            if n_parents > 0:
                # Select parents spread across the front
                parent_indices = np.linspace(0, len(all_X)-1, n_parents).astype(int)
                parents_X = X_arr[parent_indices]
                parents_F = F_arr[parent_indices]
                
                for _ in range(phase2_budget):
                    if evals_used >= max_evals or time.time() > deadline:
                        break
                    
                    # Pick a random parent
                    p_idx = np.random.randint(len(parents_X))
                    parent = parents_X[p_idx]
                    
                    # Small perturbation
                    step_size = 0.05 * np.max(hi - lo)
                    perturbation = np.random.randn(self.dim) * step_size
                    x = parent + perturbation
                    x = np.clip(x, lo, hi)
                    
                    f = problem(x)
                    all_X.append(x)
                    all_F.append(f)
                    evals_used += 1

        # Phase 3: Final polish - very small steps around best points
        if len(all_F) > 0 and evals_used < max_evals:
            F_arr = np.array(all_F)
            front_idx = fast_non_dominated_front(F_arr, len(all_F))
            if len(front_idx) > 0:
                best_X = np.array([all_X[i] for i in front_idx])
                best_F = np.array([all_F[i] for i in front_idx])
                
                # Keep only top few for final polish
                n_final = min(3, len(best_X))
                if n_final > 0:
                    final_indices = np.linspace(0, len(best_X)-1, n_final).astype(int)
                    final_X = best_X[final_indices]
                    
                    for _ in range(phase3_budget):
                        if evals_used >= max_evals or time.time() > deadline:
                            break
                        
                        p_idx = np.random.randint(len(final_X))
                        parent = final_X[p_idx]
                        
                        # Very small step
                        step_size = 0.01 * np.max(hi - lo)
                        perturbation = np.random.randn(self.dim) * step_size
                        x = parent + perturbation
                        x = np.clip(x, lo, hi)
                        
                        f = problem(x)
                        all_X.append(x)
                        all_F.append(f)
                        evals_used += 1

        return np.array(all_X) if all_X else np.zeros((0, self.dim))
