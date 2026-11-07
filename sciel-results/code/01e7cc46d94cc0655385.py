"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Objective Differential Evolution with Archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin of 10s before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Initialize population
        # Use a moderate population size. 
        # Since dim=4 and n_obj=3, a population of ~50-100 is reasonable.
        # Let's use 80 to balance exploration and exploitation.
        pop_size = min(80, budget)
        
        # Random initialization
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        
        used = pop_size
        
        # Archive to keep non-dominated points found so far
        # We will maintain a list of (x, f) pairs that are non-dominated in the history
        # This helps in selecting parents that represent the current Pareto front approximation
        
        # Helper function to check if a point is dominated by another
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 in all objectives and f1 < f2 in at least one
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Helper to get non-dominated set from a list of F vectors
        def get_nondominated(F_list):
            if len(F_list) == 0:
                return []
            nondom_idx = []
            for i in range(len(F_list)):
                is_dominated = False
                for j in range(len(F_list)):
                    if i == j:
                        continue
                    if dominates(F_list[j], F_list[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nondom_idx.append(i)
            return nondom_idx

        # Initial archive
        archive_X = []
        archive_F = []
        
        # Add initial non-dominated points to archive
        init_nd_idx = get_nondominated(F)
        for idx in init_nd_idx:
            archive_X.append(X[idx].copy())
            archive_F.append(F[idx].copy())

        # Differential Evolution Parameters
        # CR (crossover rate)
        CR = 0.9
        # F (mutation weight) - adaptive
        F_base = 0.8
        F_min = 0.4
        F_max = 1.2
        
        # Adaptive F logic: decrease F if population stagnates, increase if exploring
        # Simple heuristic: track best hypervolume or just use a decaying F
        # For simplicity and robustness, we'll use a slightly decaying F to encourage exploitation later
        # But since we don't have easy HV calculation in the loop without overhead, we'll use a fixed F with noise
        # Or better: use the standard DE/rand/1/bin strategy
        
        while used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Select parents from the archive (non-dominated points)
            # If archive is small, use random from population
            if len(archive_X) > 3:
                # Select 3 distinct parents from archive
                parents_idx = np.random.choice(len(archive_X), size=3, replace=False)
                p1 = archive_X[parents_idx[0]]
                p2 = archive_X[parents_idx[1]]
                p3 = archive_X[parents_idx[2]]
            else:
                # Fallback to population if archive is too small
                parents_idx = np.random.choice(pop_size, size=3, replace=False)
                p1 = X[parents_idx[0]]
                p2 = X[parents_idx[1]]
                p3 = X[parents_idx[2]]
                
            # Mutation: DE/rand/1
            # v = p1 + F * (p2 - p3)
            F_current = F_base + 0.2 * np.random.random() # Add some noise
            v = p1 + F_current * (p2 - p3)
            
            # Crossover: Binomial
            j_rand = np.random.randint(dim)
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = p1[j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Update Archive
            # Check if trial is dominated by any point in archive
            is_dominated_by_archive = False
            dominated_indices = []
            
            for i, f_arch in enumerate(archive_F):
                if dominates(f_arch, f_trial):
                    is_dominated_by_archive = True
                    break
                elif dominates(f_trial, f_arch):
                    dominated_indices.append(i)
            
            if not is_dominated_by_archive:
                # Remove dominated points from archive
                for i in dominated_indices:
                    archive_X.pop(i)
                    archive_F.pop(i)
                
                # Add trial to archive
                archive_X.append(trial.copy())
                archive_F.append(f_trial.copy())
                
                # Limit archive size to prevent O(N^2) dominance checks from becoming too slow
                # If archive gets too large, prune it by keeping only the most "spread out" points
                # A simple heuristic: if len > 200, randomly remove some dominated or similar points?
                # Actually, just keeping all non-dominated is fine for 200 budget. 
                # Max non-dominated points in 4D with 200 evals is small.
                
            # Update Population (optional, but DE usually replaces the base vector)
            # In standard DE, we compare trial with p1 (the base vector).
            # Here, we want to maintain diversity. 
            # Let's replace p1 in the main population X with trial if trial is better or non-dominated
            # But since we are using archive for parents, the main population X is less critical 
            # except for the initial phase or if archive is empty.
            
            # For now, we just keep the population X as a history buffer or update it simply.
            # To ensure we don't lose good points, let's update X with trial if it's non-dominated in X
            # Actually, simpler: just use the archive for everything. 
            # But we need to fill the budget. The loop above does that.
            
            # To keep the population X somewhat representative for the "fallback" case,
            # we can replace a random member of X with trial.
            idx_replace = np.random.randint(pop_size)
            X[idx_replace] = trial
            F[idx_replace] = f_trial

        return X
# EVOLVE-BLOCK-END
