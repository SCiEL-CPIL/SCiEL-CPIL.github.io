import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Global Exploration with Sobol Sequence
        # Allocate 50% of budget to global exploration
        n_init = self.budget // 2
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol requires power of 2 for best performance, but we can use any number
        # We round up to next power of 2 for sampler generation, then slice
        n_sobol = 2 ** int(np.ceil(np.log2(n_init)))
        points = sampler.random(n_sobol)
        points = points[:n_init]
        
        # Scale to bounds
        X_init = lo + points * (hi - lo)
        
        # Evaluate
        F_init = np.array([problem(x) for x in X_init])
        
        # Phase 2: Local Refinement
        # Identify Pareto front from initial population
        pareto_indices = self._pareto_front_indices(F_init)
        
        if len(pareto_indices) == 0:
            # Fallback if all dominated (should not happen for first front)
            return None

        # Select top candidates for local search
        n_local_candidates = min(10, len(pareto_indices))
        
        # Sort Pareto front by first objective to select diverse points
        pareto_F = F_init[pareto_indices]
        sorted_pareto_idx = np.argsort(pareto_F[:, 0])
        pareto_indices_sorted = pareto_indices[sorted_pareto_idx]
        
        # Select n_local_candidates evenly
        if n_local_candidates > 1:
            step = len(pareto_indices_sorted) / n_local_candidates
            selected_indices = [pareto_indices_sorted[int(i * step)] for i in range(n_local_candidates)]
        else:
            selected_indices = [pareto_indices_sorted[0]]
            
        # Local Search Budget
        remaining_budget = self.budget - n_init
        # Distribute remaining budget among selected candidates
        # We will do "Best of K" local search.
        # Let's say we do 2 iterations of "Best of 5" for each candidate.
        # That is 10 evaluations per candidate.
        # If we have 10 candidates, that's 100 evals.
        
        n_evals_per_candidate = remaining_budget // len(selected_indices)
        
        # To be safe, let's use a fixed structure:
        # If n_evals_per_candidate >= 5, do Best of 5 for (n_evals_per_candidate // 5) iterations.
        # If n_evals_per_candidate < 5, do simple random walk.
        
        for i, idx in enumerate(selected_indices):
            x_curr = X_init[idx].copy()
            f_curr = F_init[idx].copy()
            
            evals_used_for_this = 0
            while evals_used_for_this < n_evals_per_candidate:
                # Determine how many neighbors to generate in this batch
                # Max 5 neighbors per batch
                batch_size = min(5, n_evals_per_candidate - evals_used_for_this)
                
                neighbors = []
                neighbor_Fs = []
                
                # Generate neighbors
                scale = 0.05 * (hi - lo)
                for _ in range(batch_size):
                    delta = np.random.randn(self.dim) * scale
                    x_new = x_curr + delta
                    x_new = np.clip(x_new, lo, hi)
                    neighbors.append(x_new)
                    
                # Evaluate neighbors
                for x_new in neighbors:
                    f_new = np.array(problem(x_new))
                    neighbor_Fs.append(f_new)
                    evals_used_for_this += 1
                    
                # Select the best neighbor that dominates x_curr
                best_neighbor_idx = -1
                best_neighbor_F = None
                for j in range(len(neighbors)):
                    f_new = neighbor_Fs[j]
                    if np.all(f_new <= f_curr) and np.any(f_new < f_curr):
                        if best_neighbor_F is None:
                            best_neighbor_idx = j
                            best_neighbor_F = f_new
                        else:
                            # If multiple dominate, pick the one that is "better" in some sense
                            # Simple: pick the first one found, or compare them
                            # For simplicity, keep the first one found
                            pass
                
                if best_neighbor_idx != -1:
                    x_curr = neighbors[best_neighbor_idx]
                    f_curr = best_neighbor_F
            # End local search for this candidate
        return None

    def _pareto_front_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]
