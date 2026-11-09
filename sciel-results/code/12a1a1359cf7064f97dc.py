"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _hypervolume_contribution(self, F, ref):
        """Compute simple hypervolume contribution for each point (approximate)."""
        n = len(F)
        if n == 0:
            return np.array([])
        contribs = np.zeros(n)
        for i in range(n):
            # Point i contributes if it's not dominated by any other point
            # Simple approximation: count how many objectives are strictly best
            is_best = 0
            for j in range(self.n_obj):
                if F[i, j] == np.min(F[:, j]):
                    is_best += 1
            contribs[i] = is_best
        return contribs

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s/330s limits
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Initial population
        pop_size = min(20, self.budget // 5)
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume (from task)
        ref = np.array([3.0, 5.0, 7.0])
        
        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            non_dom_idx = np.where(~dominated)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))
            
            # Select parent from non-dominated, weighted by hypervolume contribution
            contribs = self._hypervolume_contribution(F[non_dom_idx], ref)
            total_contrib = np.sum(contribs)
            if total_contrib == 0:
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            else:
                probs = contribs / total_contrib
                parent_idx = non_dom_idx[rng.choice(len(non_dom_idx), p=probs)]
            
            # Adaptive mutation: larger early, smaller late
            progress = used / self.budget
            mutation_scale = 0.3 * (1 - progress) + 0.05
            
            # Generate child
            child = X[parent_idx] + mutation_scale * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace worst point (by sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc
        
        # Local polish phase: use remaining budget for local search on best points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Get non-dominated points for polishing
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            non_dom_idx = np.where(~dominated)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))
            
            # Polish each non-dominated point with local search
            polish_per_point = max(1, remaining // max(1, len(non_dom_idx)))
            for idx in non_dom_idx:
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Local search with decreasing step size
                for step in [0.05, 0.02, 0.01, 0.005]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    best_x = X[idx].copy()
                    best_f = F[idx].copy()
                    
                    for dim_i in range(self.dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        for direction in [-1, 1]:
                            trial = best_x.copy()
                            trial[dim_i] += direction * step * range_[dim_i]
                            trial = np.clip(trial, lo, hi)
                            
                            if np.array_equal(trial, best_x):
                                continue
                            
                            trial_f = problem(trial)
                            used += 1
                            
                            # Check if trial dominates best
                            if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                                best_x = trial
                                best_f = trial_f
                                X[idx] = best_x
                                F[idx] = best_f
        
        return X
# EVOLVE-BLOCK-END
