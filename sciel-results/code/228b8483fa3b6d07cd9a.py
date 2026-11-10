# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Halton global sampling + Island-based local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, self.n_obj

    def _halton(self, n, dim, skip=10):
        """Generate n points in [0,1]^dim using Halton sequence."""
        def halton_index(i, base):
            f = 1.0
            r = 0.0
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                # Use the j-th prime
                # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37...
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
                base = primes[j % len(primes)]
                points[i, j] = halton_index(i + skip, base)
        return points

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1
        front0 = [i for i in range(n) if nd[i] == 0]
        fronts.append(front0)
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    def _island_local_search(self, problem, bounds, x0, weights, max_evals, lo, hi):
        """
        Perform local search guided by weighted sum scalarization.
        Returns best x and f found within max_evals.
        """
        best_x = x0.copy()
        f0 = problem(best_x)
        evals_used = 1
        best_score = np.dot(weights, f0)
        best_f = f0.copy()
        
        step_size = 0.1
        # Adaptive step size coordinate descent
        for _ in range(3): # Phases
            if evals_used >= max_evals:
                break
            improved = False
            for d in range(self.dim):
                if evals_used >= max_evals:
                    break
                for sign in [1, -1]:
                    if evals_used >= max_evals:
                        break
                    trial = best_x.copy()
                    trial[d] += sign * step_size * (hi[d] - lo[d])
                    trial = np.clip(trial, lo, hi)
                    ft = problem(trial)
                    evals_used += 1
                    score = np.dot(weights, ft)
                    if score < best_score:
                        best_score = score
                        best_x = trial
                        best_f = ft
                        improved = True
                        break
            if improved:
                step_size *= 0.8
            else:
                step_size *= 0.5
        return best_x, best_f, evals_used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget
        
        # 1. Global Sampling with Halton Sequence
        n_global = int(budget * 0.4) # 40% for global
        n_local = budget - n_global
        
        # Generate Halton points
        X_global = self._halton(n_global, dim)
        X_global = lo + (hi - lo) * X_global
        
        # Evaluate global points
        F_global = np.array([problem(x) for x in X_global])
        used = n_global
        
        X_all = X_global
        F_all = F_global
        
        # 2. Island-based Local Search
        # Define islands (weight vectors) to cover the Pareto front
        # We use a simplex lattice or random weights on the simplex
        n_islands = 5
        # Generate random weights on the simplex
        # Using Dirichlet(1,1,1) for 3 objectives
        weights_list = []
        for _ in range(n_islands):
            w = np.random.dirichlet(np.ones(n_obj))
            weights_list.append(w)
        
        # Distribute remaining budget among islands
        evals_per_island = max(1, n_local // n_islands)
        
        # For each island, pick a starting point from the global set that is best for that weight
        # Or just pick the best overall point if budget is tight.
        # Better: For each island, find the global point that minimizes the weighted sum.
        
        for w in weights_list:
            if used >= budget or time.time() > deadline:
                break
            
            # Find best global point for this weight
            scores = np.array([np.dot(w, f) for f in F_all])
            best_global_idx = np.argmin(scores)
            x0 = X_all[best_global_idx].copy()
            
            # Perform local search
            remaining = budget - used
            if remaining < 5:
                break
            
            x_best, f_best, evals_used_local = self._island_local_search(
                problem, bounds, x0, w, remaining, lo, hi
            )
            
            used += evals_used_local
            
            # Add result to archive
            # We just append to the lists, dominance sorting will happen at the end if needed
            # But the evaluator records ALL returned objective vectors? 
            # "Every returned objective vector is recorded" -> This implies we should return a list/array of all evaluated points?
            # "The return value of __call__ is ignored."
            # "Every returned objective vector is recorded" usually means the evaluator wraps problem() and records every call.
            # So we don't need to return the archive, we just need to make the calls.
            # However, to be safe and consistent with the previous structure which returned X_all, 
            # I will continue to maintain X_all and F_all, although the return is ignored.
            
            X_all = np.vstack([X_all, [x_best]])
            F_all = np.vstack([F_all, [f_best]])

        return X_all
# EVOLVE-BLOCK-END
