"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Sobol sweep + local Gaussian refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Efficient non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=np.int32)

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1

        front = [i for i in range(n) if nd[i] == 0]
        fronts = []
        if not front:
            front = [0]
        fronts.append(front)

        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front

        return fronts

    def _generate_sobol(self, n, dim):
        """Generate n points in [0,1]^dim using a simple Sobol-like sequence."""
        # Use a simple LCG-based quasi-random sequence for reproducibility
        # and good coverage
        points = np.zeros((n, dim))
        # Use Halton-like sequence with different bases for each dimension
        bases = [2, 3, 5, 7]  # first 4 primes
        for i in range(n):
            for d in range(dim):
                b = bases[d % len(bases)]
                x = 0.0
                f = 1.0 / b
                j = i + 1
                while j > 0:
                    x += f * (j % b)
                    j //= b
                    f /= b
                points[i, d] = x
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Phase 1: Deterministic Sobol sweep
        # Use about 60% of budget for the initial sweep
        sweep_budget = int(0.6 * self.budget)
        # Ensure at least 32 points for good coverage
        sweep_budget = max(32, min(sweep_budget, self.budget - 20))

        X_sweep = self._generate_sobol(sweep_budget, self.dim)
        X_sweep = np.clip(X_sweep, 0, 1)
        F_sweep = np.array([problem(x) for x in X_sweep])
        used = sweep_budget

        all_X = X_sweep.copy()
        all_F = F_sweep.copy()

        # Phase 2: Local refinement around non-dominated points
        # Find non-dominated points from the sweep
        fronts = self._fast_non_dominated_sort(F_sweep)
        if fronts:
            nd_indices = fronts[0]
            # Sort by hypervolume contribution proxy (sum of normalized objectives, lower is better)
            F_nd = F_sweep[nd_indices]
            F_min = F_sweep.min(axis=0)
            F_max = F_sweep.max(axis=0)
            F_range = np.where(F_max > F_min, F_max - F_min, 1.0)
            F_norm = (F_nd - F_min) / F_range
            scores = F_norm.sum(axis=1)
            nd_sorted = np.argsort(scores)
            # Take top 10 non-dominated points for refinement
            n_refine = min(10, len(nd_sorted))
            refine_indices = [nd_indices[i] for i in nd_sorted[:n_refine]]
        else:
            refine_indices = []

        # Refine each non-dominated point with local Gaussian perturbations
        remaining = self.budget - used
        if refine_indices and remaining > 0:
            # Distribute remaining budget among refinement points
            evals_per_point = max(1, remaining // len(refine_indices))
            
            for idx in refine_indices:
                if used >= self.budget or time.time() > deadline - 5:
                    break
                
                x_center = X_sweep[idx].copy()
                f_center = F_sweep[idx].copy()
                
                # Adaptive step size: start moderate, shrink
                for step in [0.15, 0.1, 0.05, 0.03, 0.02, 0.01]:
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    
                    # Generate Gaussian perturbations
                    n_trials = min(evals_per_point // 6, 8)  # up to 8 trials per step size
                    for _ in range(n_trials):
                        if used >= self.budget or time.time() > deadline - 5:
                            break
                        
                        x_trial = x_center + step * np.random.standard_normal(self.dim)
                        x_trial = np.clip(x_trial, 0, 1)
                        f_trial = problem(x_trial)
                        used += 1
                        
                        # Accept if it improves on f_center in at least one objective
                        # and doesn't worsen others too much
                        if np.all(f_trial <= f_center + 1e-6) and np.any(f_trial < f_center - 1e-6):
                            x_center = x_trial
                            f_center = f_trial
                        
                        # Also record for hypervolume
                        all_X = np.vstack([all_X, x_trial.reshape(1, -1)])
                        all_F = np.vstack([all_F, f_trial.reshape(1, -1)])

        return all_X
# EVOLVE-BLOCK-END
