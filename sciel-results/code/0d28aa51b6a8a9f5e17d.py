"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary search with adaptive mutation and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial random sampling
        pop_size = 20
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(f1, f2):
            """Check if f1 dominates f2 (all <= and at least one <)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def get_nondominated_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        def crowding_distance(F, nd_idx):
            """Compute crowding distance for non-dominated points."""
            m = len(nd_idx)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for k in range(self.n_obj):
                f_vals = F[nd_idx, k]
                sorted_idx = np.argsort(f_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, m - 1):
                    diff = f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]
                    if diff > 0:
                        cd[sorted_idx[i]] += diff / (f_vals.max() - f_vals.min() + 1e-12)
            return cd

        # Main evolutionary loop
        step_size = 0.15
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated parents
            nd_idx = get_nondominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Select parent using crowding distance among non-dominated
            cd = crowding_distance(F, nd_idx)
            # Weighted selection: prefer high crowding distance
            probs = cd / (cd.sum() + 1e-12)
            parent_local_idx = rng.choice(len(nd_idx), p=probs)
            parent_idx = nd_idx[parent_local_idx]

            # Adaptive mutation
            mutation = step_size * span * rng.standard_normal(self.dim)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point (by sum of normalized objectives)
            F_min = F.min(axis=0)
            F_max = F.max(axis=0)
            F_norm = (F - F_min) / (F_max - F_min + 1e-12)
            worst = int(np.argmax(np.sum(F_norm, axis=1)))
            X[worst] = child
            F[worst] = fc

            # Adapt step size: shrink if we've been making progress
            if used % 20 == 0:
                step_size *= 0.9

        # Final local refinement on best non-dominated points
        remaining = self.budget - used
        if remaining > 5 and time.time() < deadline - 10:
            nd_idx = get_nondominated_indices(F)
            # Take top 3 non-dominated points for local search
            top_nd = nd_idx[:min(3, len(nd_idx))]
            for idx in top_nd:
                if used >= self.budget or time.time() > deadline - 5:
                    break
                # Nelder-Mead style local search
                base = X[idx].copy()
                base_F = F[idx].copy()
                best_x = base.copy()
                best_f = base_F.copy()
                for _ in range(5):
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    # Try small perturbations
                    for dim_i in range(self.dim):
                        if used >= self.budget or time.time() > deadline - 5:
                            break
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline - 5:
                                break
                            trial = best_x.copy()
                            trial[dim_i] += direction * 0.02 * span[dim_i]
                            trial = np.clip(trial, lo, hi)
                            tf = problem(trial)
                            used += 1
                            # Check if trial dominates best_f
                            if dominates(tf, best_f):
                                best_x = trial
                                best_f = tf
                            elif dominates(best_f, tf):
                                pass
                            else:
                                # Non-dominated: keep both, update archive
                                pass
                # Update the point in our population
                if dominates(best_f, F[idx]):
                    X[idx] = best_x
                    F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
