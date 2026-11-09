"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective search with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size: balance exploration vs exploitation
        pop_size = 30
        # Initial population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            m = len(F)
            S = [[] for _ in range(m)]
            n_dom = np.zeros(m, dtype=int)
            fronts = []
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([i])
            k = 0
            while k < len(fronts):
                current = fronts[k]
                next_front = []
                for p in current:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for obj in range(n_obj):
                vals = np.array([F[i, obj] for i in indices])
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                v_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if v_range > 1e-15:
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / v_range
            return cd

        # Main DE loop
        while used < n and time.time() < deadline:
            # Check time with margin
            if time.time() > deadline - 15:
                break
            fronts = non_dominated_sort(F)
            # Select parents from first front (or all if small)
            front0 = fronts[0]
            parents = np.array(front0) if len(front0) >= 3 else np.arange(pop_size)
            cd = crowding_distance(F, list(parents))
            # Weighted selection by crowding distance
            probs = cd / cd.sum()
            p1 = parents[rng.choice(len(parents), p=probs)]
            p2 = parents[rng.choice(len(parents), p=probs)]
            p3 = parents[rng.choice(len(parents), p=probs)]
            # Differential evolution
            if p1 == p2 or p1 == p3 or p2 == p3:
                # Fallback: random mutation
                base = rng.integers(pop_size)
                child = X[base] + 0.3 * span * rng.standard_normal(dim)
            else:
                F_ = 0.8 + 0.4 * rng.random()
                C_r = 0.5
                child = X[p1] + F_ * (X[p2] - X[p3])
                # Blend with random component
                mask = rng.random(dim) < C_r
                child[mask] = X[p1][mask] + F_ * (X[p2][mask] - X[p3][mask])
                child[~mask] = X[p1][~mask]
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst dominated individual or random
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            if np.any(dominated):
                worst = int(np.argmax(np.where(dominated, np.sum(F, axis=1), -np.inf)))
            else:
                worst = int(rng.integers(pop_size))
            X[worst] = child
            F[worst] = fc

        # Local refinement pass on non-dominated solutions
        if time.time() < deadline - 10:
            fronts = non_dominated_sort(F)
            pareto = np.array(fronts[0])
            # Refine each Pareto point with local search
            for idx in pareto:
                if time.time() > deadline - 5:
                    break
                base_x = X[idx].copy()
                base_f = F[idx].copy()
                step = 0.05
                for _ in range(8):
                    if time.time() > deadline - 3:
                        break
                    trial = base_x + step * span * rng.standard_normal(dim)
                    trial = np.clip(trial, lo, hi)
                    tf = problem(trial)
                    used += 1
                    if used >= n:
                        break
                    if np.all(tf <= base_f) and np.any(tf < base_f):
                        base_x = trial
                        base_f = tf
                        step *= 0.7
                    else:
                        step *= 0.5
                X[idx] = base_x
                F[idx] = base_f

        return X
# EVOLVE-BLOCK-END
