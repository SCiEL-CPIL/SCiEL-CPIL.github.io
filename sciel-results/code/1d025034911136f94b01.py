"""Multi-objective optimizer for ZDT4 using a refined population-based search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F (minimization)."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(i + 1, n):
                if dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n_dim = self.dim

        # Population size: balance exploration and exploitation
        pop_size = min(30, max(10, self.budget // 10))
        # Ensure we can do at least one generation
        if pop_size * 2 > self.budget:
            pop_size = max(5, self.budget // 4)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n_dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best solutions found (for return)
        all_points = []
        for f in F:
            all_points.append(f.copy())

        # Adaptive mutation step size
        base_step = 0.15
        step = base_step

        # Main search loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_idx = self._non_dominated_indices(F)
            
            # If all are dominated (shouldn't happen often), use all
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Select parents from non-dominated set with diversity
            n_parents = min(len(nd_idx), max(2, pop_size // 3))
            
            # Use tournament-like selection from non-dominated
            parent_indices = []
            for _ in range(n_parents):
                # Pick 2 random from non-dominated, take the one with better spread
                idx1 = nd_idx[rng.integers(len(nd_idx))]
                idx2 = nd_idx[rng.integers(len(nd_idx))]
                # Simple: just pick randomly
                parent_indices.append(idx1)

            # Generate children
            n_children = min(pop_size // 2, self.budget - used)
            if n_children <= 0:
                break

            children_X = []
            children_F = []

            for c in range(n_children):
                # Check time periodically
                if c % 5 == 0 and time.time() > deadline:
                    break

                # Select two parents (or one for mutation)
                if len(parent_indices) >= 2 and rng.random() < 0.5:
                    p1_idx = parent_indices[rng.integers(len(parent_indices))]
                    p2_idx = parent_indices[rng.integers(len(parent_indices))]
                    # SBX-like crossover
                    child = (X[p1_idx] + X[p2_idx]) / 2.0
                    # Add mutation
                    child = child + step * (hi - lo) * rng.standard_normal(n_dim)
                else:
                    p1_idx = parent_indices[rng.integers(len(parent_indices))]
                    child = X[p1_idx].copy()
                    # Mutation
                    child = child + step * (hi - lo) * rng.standard_normal(n_dim)

                # Clip to bounds
                child = np.clip(child, lo, hi)

                # Evaluate
                fc = problem(child)
                children_X.append(child)
                children_F.append(fc)
                used += 1

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine population with children
            X_new = np.vstack([X, children_X])
            F_new = np.vstack([F, children_F])

            # Truncate to population size using non-dominated sorting + crowding
            # For simplicity, keep non-dominated points and fill with best dominated
            nd_idx_new = self._non_dominated_indices(F_new)
            
            if len(nd_idx_new) >= pop_size:
                # Keep only non-dominated, select pop_size of them
                # Use diversity: spread out the selected points
                nd_F = F_new[nd_idx_new]
                # Sort by first objective, then select spread
                order = np.argsort(nd_F[:, 0])
                selected = []
                remaining = list(order)
                
                # Greedy diversity selection
                selected.append(remaining.pop(rng.integers(len(remaining))))
                while len(selected) < pop_size and remaining:
                    # Pick the point farthest from already selected
                    best_idx = 0
                    best_dist = -1
                    for ri, ridx in enumerate(remaining):
                        dist = sum(
                            min(abs(nd_F[ridx, 0] - nd_F[s, 0]), 
                                abs(nd_F[ridx, 1] - nd_F[s, 1]))
                            for s in selected
                        ) / len(selected)
                        if dist > best_dist:
                            best_dist = dist
                            best_idx = ri
                    selected.append(remaining.pop(best_idx))
                
                X = X_new[nd_idx_new[selected]]
                F = F_new[nd_idx_new[selected]]
            else:
                # Keep all non-dominated, fill with best dominated
                dominated_idx = np.setdiff1d(np.arange(len(F_new)), nd_idx_new)
                if len(dominated_idx) > 0:
                    # Sort dominated by sum of objectives
                    dom_F = F_new[dominated_idx]
                    dom_order = np.argsort(np.sum(dom_F, axis=1))
                    need = pop_size - len(nd_idx_new)
                    fill_idx = dominated_idx[dom_order[:need]]
                    X = np.vstack([X_new[nd_idx_new], X_new[fill_idx]])
                    F = np.vstack([F_new[nd_idx_new], F_new[fill_idx]])
                else:
                    X = X_new
                    F = F_new

            # Record all points
            for f in F:
                all_points.append(f.copy())

            # Adapt step size: reduce if we're converging
            step = max(0.02, step * 0.95)

        # Final polish: local search around best points
        if time.time() < deadline - 10:
            # Sort by first objective
            order = np.argsort(F[:, 0])
            top_k = min(5, len(F))
            for idx in order[:top_k]:
                if time.time() > deadline:
                    break
                # Local search
                x = X[idx].copy()
                fx = F[idx].copy()
                for _ in range(10):
                    if used >= self.budget or time.time() > deadline:
                        break
                    trial = x + 0.01 * (hi - lo) * rng.standard_normal(n_dim)
                    trial = np.clip(trial, lo, hi)
                    ftrial = problem(trial)
                    used += 1
                    # Accept if it dominates or is non-dominated with better sum
                    if np.all(ftrial <= fx) and np.any(ftrial < fx):
                        x = trial
                        fx = ftrial
                        X[idx] = x
                        F[idx] = fx
                        all_points.append(fx.copy())
                    elif not np.any(np.all(F <= ftrial, axis=1) & np.any(F < ftrial, axis=1)):
                        # Non-dominated, replace worst
                        worst = int(np.argmax(np.sum(F, axis=1)))
                        if np.sum(ftrial) < np.sum(F[worst]):
                            X[worst] = trial
                            F[worst] = ftrial
                            all_points.append(ftrial.copy())

        return X
# EVOLVE-BLOCK-END
