"""Multi-objective optimizer for WFG9 with deadline guard and adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive multi-objective optimizer with deadline guard for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondominated_indices(self, F):
        """Return indices of non-dominated points in F (shape n x n_obj)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        # F[i] dominates F[j] if F[i] <= F[j] and F[i] != F[j]
        # A point is non-dominated if no other point dominates it
        is_dom = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dom[i]:
                continue
            # Check if any other point dominates i
            # point j dominates i if F[j] <= F[i] and F[j] != F[i]
            le = np.all(F <= F[i], axis=1)
            lt = np.any(F < F[i], axis=1)
            dom_mask = le & lt
            dom_mask[i] = False  # ignore self
            if np.any(dom_mask):
                is_dom[i] = True
        return np.where(~is_dom)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Initial population: stratified sampling for coverage
        pop_size = min(50, max(10, budget // 4))
        # Use deterministic stratified sampling
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            # Divide range into pop_size equal intervals, pick random point in each
            edges = np.linspace(lo[d], hi[d], pop_size + 1)
            for i in range(pop_size):
                X[i, d] = np.random.uniform(edges[i], edges[i + 1])

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best hypervolume-related points
        # We'll maintain a population and replace dominated/worst points

        # Adaptive mutation: start with large steps, reduce as we find good points
        step_scale = 0.3  # fraction of range

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated points in current population
            nd_idx = self._nondominated_indices(F)

            if len(nd_idx) == 0:
                nd_idx = np.array([0])

            # Select a parent from non-dominated set (random)
            parent_idx = nd_idx[np.random.randint(len(nd_idx))]
            parent = X[parent_idx]

            # Adaptive mutation: Gaussian perturbation with adaptive scale
            # Use smaller steps as we converge
            progress = used / budget
            current_scale = step_scale * (1.0 - 0.5 * progress)
            mutation = np.random.randn(dim) * current_scale * (hi - lo)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst point in the population
            # Use a simple criterion: sum of objectives (or dominance)
            # Find the point that is dominated by the most others or has worst sum
            # For efficiency, replace the point with the highest sum of objectives
            # among dominated points, or the worst overall if none dominated

            # Check if child is non-dominated
            is_child_nd = True
            for i in range(len(F)):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    is_child_nd = False
                    break

            if is_child_nd:
                # Child is non-dominated; replace the worst dominated point
                # Find dominated points
                dominated_mask = np.zeros(len(F), dtype=bool)
                for i in range(len(F)):
                    for j in range(len(F)):
                        if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated_mask[i] = True
                            break
                if np.any(dominated_mask):
                    # Replace the one with highest sum among dominated
                    dom_indices = np.where(dominated_mask)[0]
                    worst_idx = dom_indices[np.argmax(F[dom_indices].sum(axis=1))]
                else:
                    # All non-dominated; replace the one with highest sum
                    worst_idx = np.argmax(F.sum(axis=1))
            else:
                # Child is dominated; only replace if it's better than the worst
                worst_idx = np.argmax(F.sum(axis=1))
                if fc.sum() < F[worst_idx].sum():
                    X[worst_idx] = child
                    F[worst_idx] = fc
                    continue
                else:
                    # Child is worse; still evaluate but don't replace
                    # Actually, we should still keep evaluating to explore
                    # But don't replace the population
                    continue

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
