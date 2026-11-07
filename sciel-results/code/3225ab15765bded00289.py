"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D-style decomposition-based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n_obj = self.n_obj
        dim = self.dim

        # Generate weight vectors using Tchebycheff scalarization
        # Use a grid of weights that sum to 1
        n_points = int(np.sqrt(self.budget // 2))  # ~10 points per dimension
        n_points = max(n_points, 5)

        # Generate weight vectors on a simplex
        weight_vectors = self._generate_weights(n_obj, n_points)
        n_subproblems = len(weight_vectors)

        # Initialize population
        rng = np.random.default_rng(0)
        pop = np.zeros((n_subproblems, dim))
        for i in range(n_subproblems):
            pop[i] = lo + (hi - lo) * rng.random(dim)

        # Evaluate initial population
        F = np.zeros((n_subproblems, n_obj))
        for i in range(n_subproblems):
            F[i] = problem(pop[i])
        used = n_subproblems

        # Track best per subproblem (Tchebycheff value)
        def tchebycheff(f, w):
            return np.max(w * np.abs(f - np.min(f, axis=0)))

        # External archive for non-dominated points
        archive_F = [F[i].copy() for i in range(n_subproblems)]
        archive_X = [pop[i].copy() for i in range(n_subproblems)]

        def is_dominated(f1, f2):
            """Check if f1 dominates f2"""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(f, x):
            """Add to archive if not dominated, remove dominated points"""
            new_idx = None
            for i, af in enumerate(archive_F):
                if is_dominated(f, af):
                    return  # dominated by archive
                if is_dominated(af, f):
                    archive_F.pop(i)
                    archive_X.pop(i)
                    i -= 1
                    continue
            archive_F.append(f.copy())
            archive_X.append(x.copy())

        # Initial archive
        archive_F = []
        archive_X = []
        for i in range(n_subproblems):
            add_to_archive(F[i], pop[i])

        # Neighborhood size
        T = min(n_subproblems, 5)

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Select a subproblem to update
            k = rng.integers(n_subproblems)

            # Get neighborhood
            neighbors = self._get_neighbors(k, n_subproblems, T, weight_vectors)

            # Crossover: pick two parents from neighborhood
            p1_idx = neighbors[rng.integers(len(neighbors))]
            p2_idx = neighbors[rng.integers(len(neighbors))]

            # SBX crossover
            child = self._sbx_crossover(pop[p1_idx], pop[p2_idx], lo, hi, rng)

            # Mutation
            child = self._polynomial_mutation(child, lo, hi, dim, rng)

            # Evaluate
            fc = problem(child)
            used += 1

            # Update subproblem k and possibly neighbors
            # Check if child is better for subproblem k
            if self._better(fc, F[k], weight_vectors[k]):
                pop[k] = child
                F[k] = fc
                add_to_archive(fc, child)

            # Also check neighbors
            for ni in neighbors:
                if ni != k and self._better(fc, F[ni], weight_vectors[ni]):
                    # With some probability, replace
                    if rng.random() < 0.3:
                        pop[ni] = child
                        F[ni] = fc

        # Final: combine population and archive, return
        # Return the population (the evaluator records all returned objective vectors)
        return pop

    def _generate_weights(self, n_obj, n_points):
        """Generate weight vectors on the simplex"""
        weights = []
        if n_obj == 3:
            for i in range(n_points + 1):
                for j in range(n_points + 1 - i):
                    w1 = i / n_points
                    w2 = j / n_points
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0:
                        weights.append(np.array([w1, w2, w3]))
        elif n_obj == 2:
            for i in range(n_points + 1):
                w1 = i / n_points
                weights.append(np.array([w1, 1.0 - w1]))
        else:
            # For general case, use random weights
            for _ in range(n_points * n_points):
                w = rng.random(n_obj)
                w /= np.sum(w)
                weights.append(w)
        return np.array(weights) if weights else np.array([[1.0/n_obj]*n_obj])

    def _get_neighbors(self, k, n_subproblems, T, weight_vectors):
        """Get T nearest neighbors to subproblem k"""
        if k >= n_subproblems:
            return [0]
        w_k = weight_vectors[k]
        dists = np.array([np.linalg.norm(weight_vectors[i] - w_k) for i in range(n_subproblems)])
        dists[k] = np.inf
        neighbors = np.argsort(dists)[:T]
        return list(neighbors)

    def _sbx_crossover(self, x1, x2, lo, hi, rng, eta=20.0):
        """Simulated Binary Crossover"""
        child = x1.copy()
        for i in range(len(x1)):
            if rng.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                else:
                    child[i] = x1[i]
        return np.clip(child, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, dim, rng, eta=20.0, prob=0.1):
        """Polynomial mutation"""
        child = x.copy()
        for i in range(dim):
            if rng.random() < prob:
                dx = hi[i] - lo[i]
                if dx < 1e-14:
                    continue
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                child[i] = x[i] + delta * dx
                child[i] = np.clip(child[i], lo[i], hi[i])
        return child

    def _better(self, f_new, f_old, w):
        """Check if f_new is better than f_old using Tchebycheff with weight w"""
        # Use Tchebycheff distance to ideal point
        # Approximate ideal as min of current population
        return np.max(w * np.abs(f_new - f_old)) < np.max(w * np.abs(f_old - f_old)) + 1e-10 \
               and np.sum(f_new) < np.sum(f_old) + 1e-10
# EVOLVE-BLOCK-END
