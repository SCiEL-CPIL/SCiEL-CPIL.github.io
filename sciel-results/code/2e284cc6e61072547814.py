"""Multi-objective optimizer on WFG9: NSGA-II style with global RNG + deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective search (Weighted Sum) minimizing all objectives."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _generate_weights(self, n_obj, points_per_edge=5):
        """Generate weight vectors on the simplex."""
        weights = []
        def gen(remaining, n_left, current):
            if n_left == 1:
                current.append(remaining)
                weights.append(current[:])
                current.pop()
                return
            for i in range(remaining + 1):
                current.append(i)
                gen(remaining - i, n_left - 1, current)
                current.pop()
        
        gen(points_per_edge * (n_obj - 1), n_obj, [])
        weights = np.array(weights, dtype=float)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def __call__(self, problem, bounds):
        dim, n_obj, budget = self.dim, self.n_obj, self.budget
        rng = np.random  # use the evaluator's global seed (seed 0 / seed 1)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        deadline = time.time() + 280.0   # safety margin vs 300s limit

        weights = self._generate_weights(n_obj, 5)
        n_weights = len(weights)
        
        X = lo + span * rng.random_sample((n_weights, dim))
        F = np.empty((n_weights, n_obj))
        used = 0
        for i in range(n_weights):
            F[i] = problem(X[i])
            used += 1
            
        W_sum = np.sum(weights * F, axis=1)

        while used < budget:
            if time.time() > deadline:
                break
            
            idx = rng.integers(n_weights)
            child = X[idx].copy()
            
            eta = 20.0
            for d in range(dim):
                if rng.random_sample() < 0.9:
                    u = rng.random_sample()
                    if u <= 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[d] += delta_q * span[d]
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            new_sum = np.dot(weights[idx], fc)
            if new_sum < W_sum[idx]:
                X[idx] = child
                F[idx] = fc
                W_sum[idx] = new_sum
                
            for j in range(n_weights):
                if j == idx:
                    continue
                sum_j = np.dot(weights[j], fc)
                if sum_j < W_sum[j]:
                    X[j] = child
                    F[j] = fc
                    W_sum[j] = sum_j

        return X
# EVOLVE-BLOCK-END
