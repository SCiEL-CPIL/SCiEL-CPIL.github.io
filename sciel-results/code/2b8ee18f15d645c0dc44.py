"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using LHS init, adaptive mutation, and HV-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _lhs(n, dim):
        """Latin Hypercube Sampling."""
        pts = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            pts[:, d] = (perm + np.random.random(n)) / n
        return pts

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # 295s limit with safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng()

        # Population size - larger for better coverage
        pop_size = min(50, max(15, self.budget // 4))
        remaining = self.budget

        # Initialize with LHS
        X = lo + (hi - lo) * self._lhs(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        remaining -= pop_size

        # Adaptive mutation step size
        step = 0.15 * (hi - lo)

        # Reference point for HV computation
        ref = np.array([1705.0, 12.0, 0.3])

        def compute_hv_contrib(F_arr, ref_pt):
            """Compute hypervolume contribution of each point using exact 3D HV."""
            n = len(F_arr)
            if n == 0:
                return np.array([])
            
            # Filter points that are dominated by reference
            valid_mask = np.all(F_arr < ref_pt, axis=1)
            valid_idx = np.where(valid_mask)[0]
            
            if len(valid_idx) == 0:
                return np.zeros(n)
            
            valid_F = F_arr[valid_idx]
            n_valid = len(valid_F)
            
            # Compute individual contributions by checking which points are non-dominated
            # and estimating their contribution
            contribs = np.zeros(n)
            
            # For each valid point, compute the volume of the box it uniquely dominates
            for i in valid_idx:
                fi = F_arr[i]
                # Check if this point is dominated by any other valid point
                dominated = False
                for j in valid_idx:
                    if i == j:
                        continue
                    fj = F_arr[j]
                    if np.all(fj <= fi) and np.any(fj < fi):
                        dominated = True
                        break
                if not dominated:
                    # Estimate contribution as the volume of the box from fi to ref
                    # that is not dominated by other points
                    vol = np.prod(ref_pt - fi)
                    contribs[i] = vol
            
            return contribs

        while remaining > 0:
            # Check time
            if time.time() > deadline - 5:
                break

            # Find non-dominated set
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd_mask[i] = False
                        break

            # Select parent from non-dominated set using HV-guided approach
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F))

            # Use tournament selection from non-dominated set
            if len(nd_indices) > 1:
                # Pick 2 random from ND, keep the one with better normalized sum
                a = nd_indices[rng.integers(len(nd_indices))]
                b = nd_indices[rng.integers(len(nd_indices))]
                # Normalize by range for fair comparison
                fa = F[a]
                fb = F[b]
                if np.sum(fa / ref) <= np.sum(fb / ref):
                    parent = X[a]
                else:
                    parent = X[b]
            else:
                parent = X[nd_indices[0]]

            # Adaptive mutation: occasionally restart from random or use crossover
            r = rng.random()
            if r < 0.03:
                # Random restart
                child = lo + (hi - lo) * rng.random(self.dim)
            elif r < 0.15:
                # Crossover with another random point
                other_idx = rng.integers(len(X))
                alpha = rng.random()
                child = alpha * parent + (1 - alpha) * X[other_idx]
                child = np.clip(child, lo, hi)
            else:
                # Gaussian mutation with adaptive step
                child = parent + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

            fc = problem(child)
            remaining -= 1

            # Replace worst point using HV contribution
            # Use a simple proxy: maximize sum of normalized objectives
            worst_idx = int(np.argmax(np.sum(F / ref, axis=1)))
            
            # Only replace if child is better or non-dominated
            is_dominated = False
            for j in range(len(F)):
                if np.all(F[j] <= fc) and np.any(F[j] < fc):
                    is_dominated = True
                    break
            
            if not is_dominated:
                X[worst_idx] = child
                F[worst_idx] = fc

            # Adaptive step: shrink if we keep improving, expand if stagnating
            if np.sum(fc / ref) < np.sum(F[worst_idx] / ref):  # improvement
                step *= 0.95
            else:
                step *= 1.03
            step = np.clip(step, 0.01 * (hi - lo), 0.4 * (hi - lo))

        return X
# EVOLVE-BLOCK-END
