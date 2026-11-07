"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using an evolutionary strategy with deadline guard
    and tracked archive for hypervolume computation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dominance_mask(self, F):
        """Return boolean mask where True means dominated by another point in F."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        # Vectorized: F[j] <= F[i] for all objectives AND F[j] < F[i] for at least one
        # Shape: (n, n, n_obj) -> compare all pairs
        # F[i] vs F[j]: F[j] <= F[i] means F[i] - F[j] >= 0
        diff = F[np.newaxis, :, :] - F[:, np.newaxis, :]  # diff[i,j,k] = F[i,k] - F[j,k]
        # j dominates i if: all(diff[i,j,:] >= 0) and any(diff[i,j,:] > 0)
        all_leq = np.all(diff >= 0, axis=2)  # (n, n): all_leq[i,j] = F[j] <= F[i]
        any_lt = np.any(diff > 0, axis=2)     # (n, n): any_lt[i,j] = F[j] < F[i] in some obj
        dom_by = all_leq & any_lt              # (n, n): dom_by[i,j] = j dominates i
        np.fill_diagonal(dom_by, False)
        dominated = np.any(dom_by, axis=1)
        return dominated

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of points (assumed non-dominated)."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for k in range(F.shape[1]):
            order = np.argsort(F[:, k])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                if F[order[i+1], k] - F[order[i-1], k] > 0:
                    cd[order[i]] += (F[order[i+1], k] - F[order[i-1], k]) / (F[order[i+1], k] - F[order[i-1], k] + 1e-15)
                else:
                    cd[order[i]] += 0
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Initialize population
        pop_size = min(30, self.budget // 3)
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_F = list(F)

        # Adaptive step size
        base_step = 0.15
        step_decay = 0.9995

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            elapsed_frac = (time.time() - start_time) / 280.0
            current_step = base_step * (1.0 - 0.5 * elapsed_frac)
            
            # Find non-dominated solutions using vectorized dominance
            dominated = self._dominance_mask(F)
            non_dominated = ~dominated
            if not np.any(non_dominated):
                non_dominated = np.ones(len(F), dtype=bool)
            
            nd_indices = np.where(non_dominated)[0]
            
            # Select parent using crowding distance for diversity
            if len(nd_indices) > 1:
                nd_F = F[nd_indices]
                cd = self._crowding_distance(nd_F)
                # Probabilistic selection weighted by crowding distance
                probs = cd / cd.sum()
                parent_local_idx = rng.choice(len(nd_indices), p=probs)
                parent_idx = nd_indices[parent_local_idx]
            else:
                parent_idx = nd_indices[0]
            
            parent = X[parent_idx]
            
            # Generate child with adaptive step size
            child = parent + current_step * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Record point
            all_F.append(fc)
            
            # Replace worst solution: find the most crowded dominated solution
            # First, try to replace a dominated solution
            dominated_indices = np.where(dominated)[0]
            if len(dominated_indices) > 0:
                # Among dominated, replace the one with highest crowding distance
                dom_F = F[dominated_indices]
                if len(dom_F) > 1:
                    cd_dom = self._crowding_distance(dom_F)
                    worst_local = int(np.argmax(cd_dom))
                    worst_idx = dominated_indices[worst_local]
                else:
                    worst_idx = dominated_indices[0]
            else:
                # All non-dominated: replace the most crowded one
                if len(F) > 1:
                    cd_all = self._crowding_distance(F)
                    worst_idx = int(np.argmax(cd_all))
                else:
                    worst_idx = 0
            
            X[worst_idx] = child
            F[worst_idx] = fc

        return all_F
# EVOLVE-BLOCK-END
