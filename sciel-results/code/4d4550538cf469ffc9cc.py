"""Combined optimizer for WFG9: fast vectorized non-dominated sorting with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast vectorized multi-objective optimizer with deadline guard and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _dominates(F):
        """Vectorized dominance check. D[i,j] True if row i dominates row j."""
        # F shape: (n, m)
        # i <= j in all objectives AND i < j in at least one
        le = F[:, None, :] <= F[None, :, :]
        lt = F[:, None, :] < F[None, :, :]
        return le.all(axis=2) & lt.any(axis=2)

    def _get_nd_indices(self, F):
        """Return indices of non-dominated points in F (vectorized)."""
        n = F.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        D = self._dominates(F)
        # A point is non-dominated if no other point dominates it
        # D[i,j] is True if i dominates j. So j is dominated if any i != j has D[i,j]=True
        dominated = D.any(axis=0)
        # Exclude self-domination (shouldn't happen with strict inequality, but safe)
        nd_mask = ~dominated
        return np.where(nd_mask)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Parameters
        pop_size = 40
        max_pop = 80
        polish_budget = 40
        evolve_budget = budget - polish_budget
        
        # RNG
        rng = np.random.default_rng(0)
        
        # Initialization
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive management
        # We maintain a list of (x, f) for the archive
        archive_X = []
        archive_F = []
        
        # Helper to update archive
        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if f is dominated by any in archive
            if archive_F:
                af = np.array(archive_F)
                # Check if any existing point dominates f
                # af[i] <= f and af[i] < f
                le = af <= f
                lt = af < f
                dominated_by_archive = (le.all(axis=1) & lt.any(axis=1)).any()
                if dominated_by_archive:
                    return False
            
            # Check if f dominates any in archive
            dominated_indices = []
            if archive_F:
                af = np.array(archive_F)
                le = f <= af
                lt = f < af
                dom_mask = (le.all(axis=1) & lt.any(axis=1))
                dominated_indices = np.where(dom_mask)[0]
            
            # Add to archive
            archive_X.append(x)
            archive_F.append(f)
            
            # Remove dominated
            if dominated_indices:
                new_X = []
                new_F = []
                for i in range(len(archive_X)):
                    if i not in dominated_indices:
                        new_X.append(archive_X[i])
                        new_F.append(archive_F[i])
                archive_X = new_X
                archive_F = new_F
                
            return True

        # Initialize archive
        for i in range(len(X)):
            update_archive(X[i], F[i])

        # Evolution Loop
        sigma = 0.15
        
        while used < evolve_budget:
            if time.time() > deadline:
                break
            
            # Decay sigma
            sigma *= 0.995
            
            # Select parent from archive
            if len(archive_X) > 0:
                p_idx = rng.integers(len(archive_X))
                parent = archive_X[p_idx]
            else:
                p_idx = rng.integers(len(X))
                parent = X[p_idx]
            
            # Mutate
            child = parent + sigma * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update archive
            is_nd = update_archive(child, fc)
            
            # Update population
            if is_nd:
                # Add to population
                if len(X) < max_pop:
                    X = np.vstack([X, [child]])
                    F = np.vstack([F, [fc]])
                else:
                    # Replace worst
                    # Use sum of objectives as a proxy for quality, or distance to ideal
                    # Simple: replace point with highest sum
                    sums = np.sum(F, axis=1)
                    worst_idx = np.argmax(sums)
                    # Only replace if child is better than worst in sum
                    if np.sum(fc) < sums[worst_idx]:
                        X[worst_idx] = child
                        F[worst_idx] = fc
            else:
                # Child is dominated. Replace worst if better
                sums = np.sum(F, axis=1)
                worst_idx = np.argmax(sums)
                if np.sum(fc) < sums[worst_idx]:
                    X[worst_idx] = child
                    F[worst_idx] = fc

        # Polish Phase
        if used < budget and len(archive_X) > 0:
            # Pick diverse points from archive for local search
            # Sort by sum to get "good" points, but also try to ensure diversity
            # For simplicity, pick the best 5 by sum
            if len(archive_X) > 5:
                sums = np.sum(archive_F, axis=1)
                top_indices = np.argsort(sums)[:5]
                polish_points = [archive_X[i] for i in top_indices]
            else:
                polish_points = archive_X[:]
            
            for p in polish_points:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                local_sigma = 0.05
                current_p = p
                current_f = None
                # Find f for current_p in archive
                for i, ax in enumerate(archive_X):
                    if np.allclose(ax, current_p):
                        current_f = archive_F[i]
                        break
                
                for _ in range(4):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    candidate = current_p + local_sigma * (hi - lo) * rng.standard_normal(dim)
                    candidate = np.clip(candidate, lo, hi)
                    fc = problem(candidate)
                    used += 1
                    
                    # Check if candidate improves current_p in any objective without worsening others too much?
                    # Or just check if it's non-dominated in the local neighborhood?
                    # Simple: if it's better than current_p in sum, or non-dominated w.r.t archive
                    is_nd = update_archive(candidate, fc)
                    
                    if is_nd:
                        current_p = candidate
                        local_sigma *= 0.8
                    else:
                        # If dominated, maybe try a different direction or stop
                        local_sigma *= 0.8
                        if local_sigma < 0.01:
                            break

        return X
# EVOLVE-BLOCK-END
