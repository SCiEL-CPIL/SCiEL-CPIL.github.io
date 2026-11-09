"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HV-guided local search with adaptive step sizes."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1705.0, 12.0, 0.3])

        # Initialize with a diverse set of points
        n_init = min(20, self.budget // 4)
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # Track all evaluated points for hypervolume calculation
        all_X = X.copy()
        all_F = F.copy()

        # Adaptive step size
        step_size = 0.1

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find the point with highest marginal hypervolume contribution
            # We approximate this by finding the non-dominated point that is "most isolated"
            # in the objective space, as it likely contributes most to HV.
            
            # Simple approach: compute a proxy for HV contribution
            # For each point, compute the volume of the box it "owns" in objective space
            # This is expensive, so we use a simpler heuristic:
            # Find non-dominated points, then pick the one with largest distance to others
            
            # Compute non-dominated mask
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            if np.any(~dominated):
                non_dom_idx = np.where(~dominated)[0]
                # Among non-dominated, pick the one with largest average distance to others
                # This encourages diversity
                if len(non_dom_idx) > 1:
                    # Compute pairwise distances in objective space
                    F_nd = F[non_dom_idx]
                    # Normalize to [0,1] for distance calculation
                    F_min = F_nd.min(axis=0)
                    F_max = F_nd.max(axis=0)
                    F_range = F_max - F_min
                    F_range[F_range == 0] = 1
                    F_norm = (F_nd - F_min) / F_range
                    # Average distance to other non-dominated points
                    avg_dists = np.zeros(len(non_dom_idx))
                    for k in range(len(non_dom_idx)):
                        if k == 0:
                            avg_dists[k] = np.mean(np.linalg.norm(F_norm[1:] - F_norm[0], axis=1))
                        else:
                            avg_dists[k] = np.mean(np.linalg.norm(F_norm[:k] - F_norm[k], axis=1))
                    best_idx = non_dom_idx[np.argmax(avg_dists)]
                else:
                    best_idx = non_dom_idx[0]
            else:
                # All dominated, pick the best single point (lowest sum)
                best_idx = np.argmin(np.sum(F, axis=1))

            # Generate children around the selected point with adaptive step size
            parent = X[best_idx]
            n_children = min(5, self.budget - used)
            
            for c in range(n_children):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Adaptive mutation: mix of small and large steps
                if rng.random() < 0.3:
                    # Large exploration step
                    child = parent + step_size * 2 * (hi - lo) * rng.standard_normal(dim)
                else:
                    # Small exploitation step
                    child = parent + step_size * (hi - lo) * rng.standard_normal(dim)
                
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Add to archive
                all_X = np.vstack([all_X, child.reshape(1, -1)])
                all_F = np.vstack([all_F, fc.reshape(1, -1)])
                
                # Update population: replace worst point if child is better
                # Simple replacement: if child dominates any point, replace it
                # Otherwise, if child has lower sum, replace the worst
                dominated_count = 0
                for i in range(len(F)):
                    if np.all(fc <= F[i]) and np.any(fc < F[i]):
                        dominated_count += 1
                
                if dominated_count > 0:
                    # Replace a dominated point
                    for i in range(len(F)):
                        if np.all(fc <= F[i]) and np.any(fc < F[i]):
                            X[i] = child
                            F[i] = fc
                            break
                elif np.sum(fc) < np.sum(F[np.argmax(np.sum(F, axis=1))]):
                    # Replace worst point
                    worst = np.argmax(np.sum(F, axis=1))
                    X[worst] = child
                    F[worst] = fc

            # Adapt step size: if we found improvements, keep or increase slightly
            # Otherwise, decrease
            # Simple heuristic: track improvement rate
            # For now, slowly decay step size
            step_size *= 0.98

        return X
# EVOLVE-BLOCK-END
