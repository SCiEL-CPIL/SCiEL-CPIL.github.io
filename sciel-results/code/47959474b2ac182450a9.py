"""Multi-objective optimizer using MOSA (Multi-Objective Simulated Annealing) for RE34."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Objective Simulated Annealing optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random

        # Reference point for hypervolume calculation
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Initialize: structured points
        init_points = []
        # Corners
        for i in range(min(16, 2 ** dim)):
            point = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    point[j] = hi[j]
                else:
                    point[j] = lo[j]
            init_points.append(point)
        # Center
        init_points.append((lo + hi) / 2.0)
        # Random fill
        pop_size = min(40, budget // 4)
        while len(init_points) < pop_size:
            init_points.append(lo + span * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive: list of (x, f)
        archive = []

        def is_dominated_by_any(f, archive_f_list):
            for a in archive_f_list:
                if np.all(a <= f) and np.any(a < f):
                    return True
            return False

        def add_to_archive(archive, x, f):
            """Add to archive if non-dominated, remove dominated points."""
            archive_f = [a[1] for a in archive]
            if is_dominated_by_any(f, archive_f):
                return archive
            
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            return new_archive

        for i in range(len(X)):
            archive = add_to_archive(archive, X[i], F[i])

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D non-dominated points."""
            if len(points) == 0:
                return 0.0
            # Filter points that are strictly better than ref in all objectives
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            
            # Sort by first objective
            valid.sort(key=lambda p: p[0])
            
            total_hv = 0.0
            prev_y = ref[1]
            prev_z = ref[2]
            
            for p in valid:
                x, y, z = p
                # Contribution: box from (x,y,z) to (next_x, prev_y, prev_z) in sorted order
                # This is a simplified approximation; exact 3D HV is complex.
                # For scoring purposes, we need a consistent metric.
                # Let's use a simpler dominance-based metric instead.
                pass
            
            # Actually, let's use a simpler "spread + closeness" metric for speed
            # or just count non-dominated points as a proxy.
            # Since exact HV is expensive, let's use a fast approximation:
            # Sum of volumes of boxes from each point to the reference, minus overlaps.
            # But for acceptance criteria, we can use a simpler score.
            
            return 0.0  # Placeholder, we'll use a different acceptance metric

        # Better acceptance metric: weighted sum of improvements
        def compute_score(f, archive_f):
            """Compute a score for acceptance based on hypervolume contribution approximation."""
            # Simple: if non-dominated, high score. If dominated, low score.
            # We'll use a temperature-based acceptance instead.
            return 0.0

        # MOSA parameters
        T0 = 1.0  # Initial temperature
        T_min = 0.01  # Minimum temperature
        cooling_rate = 0.995  # Geometric cooling
        
        # Current solution
        current_idx = 0
        current_x = X[current_idx].copy()
        current_f = F[current_idx].copy()
        current_archive = archive[:]  # Copy archive for current state
        
        best_archive = archive[:]  # Keep track of best archive found
        
        # Temperature schedule
        T = T0
        
        while used < budget:
            if time.time() - start_time > deadline:
                break
            
            # Generate neighbor
            # Use DE-like mutation for larger steps, or Gaussian for smaller
            if rng.random() < 0.5:
                # DE mutation
                idxs = rng.choice(len(X), size=3, replace=False)
                v = X[idxs[0]] + 0.8 * (X[idxs[1]] - X[idxs[2]])
            else:
                # Gaussian perturbation
                sigma = 0.5 * span
                v = current_x + rng.normal(0, sigma, dim)
            
            trial_x = np.clip(v, lo, hi)
            trial_f = problem(trial_x)
            used += 1
            
            # Update archive with trial
            trial_archive = add_to_archive(current_archive, trial_x, trial_f)
            
            # Acceptance criterion
            # 1. If trial is non-dominated and improves hypervolume (approx), accept
            # 2. If dominated, accept with probability exp(-delta/T)
            
            # Simple acceptance: 
            # - If trial is not dominated by current_archive, accept (improves or maintains)
            # - If dominated, accept with probability based on temperature
            
            dominated_by_current = is_dominated_by_any(trial_f, [a[1] for a in current_archive])
            
            if not dominated_by_current:
                # Non-dominated: accept and update
                current_x = trial_x
                current_f = trial_f
                current_archive = trial_archive
                
                # Update best archive
                best_archive = add_to_archive(best_archive, trial_x, trial_f)
                
                # Add to population
                X[current_idx] = trial_x
                F[current_idx] = trial_f
            else:
                # Dominated: accept with probability
                # Estimate "delta" as how much worse it is
                # Use a simple penalty: number of objectives worse
                delta = np.sum(trial_f > current_f) * 0.1 + np.sum(trial_f - current_f) * 0.01
                delta = max(delta, 0.01)
                
                if rng.random() < np.exp(-delta / T):
                    current_x = trial_x
                    current_f = trial_f
                    # Don't update archive since it's dominated
                    X[current_idx] = trial_x
                    F[current_idx] = trial_f
            
            # Cool down
            T = max(T * cooling_rate, T_min)
            
            # Periodically restart with new random point to escape local optima
            if used % 100 == 0 and used < budget - 10:
                restart_x = lo + span * rng.random(dim)
                restart_f = problem(restart_x)
                used += 1
                current_x = restart_x
                current_f = restart_f
                current_archive = add_to_archive(best_archive, restart_x, restart_f)
                X[current_idx] = restart_x
                F[current_idx] = restart_f

        return X
# EVOLVE-BLOCK-END
