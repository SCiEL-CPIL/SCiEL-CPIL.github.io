"""Multi-objective optimizer using differential evolution with LHS init and adaptive HV-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer: DE-based search with LHS init and adaptive HV-guided archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        # 1. Initialization: Latin Hypercube Sampling for superior spread
        n_init = min(30, max(10, budget // 7))
        
        # Generate LHS points in [0, 1]^dim
        lhs = qmc.LatinHypercube(d=dim, seed=0)
        X_unit = lhs.random(n_init)
        
        # Map to bounds
        X = lo + (hi - lo) * X_unit
        F = np.array([problem(x) for x in X])
        used = n_init

        # Archive: maintain a diverse set of non-dominated points
        archive_X = [X[i].copy() for i in range(n_init)]
        archive_F = [F[i].copy() for i in range(n_init)]

        def dominates(a, b):
            """Check if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(point, front):
            """Check if point is non-dominated with respect to front."""
            for f in front:
                if dominates(f, point):
                    return False
            return True

        def remove_dominated(point, front):
            """Remove points from front that are dominated by point."""
            return [f for f in front if not dominates(point, f)]

        def update_archive(point, fval, front_X, front_F):
            """Update the Pareto front with a new point. Returns new front lists."""
            if not is_non_dominated(fval, front_F):
                return front_X, front_F
            
            # Add new point
            new_front_F = remove_dominated(fval, front_F)
            new_front_F.append(fval)
            
            # Rebuild X list to match new F list
            # We need to keep track of which X corresponds to which F.
            # Since we are removing dominated F points, we must also remove corresponding X points.
            # This implementation is slightly inefficient (O(N^2)) but N is small (<200).
            
            # Let's rebuild X from scratch based on the new F list is hard because we don't have the mapping.
            # Better approach: 
            # 1. Identify indices of dominated points in current front_F.
            # 2. Remove those X and F.
            # 3. Append new X and F.
            
            dominated_indices = []
            for i, f in enumerate(front_F):
                if dominates(fval, f):
                    dominated_indices.append(i)
            
            # Filter out dominated
            new_front_X = [x for i, x in enumerate(front_X) if i not in dominated_indices]
            new_front_F = [f for i, f in enumerate(front_F) if i not in dominated_indices]
            
            new_front_X.append(point)
            new_front_F.append(fval)
            
            return new_front_X, new_front_F

        # DE parameters
        CR = 0.9  # crossover rate
        # Adaptive F: Start high for exploration, decay for exploitation
        F_start = 0.8
        F_end = 0.3
        pop_size = 20  # population size for DE

        # Main optimization loop
        while used < budget:
            if time.time() > deadline:
                break

            # Progress factor for adaptive F
            progress = used / budget
            F_mut = F_start + (F_end - F_start) * progress

            # Build current population from archive
            # If archive is small, use it all. If large, sample.
            if len(archive_F) <= pop_size:
                pop_F = archive_F
                pop_X = archive_X
                n_pop = len(pop_X)
            else:
                # Select pop_size points from archive
                # Strategy: 70% Exploitation (low sum), 30% Exploration (uniform)
                idxs = []
                n_exploit = int(pop_size * 0.7)
                n_explore = pop_size - n_exploit
                
                # Exploitation: Sample with probability proportional to 1/(sum(f/ref)+eps)
                norms = np.array([np.sum(f / ref) for f in archive_F])
                inv_norms = 1.0 / (norms + 1e-10)
                inv_norms /= np.sum(inv_norms)
                
                # Sample without replacement for exploitation
                idx_exploit = rng.choice(len(archive_F), size=n_exploit, replace=False, p=inv_norms)
                idxs.extend(idx_exploit)
                
                # Exploration: Uniform sample from remaining
                remaining_idx = [i for i in range(len(archive_F)) if i not in idx_exploit]
                if len(remaining_idx) >= n_explore:
                    idx_explore = rng.choice(remaining_idx, size=n_explore, replace=False)
                else:
                    # If not enough remaining, sample from all with replacement allowed? 
                    # Better: just sample from all, duplicates are okay for parent selection in DE
                    idx_explore = rng.choice(len(archive_F), size=n_explore, replace=True)
                idxs.extend(idx_explore)
                
                pop_X = [archive_X[i] for i in idxs]
                pop_F = [archive_F[i] for i in idxs]
                n_pop = len(pop_X)

            # Differential evolution step
            if n_pop < 3:
                # Too few points, just do random perturbation
                base = rng.choice(n_pop)
                v = pop_X[base] + F_mut * (hi - lo) * rng.standard_normal(dim)
                v = np.clip(v, lo, hi)
            else:
                # Standard DE/rand/1
                a = rng.randint(0, n_pop)
                while True:
                    b = rng.randint(0, n_pop)
                    if b != a:
                        break
                while True:
                    c = rng.randint(0, n_pop)
                    if c != a and c != b:
                        break
                
                v = pop_X[a] + F_mut * (pop_X[b] - pop_X[c])
                
                # Binomial crossover
                trial = v.copy()
                j_rand = rng.randint(0, dim)
                for j in range(dim):
                    if rng.random() <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop_X[a][j]
                trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Update archive
            # Check if new point is non-dominated
            if is_non_dominated(f_trial, archive_F):
                archive_X, archive_F = update_archive(trial, f_trial, archive_X, archive_F)

        return X
# EVOLVE-BLOCK-END
