# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _dominates(a, b):
        """Check if vector a dominates vector b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 15.0  # safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Archive: list of (x, f) tuples for non-dominated points
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            f = np.asarray(f, dtype=float)
            # Check if new point is dominated by any archive point
            for i in range(len(archive_F)):
                if self._dominates(archive_F[i], f):
                    return
            # Remove archive points dominated by new point
            keep = []
            for i in range(len(archive_F)):
                if self._dominates(f, archive_F[i]):
                    continue
                keep.append(i)
            archive_F = [archive_F[i] for i in keep]
            archive_X = [archive_X[i] for i in keep]
            archive_F.append(f)
            archive_X.append(x)

        # Phase 1: Latin Hypercube Sampling for initial coverage
        # Use as many points as we can afford, leaving budget for refinement
        # Let's use about 40% of budget for initial sampling
        n_init = max(20, min(80, budget // 3))
        
        # Generate LHS points
        # For LHS, we divide each dimension into n_init equal intervals
        # and place one point in each interval per dimension, then shuffle
        lxs = np.zeros((n_init, dim))
        for d in range(dim):
            intervals = np.linspace(lo[d], hi[d], n_init + 1)
            # Randomly assign each interval to one row
            perm = rng.permutation(n_init)
            for i in range(n_init):
                row = perm[i]
                # Uniform within the interval
                lxs[row, d] = intervals[i] + rng.random() * (intervals[i + 1] - intervals[i])
        
        # Evaluate initial population
        for i in range(n_init):
            if time.time() >= deadline:
                break
            x = lxs[i]
            f = problem(x)
            update_archive(x, f)

        # Phase 2: Differential Evolution style exploration/exploitation
        # We maintain a small population and generate offspring
        # Budget remaining
        remaining = budget - n_init
        
        # Population size for DE
        pop_size = max(10, min(30, remaining // 4))
        
        # Initialize DE population from archive + random
        de_X = []
        de_F = []
        
        # Add best points from archive
        if archive_F:
            # Sort archive by sum of objectives to get "best" points
            archive_idx = sorted(range(len(archive_F)), key=lambda i: np.sum(archive_F[i]))
            for i in archive_idx[:min(pop_size, len(archive_idx))]:
                de_X.append(archive_X[i].copy())
                de_F.append(archive_F[i].copy())
        
        # Fill rest with random points
        while len(de_X) < pop_size:
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            update_archive(x, f)
            de_X.append(x.copy())
            de_F.append(f.copy())
        
        # DE parameters
        F_factor = 0.8  # Mutation scaling
        CR = 0.9  # Crossover rate
        
        # Main loop
        eval_count = n_init + len(de_X)
        
        while eval_count < budget:
            if time.time() >= deadline:
                break
            
            # Generate offspring
            # Pick 3 distinct parents
            idx = rng.choice(pop_size, size=3, replace=False)
            r1, r2, r3 = de_X[idx[0]], de_X[idx[1]], de_X[idx[2]]
            base = de_X[idx[2]]  # Base vector
            
            # Mutation
            mutant = base + F_factor * (r1 - r2)
            mutant = np.clip(mutant, lo, hi)
            
            # Crossover
            trial = np.zeros(dim)
            jrand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = base[j]
            
            # Evaluate
            f_trial = problem(trial)
            eval_count += 1
            update_archive(trial, f_trial)
            
            # Selection: replace the worst in the DE population
            # Use a simple criterion: sum of objectives
            sums = np.array([np.sum(f) for f in de_F])
            worst_idx = np.argmax(sums)
            
            # If trial is better than worst, replace
            if np.sum(f_trial) < sums[worst_idx]:
                de_X[worst_idx] = trial.copy()
                de_F[worst_idx] = f_trial.copy()
            else:
                # Keep base, but maybe add trial to archive if it's non-dominated
                # (already done in update_archive)
                pass
            
            # Periodically refresh DE population with archive points
            if eval_count % 50 == 0 and archive_F:
                # Reinitialize some DE members from archive
                archive_idx = sorted(range(len(archive_F)), key=lambda i: np.sum(archive_F[i]))
                for i in range(min(5, len(archive_idx))):
                    de_X[i] = archive_X[archive_idx[i]].copy()
                    de_F[i] = archive_F[archive_idx[i]].copy()

        # Phase 3: Local refinement of archive points using Nelder-Mead
        # Pick the best few points from archive and polish them
        if archive_F:
            # Sort by sum of objectives
            archive_idx = sorted(range(len(archive_F)), key=lambda i: np.sum(archive_F[i]))
            # Take top 5-10 points for refinement
            n_refine = min(10, len(archive_idx), max(3, (budget - eval_count) // 4))
            
            for i in range(n_refine):
                if eval_count >= budget:
                    break
                if time.time() >= deadline:
                    break
                
                x0 = archive_X[archive_idx[i]].copy()
                
                # Use scipy's Nelder-Mead on a scalarized objective
                # We use a weighted sum with random weights to explore different trade-offs
                # But since we want to improve the Pareto front, let's use the sum
                # Actually, let's use a few different weight vectors
                
                from scipy.optimize import minimize
                
                # Try a few weight vectors
                weights_list = [
                    np.ones(n_obj) / n_obj,
                    np.array([0.5, 0.25, 0.25]),
                    np.array([0.25, 0.5, 0.25]),
                    np.array([0.25, 0.25, 0.5]),
                ]
                
                for w in weights_list:
                    if eval_count >= budget:
                        break
                    if time.time() >= deadline:
                        break
                    
                    def objective(x):
                        nonlocal eval_count
                        f = problem(x)
                        eval_count += 1
                        return np.dot(w, f)
                    
                    try:
                        result = minimize(objective, x0, method='Nelder-Mead',
                                        bounds=list(zip(lo, hi)),
                                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                        if result.fun < np.inf:
                            x_best = result.x
                            f_best = problem(x_best)
                            eval_count += 1
                            update_archive(x_best, f_best)
                    except:
                        pass

        # Return the archive
        if archive_X:
            return np.array(archive_X)
        return np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END
