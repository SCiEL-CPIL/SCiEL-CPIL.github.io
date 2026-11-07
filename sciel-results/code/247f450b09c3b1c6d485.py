"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using Multi-Objective Differential Evolution (MODE).
    Uses Latin Hypercube Sampling for initialization, DE/rand/1 mutation, 
    binomial crossover, and a non-dominated archive for selection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        pop_size = 20
        F = 0.8  # Mutation factor
        CR = 0.5 # Crossover rate
        
        # Ensure we don't exceed budget
        if self.budget < pop_size:
            pop_size = self.budget
            if pop_size < 3:
                # Fallback to simple random search if budget is too small for DE
                X = lo + (hi - lo) * rng.random((pop_size, self.dim))
                for x in X:
                    problem(x)
                return X

        # 1. Initialize Population using Latin Hypercube Sampling
        # LHS ensures better coverage than uniform random
        n = pop_size
        X = np.zeros((n, self.dim))
        for d in range(self.dim):
            # Divide [0,1] into n intervals
            intervals = np.linspace(0, 1, n + 1)
            # Random point in each interval
            points = rng.random(n) * (intervals[1] - intervals[0]) + intervals[:n]
            # Shuffle to decorrelate dimensions
            rng.shuffle(points)
            # Scale to [lo, hi]
            X[:, d] = lo[d] + points * (hi[d] - lo[d])
            
        F_pop = np.array([problem(x) for x in X])
        used = n

        # Archive: list of (x, f) tuples for non-dominated solutions
        archive = []
        for i in range(n):
            if self._is_non_dominated(F_pop[i], [f for _, f in archive]):
                archive.append((X[i].copy(), F_pop[i].copy()))

        def _is_non_dominated(f, archive_f):
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return False
            return True

        def _dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Main Loop
        while used < self.budget:
            # We can evaluate one or more individuals per iteration
            # Let's evaluate one individual per loop to stay within budget easily
            i = rng.integers(0, n)
            
            # DE/rand/1 mutation
            # Pick 3 distinct random indices different from i
            indices = [j for j in range(n) if j != i]
            r1, r2, r3 = rng.choice(indices, size=3, replace=False)
            
            v = X[r1] + F * (X[r2] - X[r3])
            
            # Binomial Crossover
            trial = X[i].copy()
            j_rand = rng.integers(0, self.dim)
            for j in range(self.dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Selection
            f_target = F_pop[i]
            
            if _dominates(f_trial, f_target):
                # Trial dominates target
                X[i] = trial
                F_pop[i] = f_trial
                # Update archive
                # Remove any archive members dominated by trial
                new_archive = []
                for ax, af in archive:
                    if not _dominates(f_trial, af):
                        new_archive.append((ax, af))
                # Add trial if non-dominated with respect to new archive
                if self._is_non_dominated(f_trial, [af for _, af in new_archive]):
                    new_archive.append((trial.copy(), f_trial.copy()))
                archive = new_archive
                
            elif _dominates(f_target, f_trial):
                # Target dominates trial, keep target
                pass
            else:
                # Non-dominated
                # Replace with probability 0.5 to maintain diversity
                if rng.random() < 0.5:
                    X[i] = trial
                    F_pop[i] = f_trial
                    # Update archive
                    new_archive = []
                    for ax, af in archive:
                        if not _dominates(f_trial, af):
                            new_archive.append((ax, af))
                    if self._is_non_dominated(f_trial, [af for _, af in new_archive]):
                        new_archive.append((trial.copy(), f_trial.copy()))
                    archive = new_archive

        return X
# EVOLVE-BLOCK-END
