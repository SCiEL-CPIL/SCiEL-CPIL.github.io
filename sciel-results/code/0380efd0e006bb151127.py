"""Rewritten program for WFG7 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Multi-objective optimizer using a combination of random sampling and
    differential evolution-inspired search with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Use a portion of budget for initial diverse sampling
        n_init = min(30, self.budget // 5)
        X_init = lo + range_ * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        archive_X = list(X_init)
        archive_F = list(F_init)
        used = n_init
        
        def dominates(a, b):
            """Return True if a dominates b (a is better in all objectives and strictly better in at least one)."""
            return np.all(a <= b) and np.any(a < b)
        
        def is_non_dominated(f, archive_F):
            """Check if f is non-dominated by any point in archive_F."""
            for af in archive_F:
                if dominates(af, f):
                    return False
            return True
        
        def update_archive(x, f):
            """Add (x, f) to archive if non-dominated, removing dominated points."""
            non_dom = True
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if dominates(af, f):
                    non_dom = False
                    break
                if dominates(f, af):
                    continue  # f dominates af, so af is removed
                new_archive_X.append(ax)
                new_archive_F.append(af)
            if non_dom:
                new_archive_X.append(x)
                new_archive_F.append(f)
            return new_archive_X, new_archive_F
        
        archive_X, archive_F = update_archive(X_init[0], F_init[0])
        for i in range(1, n_init):
            archive_X, archive_F = update_archive(X_init[i], F_init[i])
        
        # Main loop: use differential evolution-style mutation on archive
        # Keep a population for DE
        pop_size = min(20, self.budget - used)
        if pop_size < 3:
            pop_size = max(3, self.budget - used)
        
        # Initialize population from archive + random
        pop_X = []
        pop_F = []
        for x in archive_X[:min(len(archive_X), pop_size // 2)]:
            pop_X.append(x.copy())
            pop_F.append(problem(x) if x not in [ax for ax in archive_X] else None)
        
        # Simpler approach: maintain a fixed population
        # Reset and use a cleaner DE-like approach
        n_pop = min(15, self.budget - used)
        if n_pop < 3:
            n_pop = 3
        
        # Initialize population
        pop_X = []
        pop_F = []
        for i in range(n_pop):
            if i < len(archive_X):
                x = archive_X[i].copy()
            else:
                x = lo + range_ * rng.random(self.dim)
            f = problem(x)
            pop_X.append(x)
            pop_F.append(f)
            used += 1
        
        # Differential evolution parameters
        CR = 0.9
        F_factor = 0.8
        
        while used < self.budget:
            if len(pop_X) < 3:
                break
            
            # Select three distinct parents
            idx1, idx2, idx3 = rng.choice(len(pop_X), size=3, replace=False)
            v = pop_X[idx1] + F_factor * (pop_X[idx2] - pop_X[idx3])
            
            # Crossover
            trial = pop_X[idx1].copy()
            jrand = rng.integers(self.dim)
            for j in range(self.dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = v[j]
            
            trial = np.clip(trial, lo, hi)
            f_trial = problem(trial)
            used += 1
            
            # Selection: keep the better of trial and current
            if dominates(f_trial, pop_F[idx1]):
                pop_X[idx1] = trial
                pop_F[idx1] = f_trial
                # Update archive
                archive_X, archive_F = update_archive(trial, f_trial)
            elif dominates(pop_F[idx1], f_trial):
                pass
            else:
                # Non-dominated with each other; use crowding distance or just replace randomly
                if rng.random() < 0.5:
                    pop_X[idx1] = trial
                    pop_F[idx1] = f_trial
                    archive_X, archive_F = update_archive(trial, f_trial)
            
            # Also update archive with current pop members occasionally
            if used % 10 == 0 and len(pop_X) > 0:
                for i, (px, pf) in enumerate(zip(pop_X, pop_F)):
                    if is_non_dominated(pf, archive_F):
                        archive_X, archive_F = update_archive(px, pf)
        
        return archive_X
# EVOLVE-BLOCK-END
