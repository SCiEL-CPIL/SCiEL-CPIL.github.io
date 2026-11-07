"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Multi-objective optimizer: hypervolume-guided sequential search with DE refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # Track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            fc = problem(x)
            fc = np.asarray(fc, dtype=float)
            all_X.append(np.asarray(x, dtype=float).copy())
            all_F.append(fc)
            used += 1
            return fc

        # --- Phase 1: Hypervolume-guided sequential search ---
        # Maintain a small archive of non-dominated points.
        # For each new point, try to maximize marginal hypervolume contribution.
        
        # Initialize with a small random sample
        n_init = min(15, budget // 10)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        for x in X_init:
            evaluate(x)

        # Archive: list of (x, f) pairs that are non-dominated
        archive_X = []
        archive_F = []
        
        def is_dominated(f_new, F_archive):
            for f in F_archive:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def update_archive(x_new, f_new):
            """Add point to archive if non-dominated, remove dominated points."""
            if is_dominated(f_new, archive_F):
                return
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                # Check if new point dominates archive point
                if np.all(f_new <= af) and np.any(f_new < af):
                    continue
                new_archive_X.append(ax)
                new_archive_F.append(af)
            new_archive_X.append(x_new)
            new_archive_F.append(f_new)
            return new_archive_X, new_archive_F

        # Build initial archive
        for i in range(len(all_X)):
            new_ax, new_af = update_archive(all_X[i], all_F[i])
            archive_X = new_ax
            archive_F = new_af

        # Hypervolume contribution calculation for 3D
        ref_point = np.array([3.0, 5.0, 7.0])

        def hv_contribution(f_new, F_arch):
            """Compute marginal hypervolume contribution of f_new given archive F_arch."""
            # Simple approximation: check if f_new is non-dominated and estimate contribution
            if is_dominated(f_new, F_arch):
                return 0.0
            
            # For 3 objectives, the contribution can be estimated by checking
            # the volume of the box from f_new to ref_point that isn't covered
            # by existing archive points. This is complex, so we use a simpler heuristic.
            
            # Estimate: the box from f_new to ref_point
            box = ref_point - f_new
            if np.any(box <= 0):
                return 0.0
            
            # Subtract volumes covered by archive points that dominate parts of this box
            # Simplified: just return the box volume as an upper bound
            # A more accurate approach would compute the exact contribution
            return np.prod(box)

        # Sequential search: for each step, try several candidate locations
        # and pick the one with highest estimated hypervolume contribution
        n_hv_steps = min(50, budget - used - 50)  # Reserve budget for DE phase
        
        for step in range(n_hv_steps):
            if used >= budget or time.time() > deadline:
                break
            
            # Generate candidate directions based on current archive
            candidates = []
            
            # Strategy 1: Perturb existing archive points
            for ax in archive_X:
                for scale in [0.05, 0.1, 0.2]:
                    step_vec = rng.standard_normal(dim) * scale * (hi - lo)
                    cand = np.clip(ax + step_vec, lo, hi)
                    candidates.append(cand)
            
            # Strategy 2: Interpolate between archive points
            if len(archive_X) >= 2:
                for _ in range(5):
                    i, j = rng.choice(len(archive_X), size=2, replace=False)
                    t = rng.random()
                    cand = np.clip((1-t)*archive_X[i] + t*archive_X[j], lo, hi)
                    candidates.append(cand)
            
            # Strategy 3: Random points
            for _ in range(5):
                cand = lo + (hi - lo) * rng.random(dim)
                candidates.append(cand)
            
            # Evaluate a subset of candidates and pick the best by HV contribution
            n_eval = min(len(candidates), max(3, (budget - used) // max(1, n_hv_steps - step)))
            if n_eval == 0:
                break
            
            # Shuffle and evaluate
            idx = rng.choice(len(candidates), size=min(n_eval, len(candidates)), replace=False)
            best_cand = None
            best_f = None
            best_hv = -1
            
            for i in idx:
                if used >= budget or time.time() > deadline:
                    break
                cand = candidates[i]
                f = evaluate(cand)
                
                # Compute HV contribution
                hv = hv_contribution(f, archive_F)
                if hv > best_hv:
                    best_hv = hv
                    best_cand = cand
                    best_f = f
            
            # Update archive with best candidate
            if best_cand is not None:
                new_ax, new_af = update_archive(best_cand, best_f)
                archive_X = new_ax
                archive_F = new_af

        # --- Phase 2: Differential Evolution refinement ---
        # Use the archive points as seeds for DE
        remaining_budget = budget - used
        
        if remaining_budget > 10 and time.time() < deadline:
            # Create population from archive + random
            pop_size = min(20, remaining_budget // 3)
            if pop_size < 4:
                pop_size = 4
            
            # Seed from archive
            if len(archive_X) > 0:
                n_seeds = min(pop_size // 2, len(archive_X))
                seed_idx = rng.choice(len(archive_X), size=n_seeds, replace=False)
                X = np.array([archive_X[i].copy() for i in seed_idx])
                F = np.array([archive_F[i].copy() for i in seed_idx])
                
                remaining_pop = pop_size - len(X)
                if remaining_pop > 0:
                    X_rand = lo + (hi - lo) * rng.random((remaining_pop, dim))
                    F_rand = np.array([evaluate(x) for x in X_rand])
                    X = np.vstack([X, X_rand])
                    F = np.vstack([F, F_rand])
            else:
                X = lo + (hi - lo) * rng.random((pop_size, dim))
                F = np.array([evaluate(x) for x in X])
            
            # DE loop
            max_iter = (remaining_budget - pop_size) // pop_size
            for it in range(max_iter):
                if used >= budget or time.time() > deadline:
                    break
                
                for i in range(pop_size):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # DE/rand/1 mutation
                    a, b, c = rng.choice(pop_size, size=3, replace=False)
                    while a == i or b == i or c == i:
                        a, b, c = rng.choice(pop_size, size=3, replace=False)
                    
                    F_factor = 0.5 + 0.5 * rng.random()
                    v = X[a] + F_factor * (X[b] - X[c])
                    v = np.clip(v, lo, hi)
                    
                    # Crossover
                    CR = 0.5
                    j_rand = rng.integers(dim)
                    trial = X[i].copy()
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    
                    ft = evaluate(trial)
                    
                    # Selection: keep if non-dominated and better sum
                    if np.all(ft <= F[i]) and np.any(ft < F[i]):
                        X[i] = trial
                        F[i] = ft
                    elif not is_dominated(ft, list(F)):
                        if np.sum(ft) < np.sum(F[i]):
                            X[i] = trial
                            F[i] = ft

        return all_X
# EVOLVE-BLOCK-END
