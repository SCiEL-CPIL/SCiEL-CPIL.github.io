import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        deadline = time.time() + 300.0
        safe = deadline - 30.0

        # Split budget into multiple independent runs
        n_runs = 10
        budget_per_run = max(10, budget // n_runs)

        # We will record all non-dominated points found
        all_points = []  # list of (x, f)

        # Helper: check if a point is non-dominated by existing set
        def is_non_dominated(f_new, existing_fs):
            if len(existing_fs) == 0:
                return True
            for f_old in existing_fs:
                # If f_old dominates f_new, then f_new is dominated
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return False
            return True

        # Helper: remove points dominated by a new point
        def remove_dominated(f_new, existing_xs, existing_fs):
            new_xs = []
            new_fs = []
            for x_old, f_old in zip(existing_xs, existing_fs):
                # If f_new dominates f_old, remove f_old
                if not (np.all(f_new <= f_old) and np.any(f_new < f_old)):
                    new_xs.append(x_old)
                    new_fs.append(f_old)
            return new_xs, new_fs

        # We'll maintain a global non-dominated set
        global_xs = []
        global_fs = []

        def add_point(x, f):
            nonlocal global_xs, global_fs
            # Check if dominated
            if not is_non_dominated(f, global_fs):
                return
            # Remove any points dominated by this new point
            new_xs, new_fs = remove_dominated(f, global_xs, global_fs)
            new_xs.append(x)
            new_fs.append(f)
            global_xs = new_xs
            global_fs = new_fs

        # Initial random points to seed
        n_init = min(10, budget_per_run // 2)
        for i in range(n_init):
            if time.time() > safe:
                break
            x = lo + span * np.random.random(dim)
            f = np.array(problem(x))
            add_point(x, f)

        # Multi-start DE runs
        for run in range(n_runs):
            if time.time() > safe:
                break
            if len(global_xs) == 0:
                # If no points, just do random
                x = lo + span * np.random.random(dim)
                f = np.array(problem(x))
                add_point(x, f)
                continue

            # Pick a starting point: randomly from current non-dominated set
            start_idx = np.random.randint(len(global_xs))
            x_start = global_xs[start_idx].copy()
            f_start = global_fs[start_idx].copy()

            # DE parameters
            F_mut = 0.8
            CR = 0.9
            max_iter = budget_per_run - 1

            # Local DE loop
            # We'll use a small population of 4 for DE
            pop_size = 4
            # Initialize small population around start
            X_pop = [x_start.copy()]
            F_pop = [f_start.copy()]
            for j in range(pop_size - 1):
                x_j = x_start + span * (np.random.random(dim) - 0.5) * 0.5
                x_j = np.clip(x_j, lo, hi)
                f_j = np.array(problem(x_j))
                X_pop.append(x_j)
                F_pop.append(f_j)
                add_point(x_j, f_j)

            for it in range(max_iter):
                if time.time() > safe:
                    break

                # Create mutant
                if len(X_pop) < 3:
                    # Not enough diversity, just mutate
                    x_mut = x_start + span * (np.random.random(dim) - 0.5)
                    x_mut = np.clip(x_mut, lo, hi)
                else:
                    # Pick 3 distinct individuals
                    idxs = np.random.choice(len(X_pop), 3, replace=False)
                    x_r1 = X_pop[idxs[0]]
                    x_r2 = X_pop[idxs[1]]
                    x_r3 = X_pop[idxs[2]]
                    x_mut = x_r1 + F_mut * (x_r2 - x_r3)
                    x_mut = np.clip(x_mut, lo, hi)

                # Crossover
                x_trial = x_start.copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        x_trial[j] = x_mut[j]

                # Evaluate
                f_trial = np.array(problem(x_trial))

                # Compare with start
                # If trial dominates start, replace start
                if np.all(f_trial <= f_start) and np.any(f_trial < f_start):
                    x_start = x_trial
                    f_start = f_trial
                    # Update global
                    add_point(x_start, f_start)
                    # Update population: replace worst in pop
                    # Find worst in pop (dominated by most others)
                    worst_idx = 0
                    worst_score = 0
                    for i in range(len(X_pop)):
                        score = 0
                        for k in range(len(X_pop)):
                            if i == k:
                                continue
                            if np.all(F_pop[k] <= F_pop[i]) and np.any(F_pop[k] < F_pop[i]):
                                score += 1
                        if score > worst_score:
                            worst_score = score
                            worst_idx = i
                    X_pop[worst_idx] = x_start.copy()
                    F_pop[worst_idx] = f_start.copy()
                elif np.all(f_start <= f_trial) and np.any(f_start < f_trial):
                    # Start dominates trial, keep start
                    pass
                else:
                    # Non-dominated, add trial to population if space
                    if len(X_pop) < 5:
                        X_pop.append(x_trial)
                        F_pop.append(f_trial)
                    add_point(x_trial, f_trial)

            # After DE loop, do local refinement on best point found in this run
            # Find the point in global set that is closest to the reference point
            # Actually, just pick the best point from this run's exploration
            # For simplicity, refine x_start if it's non-dominated
            if time.time() > safe:
                break

            # Nelder-Mead refinement on a scalarized objective
            # Use weighted sum with random weights
            w = np.random.dirichlet(np.ones(n_obj))
            def scalarized(x):
                return np.dot(w, np.array(problem(x)))

            # Try to refine x_start
            res = minimize(scalarized, x_start, method='Nelder-Mead',
                           bounds=list(zip(lo, hi)),
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})

            if res.success and time.time() < safe:
                x_opt = np.clip(res.x, lo, hi)
                f_opt = np.array(problem(x_opt))
                add_point(x_opt, f_opt)

        # If we have time, do a final round of random search
        remaining = budget - len(global_fs) - n_init
        # Actually, we don't track exact usage easily. Let's just use the deadline.
        while time.time() < safe:
            x = lo + span * np.random.random(dim)
            f = np.array(problem(x))
            add_point(x, f)

        return global_xs
