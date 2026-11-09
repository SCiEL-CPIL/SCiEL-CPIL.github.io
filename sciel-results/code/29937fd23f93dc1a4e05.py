import numpy as np
import time


class Optimizer:
    """Halton exploration + multi-scale (1+1)-ES with stagnation restarts + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0
        last_improvement_call = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x, last_improvement_call
            if calls_made >= budget:
                return None
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                last_improvement_call = calls_made
            return fx

        def halton_sequence(n, dim, base_start=2):
            points = np.zeros((n, dim))
            for d in range(dim):
                base = base_start + d
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        idx, rem = divmod(idx, base)
                        f /= base
                        r += rem * f
                    points[i, d] = r
            return points

        def nelder_mead(x0, max_evals, seed=42):
            nonlocal best_x, best_f, calls_made
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += 0.25 * max(0.1, abs(x0[i]))

            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                fx = eval_func(simplex[i])
                if fx is None:
                    return
                fvals[i] = fx

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for iteration in range(max_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break

                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                centroid = np.mean(simplex[:-1], axis=0)
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = eval_func(x_reflect)
                if f_reflect is None:
                    break

                if f_reflect < fvals[0]:
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = eval_func(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    if f_reflect < fvals[-1]:
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = eval_func(x_contract)
                    if f_contract is None:
                        break
                    if f_contract < min(f_reflect, fvals[-1]):
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fx = eval_func(simplex[i])
                            if fx is None:
                                return
                            fvals[i] = fx
                        break

                if np.max(fvals) - np.min(fvals) < 1e-12 * (1 + np.abs(fvals[0])):
                    break

            return simplex[0], fvals[0]

        def es_trajectory(x0, sigma, max_evals, seed=42):
            nonlocal best_x, best_f, calls_made, last_improvement_call
            rng = np.random.default_rng(seed)
            n = dim
            x = np.asarray(x0, dtype=np.float64).copy()
            fx = eval_func(x)
            if fx is None:
                return x, fx

            success_count = 0
            total_steps = 0

            for step in range(max_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break

                y = rng.standard_normal(n)
                x_trial = x + sigma * y
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = eval_func(x_trial)
                if f_trial is None:
                    break

                total_steps += 1

                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    sigma *= 1.05
                    success_count += 1
                else:
                    sigma *= 0.95

                if total_steps >= 20:
                    recent_success = success_count / total_steps
                    target_success = 1 / (n + 1)
                    if recent_success > target_success * 1.5:
                        sigma *= 1.1
                    elif recent_success < target_success * 0.5:
                        sigma *= 0.9

                sigma = np.clip(sigma, 1e-6, 5.0)

            return x, fx

        # Phase 1: Halton space-filling exploration
        n_halton = min(60, max(30, budget // 8))
        x0 = np.zeros(dim)
        fx0 = eval_func(x0)
        if fx0 is None:
            return best_x, best_f

        halton_pts = halton_sequence(n_halton - 1, dim)
        for i in range(n_halton - 1):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break
            x = halton_pts[i] * 10.0 - 5.0
            x = np.clip(x, -5.0, 5.0)
            fx = eval_func(x)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: Multi-scale ES with stagnation-based restarts
        remaining = budget - calls_made
        if remaining < 20:
            return best_x, best_f

        polish_budget = min(40, remaining // 6)
        es_budget_total = remaining - polish_budget

        # Multi-scale ES with stagnation detection
        step_sizes = [0.15, 0.5, 1.5, 3.0]
        n_traj = len(step_sizes)
        alloc = np.ones(n_traj) * (es_budget_total // n_traj)
        alloc = np.maximum(alloc, 5)
        total_alloc = alloc.sum()
        if total_alloc > es_budget_total:
            alloc = alloc * (es_budget_total / total_alloc)
            alloc = np.maximum(alloc, 3)

        traj_best_f = np.full(n_traj, float('inf'))
        traj_best_x = [None] * n_traj
        stagnation_threshold = 25  # calls without improvement triggers restart

        for round_idx in range(4):
            if time.time() > deadline:
                break
            if calls_made >= budget - polish_budget - 5:
                break

            # Check for global stagnation
            global_stagnant = (calls_made - last_improvement_call) > stagnation_threshold

            for t in range(n_traj):
                if time.time() > deadline:
                    break
                if calls_made >= budget - polish_budget - 5:
                    break

                sigma = step_sizes[t]

                if global_stagnant and round_idx > 0:
                    # Restart from a new random point when globally stagnant
                    rng_restart = np.random.default_rng(7777 + t + round_idx * 100)
                    x_start = rng_restart.uniform(-4.0, 4.0, size=dim)
                    x_start = np.clip(x_start, -5.0, 5.0)
                    traj_best_x[t] = None
                    traj_best_f[t] = float('inf')
                elif traj_best_x[t] is None:
                    rng_t = np.random.default_rng(1000 + t + round_idx * 10)
                    x_start = best_x + sigma * 0.5 * rng_t.standard_normal(dim)
                    x_start = np.clip(x_start, -5.0, 5.0)
                else:
                    x_start = traj_best_x[t].copy()

                batch_size = max(3, int(alloc[t] // 4))
                x_end, f_end = es_trajectory(x_start, sigma, batch_size, seed=2000 + t + round_idx * 10)

                if f_end < traj_best_f[t]:
                    traj_best_f[t] = f_end
                    traj_best_x[t] = x_end.copy()

                if f_end < best_f:
                    best_f = f_end
                    best_x = x_end.copy()

            # Reallocate based on improvement
            for t in range(n_traj):
                if traj_best_f[t] < float('inf') and traj_best_f[t] < best_f + 0.1:
                    alloc[t] = min(alloc[t] * 1.2, es_budget_total // 2)
                else:
                    alloc[t] = max(alloc[t] * 0.8, 3)

        # Phase 3: Nelder-Mead polish
        if best_x is not None and calls_made < budget - 5:
            remaining_polish = budget - calls_made - 5
            if remaining_polish > 10:
                nelder_mead(best_x.copy(), max_evals=remaining_polish, seed=5000)

        # Phase 4: Fine-grained local search
        if best_x is not None and calls_made < budget - 2:
            remaining_fine = budget - calls_made - 2
            if remaining_fine > 5:
                es_trajectory(best_x.copy(), sigma=0.02, max_evals=remaining_fine, seed=6000)

        return best_x, best_f
