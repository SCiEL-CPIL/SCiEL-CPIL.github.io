import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=np.float64)
            f = func(x)
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        n = self.dim

        # Phase 0: Multi-point initialization for better early coverage
        n_init = min(5, self.budget // 20)
        init_candidates = []
        for i in range(n_init):
            try:
                if i == 0:
                    x_c = rng.uniform(-0.5, 0.5, n)
                else:
                    scale = 1.5 + 0.5 * i
                    x_c = rng.uniform(-scale, scale, n)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = safe_func(x_c)
                init_candidates.append((f_c, x_c))
            except (TimeoutError, StopIteration):
                break

        if init_candidates:
            init_candidates.sort(key=lambda t: t[0])
            x = init_candidates[0][1].copy()
            best_local = init_candidates[0][0]
            best_local_x = x.copy()
        else:
            x = np.zeros(n)
            try:
                best_local = safe_func(x)
                best_local_x = x.copy()
            except (TimeoutError, StopIteration):
                return best_x, best_f

        # Phase 1: (1+1)-ES with window-based success-rate sigma adaptation
        global_budget = int(self.budget * 0.75)

        sigma = 2.0
        sigma_min = 1e-4
        sigma_max = 4.0
        window_size = 20
        success_window = []
        stall_count = 0
        stall_limit = 40
        restart_count = 0
        max_restarts = 8

        while calls < global_budget and time.time() < deadline:
            # Generate candidate
            direction = rng.standard_normal(n)
            candidate = x + sigma * direction
            candidate = np.clip(candidate, -5.0, 5.0)

            try:
                f_candidate = safe_func(candidate)
            except (TimeoutError, StopIteration):
                break

            if f_candidate < best_local - 1e-12:
                # Success: move to candidate
                best_local = f_candidate
                best_local_x = candidate.copy()
                x = candidate.copy()
                success_window.append(1)
                stall_count = 0
            else:
                # Failure: stay
                success_window.append(0)
                stall_count += 1

            # Trim window
            if len(success_window) > window_size:
                success_window.pop(0)

            # Adapt sigma based on success rate in window
            if len(success_window) >= window_size // 2:
                sr = sum(success_window) / len(success_window)
                # Target success rate ~ 0.2-0.3
                if sr > 0.3:
                    sigma *= 1.15
                elif sr < 0.15:
                    sigma *= 0.85

            sigma = max(sigma_min, min(sigma, sigma_max))

            # If stalled too long, restart from a diverse point
            if stall_count >= stall_limit:
                if restart_count < max_restarts and calls < global_budget and time.time() < deadline:
                    # Jump significantly from current best
                    jump_scale = max(sigma * 3.0, 1.5)
                    x_new = np.clip(x + jump_scale * rng.standard_normal(n), -5.0, 5.0)
                    try:
                        fx_new = safe_func(x_new)
                        if fx_new < best_local:
                            best_local = fx_new
                            best_local_x = x_new.copy()
                            x = x_new.copy()
                        else:
                            # Try a smaller perturbation
                            if calls < global_budget and time.time() < deadline:
                                x_new2 = np.clip(x + sigma * 1.5 * rng.standard_normal(n), -5.0, 5.0)
                                fx_new2 = safe_func(x_new2)
                                if fx_new2 < best_local:
                                    best_local = fx_new2
                                    best_local_x = x_new2.copy()
                                    x = x_new2.copy()
                    except (TimeoutError, StopIteration):
                        break
                    sigma = max(sigma * 0.7, sigma_min)
                    stall_count = 0
                    restart_count += 1

        # Phase 2: Local refinement with L-BFGS-B on the global best
        remaining = self.budget - calls
        if remaining > 10 and best_x is not None and time.time() < deadline:
            try:
                result = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': remaining,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass

        # Phase 3: Coordinate-descent polish with remaining budget
        remaining = self.budget - calls
        if remaining > 0 and best_x is not None and time.time() < deadline:
            step = max(sigma, 0.02)
            max_cd_rounds = min(5, remaining // (2 * n))

            for _round in range(max_cd_rounds):
                if time.time() > deadline or calls >= self.budget:
                    break
                moved_any = False
                for d in range(n):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    best_trial = None
                    best_trial_f = best_f
                    for sign in (1.0, -1.0):
                        if calls >= self.budget:
                            break
                        trial = best_x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                        try:
                            ftrial = safe_func(trial)
                            if ftrial < best_trial_f:
                                best_trial_f = ftrial
                                best_trial = trial.copy()
                        except (TimeoutError, StopIteration):
                            break
                    if best_trial is not None:
                        best_x = best_trial
                        best_f = best_trial_f
                        moved_any = True

                if not moved_any:
                    step *= 0.5
                    if step < 1e-5:
                        break

        return best_x, best_f
