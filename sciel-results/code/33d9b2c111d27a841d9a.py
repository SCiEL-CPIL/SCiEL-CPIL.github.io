import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280.0
        rng = np.random.default_rng(123)
        n = self.dim
        budget = self.budget

        def evaluate(x):
            if self.calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x_clipped.copy()
            return val

        # 1. Initial exploration: diverse set of points
        initial_candidates = []
        initial_candidates.append(np.zeros(n))
        for _ in range(20):
            initial_candidates.append(rng.uniform(-4, 4, n))
        for _ in range(10):
            initial_candidates.append(rng.uniform(0.5, 1.5, n))
            initial_candidates.append(rng.uniform(-1.5, -0.5, n))

        for x in initial_candidates:
            if self.calls >= budget or time.time() > deadline:
                break
            evaluate(x)

        if self.best_x is None:
            self.best_x = np.zeros(n)
            evaluate(self.best_x)
            return self.best_x, self.best_f

        # 2. (1+1)-ES with adaptive restarts, then Nelder-Mead polish
        # State
        x_best = self.best_x.copy()
        f_best = self.best_f
        sigma = 1.0  # initial step size
        success_ratio = 0.3
        target_success_ratio = 1.0 / (n + 1)
        success_count = 0
        total_trials = 0
        no_improve = 0
        max_no_improve = 50
        restart_count = 0
        max_restarts = 10

        # Main (1+1)-ES loop
        while self.calls < budget and time.time() < deadline:
            # Generate candidate
            candidate = x_best + sigma * rng.standard_normal(n)
            candidate = np.clip(candidate, -5, 5)

            f_candidate = evaluate(candidate)
            if f_candidate is None:
                break

            total_trials += 1
            if f_candidate < f_best:
                x_best = candidate
                f_best = f_candidate
                success_count += 1
                no_improve = 0
            else:
                no_improve += 1

            # Adapt sigma based on success ratio (every few iterations)
            if total_trials % 10 == 0:
                current_ratio = success_count / total_trials
                # Update success ratio with exponential moving average
                success_ratio = 0.5 * success_ratio + 0.5 * current_ratio
                # Adapt sigma
                if success_ratio > target_success_ratio:
                    sigma *= 1.2
                else:
                    sigma /= 1.2
                # Reset counters
                success_count = 0
                total_trials = 0

            # Check for stagnation -> restart
            if no_improve >= max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Perturb or jump
                if rng.random() < 0.5:
                    x_best = x_best + rng.normal(0, 2.0, n)
                else:
                    x_best = rng.uniform(-4, 4, n)
                x_best = np.clip(x_best, -5, 5)
                sigma = max(sigma * 0.5, 0.1)
                no_improve = 0

            # Update global best
            if f_best < self.best_f:
                self.best_f = f_best
                self.best_x = x_best.copy()

        # 3. Final polish with Nelder-Mead if budget remains
        remaining = budget - self.calls
        if remaining > 20 and time.time() < deadline:
            local_calls = [0]
            local_best_f = [self.best_f]

            def nm_wrapper(z):
                if self.calls >= budget or time.time() > deadline or local_calls[0] > remaining:
                    return local_best_f[0]
                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                self.calls += 1
                local_calls[0] += 1
                if val < local_best_f[0]:
                    local_best_f[0] = val
                    self.best_f = val
                    self.best_x = z_clipped.copy()
                return val

            try:
                scipy.optimize.minimize(
                    nm_wrapper,
                    self.best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining // 5, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                pass

        return self.best_x, self.best_f
