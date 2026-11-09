import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            try:
                fx = func(x)
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # Phase 1: Broad scout to find a good starting point
        scout_budget = 20
        scout_points = []
        points = [np.zeros(d)]
        for _ in range(scout_budget - 1):
            points.append(rng.uniform(lo, hi, d))

        for x in points:
            if time.time() - self.start_time > self.deadline:
                break
            fx = safe_call(x)
            scout_points.append((x, fx))

        if best_x is None:
            best_x = np.zeros(d)
            safe_call(best_x)

        # Phase 2: Simulated Annealing with adaptive cooling
        # SA is well-suited for step-like plateau structures as it allows uphill moves
        current_x = best_x.copy()
        current_f = best_f

        # Initial temperature based on function value spread
        if len(scout_points) > 1:
            f_vals = [t[1] for t in scout_points if np.isfinite(t[1])]
            if f_vals:
                T0 = max(np.std(f_vals) * 0.5, 1.0)
            else:
                T0 = 1.0
        else:
            T0 = 1.0

        # Cooling parameters
        alpha = 0.995  # cooling rate
        T_min = 1e-8
        max_iter = budget - call_count

        # Adaptive step size
        step_size = 1.0
        best_step_size = 1.0
        accept_ratio = 0.5
        target_accept = 0.3

        for i in range(max_iter):
            if time.time() - self.start_time > self.deadline:
                break

            # Generate candidate
            delta = rng.normal(0, step_size, d)
            candidate_x = np.clip(current_x + delta, lo, hi)
            candidate_f = safe_call(candidate_x)

            # Acceptance criterion
            if candidate_f < current_f:
                # Always accept improvements
                current_x = candidate_x
                current_f = candidate_f
                accept_ratio = accept_ratio * 0.9 + 1.0 * 0.1
            else:
                # Metropolis acceptance for uphill moves
                delta_f = candidate_f - current_f
                if T0 > T_min:
                    accept_prob = np.exp(-delta_f / T0)
                    if rng.random() < accept_prob:
                        current_x = candidate_x
                        current_f = candidate_f
                        accept_ratio = accept_ratio * 0.9 + 1.0 * 0.1
                accept_ratio = accept_ratio * 0.9 + 0.0 * 0.1

            # Adapt step size based on acceptance ratio
            if accept_ratio > target_accept * 1.5:
                step_size *= 1.2
                best_step_size = step_size
            elif accept_ratio < target_accept * 0.5:
                step_size *= 0.8
            else:
                step_size = best_step_size * 0.95 + step_size * 0.05

            # Cool down temperature
            T0 *= alpha

            # If temperature is very low, do local search around best
            if T0 < T_min * 10 and call_count < budget:
                # Switch to local refinement mode
                local_budget = min(budget - call_count, 50)
                local_step = 0.1
                for _ in range(local_budget):
                    if time.time() - self.start_time > self.deadline:
                        break
                    delta = rng.normal(0, local_step, d)
                    candidate_x = np.clip(best_x + delta, lo, hi)
                    candidate_f = safe_call(candidate_x)
                    if candidate_f < best_f:
                        current_x = candidate_x
                        current_f = candidate_f
                    local_step *= 0.95

        return best_x, best_f
