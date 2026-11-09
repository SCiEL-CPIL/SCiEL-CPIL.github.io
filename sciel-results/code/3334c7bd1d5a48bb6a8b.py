import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0  # 5s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        # 1. Scout Phase: Find a good starting point
        scout_budget = min(30, max(20, budget // 10))

        best_x = None
        best_f = np.inf
        scout_points = []

        # Start with center
        x0 = np.zeros(d)
        fx0 = func(x0)
        scout_points.append((x0, fx0))
        if fx0 < best_f:
            best_f = fx0
            best_x = x0.copy()

        for _ in range(scout_budget - 1):
            x = rng.uniform(lo, hi, d)
            fx = func(x)
            scout_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            scout_budget = 1

        # 2. Simulated Annealing Phase
        # Initialize temperature based on scout results
        scout_fs = np.array([p[1] for p in scout_points])
        f_range = np.max(scout_fs) - np.min(scout_fs)
        if f_range < 1e-8:
            f_range = 1.0
        T = f_range * 0.5  # Initial temperature
        T_min = 1e-10
        cooling_rate = 0.97  # Faster cooling for step ellipsoidal

        current_x = best_x.copy()
        current_f = best_f
        best_x_sa = best_x.copy()
        best_f_sa = best_f

        remaining_budget = budget - scout_budget
        step_size = 1.0  # Initial step size

        # Adaptive step size based on scout spread
        if len(scout_points) > 2:
            scout_xs = np.array([p[0] for p in scout_points])
            scout_fs_arr = np.array([p[1] for p in scout_points])
            good_mask = scout_fs_arr < np.percentile(scout_fs_arr, 50)
            if np.sum(good_mask) >= 2:
                good_xs = scout_xs[good_mask]
                mean_good = np.mean(good_xs, axis=0)
                dists = np.linalg.norm(good_xs - mean_good, axis=1)
                step_size = max(0.5, np.mean(dists) * 0.8)
            else:
                step_size = 1.0

        for i in range(remaining_budget):
            if time.time() - self.start_time > self.deadline:
                break

            # Generate candidate
            # Occasionally take a larger jump to escape plateaus
            if rng.random() < 0.1:
                jump_scale = 2.0
            else:
                jump_scale = 1.0
            
            delta = rng.standard_normal(d) * step_size * jump_scale
            candidate_x = current_x + delta
            candidate_x = np.clip(candidate_x, lo, hi)
            
            candidate_f = func(candidate_x)
            
            # Acceptance criterion
            if candidate_f < current_f:
                # Always accept better
                current_x = candidate_x
                current_f = candidate_f
                if current_f < best_f_sa:
                    best_f_sa = current_f
                    best_x_sa = current_x.copy()
            else:
                # Accept worse with probability exp(-delta_f / T)
                delta_f = candidate_f - current_f
                if delta_f < 1e-10:
                    # Plateau - accept to explore
                    current_x = candidate_x
                    current_f = candidate_f
                else:
                    prob = np.exp(-delta_f / T)
                    if rng.random() < prob:
                        current_x = candidate_x
                        current_f = candidate_f
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # Adaptive step size: shrink if stuck, expand if moving well
            if i % 50 == 0 and i > 0:
                # Check if we've made progress recently
                if current_f < best_f_sa + 1e-8:
                    # Stuck, try larger steps
                    step_size *= 1.2
                else:
                    # Making progress, tighten steps
                    step_size *= 0.95
                
                step_size = np.clip(step_size, 0.1, 5.0)

        return best_x_sa, best_f_sa
