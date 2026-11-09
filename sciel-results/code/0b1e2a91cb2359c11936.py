"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Powell's method with coordinate-wise line search for f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = np.zeros(dim)
        best_f = func(best_x)
        call_count = 1

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            call_count += 1
            return f

        def line_search(x, direction, f0):
            """Golden section / parabolic line search along direction from x."""
            if call_count >= budget:
                return x, f0
            # Try several step sizes to find a good step
            base_steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            best_step = 0.0
            best_f_ls = f0
            for s in base_steps:
                if call_count >= budget:
                    break
                for sign in [1.0, -1.0]:
                    x_try = np.clip(x + sign * s * direction, -5, 5)
                    f_try = safe_call(x_try)
                    if f_try is None:
                        break
                    if f_try < best_f_ls:
                        best_f_ls = f_try
                        best_step = sign * s
                if call_count >= budget:
                    break
            return x + best_step * direction, best_f_ls

        # Phase 1: Coordinate-wise refinement (Powell's method core)
        # Initialize directions to unit vectors
        directions = [np.eye(dim)[i] for i in range(dim)]

        max_rounds = 30  # Limit rounds to stay within budget
        for round_idx in range(max_rounds):
            if time.time() > deadline or call_count >= budget:
                break

            x = best_x.copy()
            f = best_f
            min_delta = float('inf')

            for i in range(dim):
                if time.time() > deadline or call_count >= budget:
                    break
                x_new, f_new = line_search(x, directions[i], f)
                delta = f - f_new
                if delta > min_delta:
                    min_delta = delta
                x, f = x_new, f_new

            if f < best_f:
                best_x = x.copy()
                best_f = f

            # Check convergence
            if min_delta < 1e-12:
                break

            # Powell's direction update: add new direction, drop worst
            new_dir = x - best_x  # Direction of improvement in this round
            if np.linalg.norm(new_dir) > 1e-15:
                # Find direction with smallest improvement and replace
                # For simplicity, replace the last direction
                directions[-1] = new_dir / np.linalg.norm(new_dir)

        # Phase 2: Fine-tuning with smaller steps near the optimum
        if time.time() < deadline and call_count < budget:
            # Do a few more coordinate passes with smaller base steps
            for _ in range(5):
                if time.time() > deadline or call_count >= budget:
                    break
                x = best_x.copy()
                f = best_f
                for i in range(dim):
                    if time.time() > deadline or call_count >= budget:
                        break
                    # Smaller steps for fine-tuning
                    for s in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                        if call_count >= budget:
                            break
                        for sign in [1.0, -1.0]:
                            x_try = np.clip(x + sign * s, -5, 5)
                            # Only modify coordinate i
                            x_try = x.copy()
                            x_try[i] = np.clip(x[i] + sign * s, -5, 5)
                            f_try = safe_call(x_try)
                            if f_try is None:
                                break
                            if f_try < f:
                                x, f = x_try, f_try
                    if call_count >= budget:
                        break
                if f < best_f:
                    best_x = x.copy()
                    best_f = f

        # Phase 3: Final Nelder-Mead polish if budget remains
        if time.time() < deadline and call_count < budget - 50:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    lambda z: func(np.clip(z, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14},
                )
                if res.fun < best_f:
                    best_x = np.clip(res.x, -5, 5)
                    best_f = res.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
