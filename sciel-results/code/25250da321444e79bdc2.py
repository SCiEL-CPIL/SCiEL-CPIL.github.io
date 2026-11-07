"""Direct search with golden-section line-search and restarts for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Direct-search method using a fixed direction set and per-direction
    golden-section minimisation with adaptive step scaling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        calls = [0]

        def call_func(x):
            calls[0] += 1
            return float(func(x))

        # Build direction set: coordinate axes, all-ones, and all pair diagonals
        dirs = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            dirs.append(e)
        dirs.append(np.ones(dim) / np.sqrt(dim))
        for i in range(dim):
            for j in range(i + 1, dim):
                d = np.zeros(dim)
                d[i] = 1.0 / np.sqrt(2.0)
                d[j] = 1.0 / np.sqrt(2.0)
                dirs.append(d)
        # Add negative counterparts
        dirs = dirs + [-d for d in dirs]
        dirs = np.array(dirs)
        n_dirs = len(dirs)

        # Golden ratio constants
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        inv_phi = 1.0 / phi
        inv_phi2 = inv_phi ** 2

        # Parameters
        sigma_init = 1.5
        sigma_min = 1e-7
        sigma_max = 8.0
        growth = 1.6
        shrink = 0.6
        gs_iters = 4  # golden-section evaluations per direction when expanding
        n_explore = min(40, budget // 6)

        # Track best
        best_x = np.zeros(dim)
        best_fx = call_func(best_x)
        x = best_x.copy()
        fx = best_fx

        # Exploration: spread out to find a better starting point
        for _ in range(n_explore):
            if calls[0] >= budget:
                break
            x_try = rng.uniform(-4.5, 4.5, dim)
            fx_try = call_func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()
                x = x_try.copy()
                fx = fx_try

        # Direct search with golden-section line search
        sigma = sigma_init
        stall = 0

        while calls[0] < budget:
            cycle_improved = False

            for di in range(n_dirs):
                if calls[0] >= budget:
                    break
                d = dirs[di]

                # Quick probe at current sigma
                y = x + sigma * d
                if np.any(y < -5) or np.any(y > 5):
                    # Skip directions that go out of bounds at this step
                    y_neg = x - sigma * d
                    if np.any(y_neg < -5) or np.any(y_neg > 5):
                        continue
                    y = y_neg
                    neg_dir = True
                else:
                    neg_dir = False

                fy = call_func(y)
                if fy < fx:
                    # Improvement found; refine with golden section
                    fx_old = fx
                    if neg_dir:
                        d_use = -d
                    else:
                        d_use = d

                    # Golden-section search along d_use starting from x
                    # Current best at t=0 (value fx) and t=sigma (value fy)
                    # We search in [0, sigma * 2] to allow expansion
                    t_lo = 0.0
                    t_hi = 2.0 * sigma

                    # Evaluate interior points
                    t1 = t_lo + inv_phi2 * (t_hi - t_lo)
                    t2 = t_lo + inv_phi * (t_hi - t_lo)

                    # We already have f at t=sigma. Let's use that.
                    # Actually let's just do a small golden section in [0, 2*sigma]
                    # We have f(0)=fx, f(sigma)=fy
                    # Need f(t1) and f(t2) where t1 < sigma < t2 (approximately)
                    # t1 = 0.382*2*sigma = 0.764*sigma, t2 = 0.618*2*sigma = 1.236*sigma

                    if calls[0] < budget:
                        x1 = np.clip(x + t1 * d_use, -5, 5)
                        f1 = call_func(x1)
                    else:
                        f1 = fy  # fallback

                    if calls[0] < budget:
                        x2 = np.clip(x + t2 * d_use, -5, 5)
                        f2 = call_func(x2)
                    else:
                        f2 = fy

                    # Now we have f at 0, t1, sigma, t2, 2*sigma (partial)
                    # Pick best
                    candidates_t = [0.0, t1, sigma, t2]
                    candidates_f = [fx, f1, fy, f2]
                    idx_best = int(np.argmin(candidates_f))
                    t_best = candidates_t[idx_best]
                    f_best = candidates_f[idx_best]

                    # If best is at 0, no real improvement
                    if t_best == 0.0:
                        x_new = x.copy()
                        fx_new = fx
                    else:
                        x_new = np.clip(x + t_best * d_use, -5, 5)
                        fx_new = f_best

                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        cycle_improved = True
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()

                    # Expand sigma if we moved significantly
                    if fx_new < fx_old - 1e-10:
                        sigma = min(sigma * growth, sigma_max)
                    else:
                        sigma = max(sigma * shrink, sigma_min)
                    stall = 0
                    continue  # don't evaluate negative direction if positive worked

                # Try negative direction
                if calls[0] >= budget:
                    break
                y_neg = x - sigma * d
                if np.any(y_neg < -5) or np.any(y_neg > 5):
                    continue
                fy_neg = call_func(y_neg)
                if fy_neg < fx:
                    fx_old = fx
                    d_use = -d
                    t_lo = 0.0
                    t_hi = 2.0 * sigma
                    t1 = t_lo + inv_phi2 * (t_hi - t_lo)
                    t2 = t_lo + inv_phi * (t_hi - t_lo)

                    if calls[0] < budget:
                        x1 = np.clip(x + t1 * d_use, -5, 5)
                        f1 = call_func(x1)
                    else:
                        f1 = fy_neg

                    if calls[0] < budget:
                        x2 = np.clip(x + t2 * d_use, -5, 5)
                        f2 = call_func(x2)
                    else:
                        f2 = fy_neg

                    candidates_t = [0.0, t1, sigma, t2]
                    candidates_f = [fx, f1, fy_neg, f2]
                    idx_best = int(np.argmin(candidates_f))
                    t_best = candidates_t[idx_best]
                    f_best = candidates_f[idx_best]

                    if t_best == 0.0:
                        x_new = x.copy()
                        fx_new = fx
                    else:
                        x_new = np.clip(x + t_best * d_use, -5, 5)
                        fx_new = f_best

                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        cycle_improved = True
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()

                    if fx_new < fx_old - 1e-10:
                        sigma = min(sigma * growth, sigma_max)
                    else:
                        sigma = max(sigma * shrink, sigma_min)
                    stall = 0
                    continue

            if not cycle_improved:
                stall += 1
                sigma = max(sigma * shrink, sigma_min)
                if stall >= 3:
                    # Restart from best or random
                    if calls[0] < budget:
                        if rng.random() < 0.5:
                            x = best_x.copy()
                            fx = best_fx
                        else:
                            x_try = rng.uniform(-4.5, 4.5, dim)
                            fx_try = call_func(x_try)
                            if fx_try < best_fx:
                                best_fx = fx_try
                                best_x = x_try.copy()
                                x = x_try.copy()
                                fx = fx_try
                    sigma = sigma_init
                    stall = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
