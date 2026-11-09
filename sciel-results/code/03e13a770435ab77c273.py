"""Optimized black-box optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage derivative-free optimizer for rotated/shifted f19 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nm_simplex(x0, f0, scale, tight=False):
            """Build simplex around x0 and return (simplex, simplex_f)."""
            nonlocal calls_made
            if time.time() > deadline or calls_made >= budget - 2:
                return None, None
            simplex = [x0.copy()]
            simplex_f = [f0]
            rng = np.random.default_rng(777 + calls_made)
            for i in range(dim):
                direction = rng.standard_normal(dim)
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.zeros(dim)
                    direction[i] = 1.0
                    norm = 1.0
                direction /= norm
                new_x = np.clip(x0 + scale * direction, -5.0, 5.0)
                if np.allclose(new_x, x0):
                    new_x[i] += 0.001 if new_x[i] < 0 else -0.001
                    new_x = np.clip(new_x, -5.0, 5.0)
                f_new = safe_call(new_x)
                simplex.append(new_x)
                simplex_f.append(f_new)
                if time.time() > deadline or calls_made >= budget - 2:
                    return None, None
            return simplex, simplex_f

        def nelder_mead(x0, f0, scale, tight=False):
            """Nelder-Mead from x0 with given simplex scale."""
            nonlocal calls_made, best_x, best_f
            simplex, simplex_f = nm_simplex(x0, f0, scale, tight)
            if simplex is None:
                return

            alpha, gamma, sigma = 1.0, 2.0, 0.5
            tol_f = 1e-10 if tight else 1e-8
            tol_x = 0.001 if tight else 0.05
            max_iter = 300 if tight else 120

            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget - 2:
                    break

                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]

                f_spread = abs(simplex_f[0] - simplex_f[-1])
                x_spread = np.max(np.linalg.norm(np.array(simplex) - np.array(simplex[0]), axis=1))
                if f_spread < tol_f * (1 + abs(simplex_f[0])) and x_spread < tol_x:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
                f_ref = safe_call(x_ref)

                if f_ref < simplex_f[0]:
                    x_exp = np.clip(centroid + gamma * (x_ref - centroid), -5.0, 5.0)
                    f_exp = safe_call(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    if f_ref < simplex_f[-1]:
                        x_con = np.clip(centroid + 0.5 * (x_ref - centroid), -5.0, 5.0)
                    else:
                        x_con = np.clip(centroid - 0.5 * (x_ref - centroid), -5.0, 5.0)
                    f_con = safe_call(x_con)
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5.0, 5.0)
                            simplex_f[i] = safe_call(simplex[i])

        def run_nm_chain(x_start, f_start):
            """Run Nelder-Mead with decreasing scales from x_start."""
            nonlocal best_x, best_f
            for scale in [0.3, 0.15, 0.08, 0.04, 0.02, 0.01]:
                if time.time() > deadline or calls_made >= budget - 5:
                    break
                nelder_mead(best_x.copy(), best_f, scale, tight=False)

        # --- Phase 1: Structured exploration ---
        rng = np.random.default_rng(2024)

        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        best_x = x0.copy()
        best_f = f0

        # Axis probes
        for i in range(dim):
            for sign in [1, -1]:
                if time.time() > deadline or calls_made >= budget - 20:
                    break
                x = np.zeros(dim)
                x[i] = sign * 4.5
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        # LHS for broader coverage
        n_lhs = int(budget * 0.30) - calls_made
        if n_lhs > 0 and time.time() < deadline:
            lhs = np.zeros((n_lhs, dim))
            for d in range(dim):
                perm = rng.permutation(n_lhs)
                lhs[:, d] = (perm + rng.random(n_lhs)) / n_lhs
                lhs[:, d] = lhs[:, d] * 10.0 - 5.0
            for x in lhs:
                if time.time() > deadline or calls_made >= budget - 30:
                    break
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        # --- Phase 2: Initial Nelder-Mead refinement ---
        if time.time() < deadline and calls_made < budget - 10:
            run_nm_chain(best_x.copy(), best_f)

        # --- Phase 3: Multi-start from top candidates ---
        # Generate perturbed starts around current best with decreasing radii
        n_starts = 10
        for i in range(n_starts):
            if time.time() > deadline or calls_made >= budget - 15:
                break
            radius = 1.5 / (i + 1)
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            x_new = np.clip(best_x + radius * direction, -5.0, 5.0)
            f_new = safe_call(x_new)
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                if time.time() < deadline and calls_made < budget - 10:
                    run_nm_chain(best_x.copy(), best_f)

        # --- Phase 4: Final high-precision polish ---
        if time.time() < deadline and calls_made < budget - 5:
            nelder_mead(best_x.copy(), best_f, 0.01, tight=True)
            if time.time() < deadline and calls_made < budget - 3:
                nelder_mead(best_x.copy(), best_f, 0.005, tight=True)

        return best_x, best_f
# EVOLVE-BLOCK-END
