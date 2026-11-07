"""CMA-ES with model-based refinement for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Gaussian process / quadratic model refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: CMA-ES ---
        # Initial population
        lambda_ = max(4 + int(3 * np.log(dim)), dim + 1)  # ~15 for dim=5
        mu = lambda_ // 2
        # Weights
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        sigma0 = 2.0  # initial step size
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        norm_c = 2 + 2 / (dim + 1)
        norm_s = np.sqrt((1 + 2 / (dim + 1)) * (1 - cc) * (mueff + 2) / (dim + mueff + 5))

        # Initial evaluations
        rng = np.random.default_rng(42)
        pop_x = []
        pop_f = []
        for i in range(lambda_):
            xi = x_mean + sigma0 * rng.standard_normal(dim)
            fi = evaluate(xi)
            pop_x.append(xi)
            pop_f.append(fi)
            if not time_ok():
                break

        # CMA-ES main loop
        max_cma_iters = 80
        for it in range(max_cma_iters):
            if not time_ok():
                break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = np.array(pop_x)[order]
            pop_f = np.array(pop_f)[order]

            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum(w[:, None] * pop_x[:mu], axis=0)

            # Update evolution paths
            C_inv_sqrt = np.linalg.cholesky(C)  # C = L L^T
            C_inv_sqrt = np.linalg.inv(C_inv_sqrt)  # C^(-1/2)
            y = (x_mean - old_mean) / sigma0
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * C_inv_sqrt @ y
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y

            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) / norm_s < (1.4 + 2 / (dim + 1))
            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y

            # Update covariance
            delta_hs = (1 - hsig) * (1 - cc)
            C = (1 + c1 * delta_hs - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + (delta_hs - 1 + cc * cc) * np.eye(dim)) \
                + cmu * np.sum(w[:, None, None] * np.outer(pop_x[:mu] - old_mean, pop_x[:mu] - old_mean) / sigma0**2, axis=0)

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update sigma (CSA)
            sigma0 *= np.exp(cs / norm_s * (np.linalg.norm(ps) / norm_s - 1))
            sigma0 = np.clip(sigma0, 1e-12, 1e3)

            # Generate new population
            new_pop_x = []
            new_pop_f = []
            for i in range(lambda_):
                if not time_ok():
                    break
                B, d = np.linalg.eigh(C)
                B = B[:, np.argsort(d)[::-1]]
                d = np.sqrt(np.clip(np.sort(d)[::-1], 0, None))
                zi = rng.standard_normal(dim)
                xi = x_mean + sigma0 * B @ (d * zi)
                fi = evaluate(xi)
                new_pop_x.append(xi)
                new_pop_f.append(fi)

            if len(new_pop_x) < lambda_:
                break
            pop_x = new_pop_x
            pop_f = new_pop_f

        # --- Phase 2: Model-based refinement ---
        # Collect all evaluated points from CMA-ES history (we only keep best trajectory info)
        # Use the best point found and do local quadratic/GP refinement
        x0 = best_x.copy()
        f0 = best_f

        # Adaptive local search with quadratic model
        if time_ok():
            # Collect a local set of points around x0
            local_pts = [x0.copy()]
            local_fvs = [f0]
            rng2 = np.random.default_rng(123)
            n_local = min(40, budget - calls_made - 5)
            sigma_local = max(0.1, sigma0 * 0.5)

            for i in range(n_local):
                if not time_ok():
                    break
                # Sample around x0 with decreasing sigma
                s = sigma_local * (0.5 ** (i / max(1, n_local // 3)))
                direction = rng2.standard_normal(dim)
                y = np.clip(x0 + s * direction, -5, 5)
                fy = evaluate(y)
                local_pts.append(y)
                local_fvs.append(fy)

            # Fit quadratic model if we have enough points
            if len(local_pts) >= dim + 1 and time_ok():
                local_pts = np.array(local_pts)
                local_fvs = np.array(local_fvs)
                # Center and scale
                xc = local_pts.mean(axis=0)
                xs = local_pts.std(axis=0) + 1e-8
                Z = (local_pts - xc) / xs

                # Build quadratic basis: [1, z_i, z_i*z_j for i<=j]
                n_q = dim * (dim + 1) // 2
                n_basis = 1 + dim + n_q
                A = np.zeros((len(local_pts), n_basis))
                A[:, 0] = 1.0
                A[:, 1:1+dim] = Z
                idx = 1 + dim
                for i in range(dim):
                    for j in range(i, dim):
                        A[:, idx] = Z[:, i] * Z[:, j]
                        idx += 1

                # Ridge regression
                lam = 1e-4
                coef, _, _, _ = np.linalg.lstsq(A + lam * np.eye(n_basis), local_fvs, rcond=None)

                # Optimize quadratic model with gradient descent
                xq = x0.copy()
                for gd_iter in range(200):
                    if not time_ok():
                        break
                    # Compute gradient of quadratic model
                    zq = (xq - xc) / xs
                    grad_z = np.zeros(dim)
                    grad_z[0:0] = 0
                    # d/dz of [1, z, z_i z_j] dot coef
                    # Linear terms: coef[1:1+dim]
                    grad_z = coef[1:1+dim].copy()
                    # Quadratic terms
                    idx = 1 + dim
                    for i in range(dim):
                        for j in range(i, dim):
                            c = coef[idx]
                            if i == j:
                                grad_z[i] += 2 * c * zq[i]
                            else:
                                grad_z[i] += c * zq[j]
                                grad_z[j] += c * zq[i]
                            idx += 1
                    grad_x = grad_z / xs

                    # Newton step
                    H = np.zeros((dim, dim))
                    idx = 1 + dim
                    for i in range(dim):
                        for j in range(i, dim):
                            c = coef[idx]
                            if i == j:
                                H[i, i] += 2 * c
                            else:
                                H[i, j] += c
                                H[j, i] += c
                            idx += 1
                    H /= np.outer(xs, xs)
                    H += 1e-6 * np.eye(dim)

                    try:
                        step = np.linalg.solve(H, grad_x)
                    except np.linalg.LinAlgError:
                        step = grad_x * 0.01

                    # Limit step size
                    step_norm = np.linalg.norm(step)
                    if step_norm > 0.5:
                        step = step / step_norm * 0.5

                    x_new = np.clip(xq - step, -5, 5)
                    f_new = evaluate(x_new)

                    if f_new < best_f:
                        xq = x_new.copy()
                    else:
                        # Try smaller step
                        x_new2 = np.clip(xq - step * 0.1, -5, 5)
                        if np.linalg.norm(x_new2 - xq) < 1e-10:
                            break
                        f_new2 = evaluate(x_new2)
                        if f_new2 < best_f:
                            xq = x_new2.copy()
                        else:
                            break

        # --- Phase 3: Final fine-tuning with very small steps ---
        if time_ok():
            rng3 = np.random.default_rng(777)
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            for _ in range(min(50, budget - calls_made)):
                if not time_ok():
                    break
                direction = rng3.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma = min(sigma * 1.2, 0.1)
                else:
                    sigma = max(sigma * 0.7, 1e-6)
                    if sigma < 1e-5:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
