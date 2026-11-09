import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        # Phase 1: Multi-start global exploration (Nelder-Mead inspired)
        # Use diverse starting points to find good basins
        n_starts = 5
        evals_per_start = (budget // 2) // n_starts  # ~50 evals per start

        # Generate diverse starting points
        start_points = [np.zeros(dim)]
        for k in range(1, n_starts):
            start_points.append(rng.uniform(-4.5, 4.5, size=dim))

        for k in range(n_starts):
            if evals_used >= budget or time.time() > deadline:
                break

            x0 = start_points[k].copy()
            f0 = func(x0)
            evals_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Run a short Nelder-Mead from this start
            # Simplex: x0 + alpha * e_i
            alpha = 0.5
            simplex = [x0.copy()]
            fs = [f0]

            for i in range(dim):
                if evals_used >= budget or time.time() > deadline:
                    break
                xi = x0.copy()
                xi[i] += alpha
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                evals_used += 1
                simplex.append(xi)
                fs.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            if len(simplex) < dim + 1:
                continue

            # NM iterations
            max_nm_iter = 30
            for it in range(max_nm_iter):
                if evals_used >= budget or time.time() > deadline:
                    break

                order = np.argsort(fs)
                simplex = [simplex[i].copy() for i in order]
                fs = [fs[i] for i in order]

                if np.std(fs) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + (centroid - simplex[-1]), -5, 5)
                fr = func(xr)
                evals_used += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fs[0]:
                    xe = np.clip(centroid + 2.0 * (centroid - simplex[-1]), -5, 5)
                    fe = func(xe)
                    evals_used += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    if fr < fs[-1]:
                        xc = np.clip(centroid + 0.5 * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    evals_used += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fr:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                            fi = func(simplex[i])
                            evals_used += 1
                            fs[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()

        # Phase 2: Fixed anisotropic ES with large population
        # Use the best point from Phase 1 as center
        x_mean = best_x.copy()
        f_mean = best_f

        # Fixed anisotropic covariance: diagonal with varying scales
        # For step ellipsoidal, axes can be highly anisotropic
        # Use a diagonal covariance that explores broadly in all directions
        # but with some structure to handle the ellipsoidal shape
        eigvals = np.array([1.0, 0.5, 0.25, 0.1, 0.05])
        # Shuffle to match potential random rotation
        perm = rng.permutation(dim)
        eigvals = eigvals[perm]

        C = np.diag(eigvals)
        sigma = 1.5

        # Large population for parallel exploration
        lam = 50
        mu = 25

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])

        # Cholesky of fixed C
        L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))

        # Adaptive sigma with 1/5 rule
        sigma_min = 1e-6
        sigma_max = 5.0

        # Run ES until budget exhausted
        while evals_used + lam <= budget and time.time() < deadline:
            # Generate population
            zs = rng.standard_normal((lam, dim))
            ys = zs @ L.T
            xs = x_mean + sigma * ys
            xs = np.clip(xs, -5, 5)

            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                f_val = func(xs[i])
                evals_used += 1
                fs[i] = f_val
                if f_val < best_f:
                    best_f = f_val
                    best_x = xs[i].copy()

            if evals_used < budget - lam + 1:
                # Didn't complete full generation, break
                break

            # Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)

            # Simple sigma adaptation: 1/5 success rule
            # Count successes in top mu
            successes = 0
            for i in range(mu):
                if fs[i] < f_mean:
                    successes += 1

            if successes > mu * 0.4:
                sigma *= 1.3
            elif successes < mu * 0.2:
                sigma *= 0.7

            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Update f_mean
            f_mean = fs[0]

        # Phase 3: Final polishing with local (1+1)-ES
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = max(sigma * 0.5, 0.01)

            for _ in range(remaining):
                if time.time() > deadline:
                    break

                z = rng.standard_normal(dim)
                x_new = np.clip(x_local + sigma_local * z, -5, 5)
                f_new = func(x_new)
                evals_used += 1

                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.85

                sigma_local = max(sigma_local, 1e-7)

        return best_x, best_f
