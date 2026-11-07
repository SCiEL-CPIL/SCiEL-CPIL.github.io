"""GP + CMA-ES + Nelder-Mead optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _gp_predict(X, y, x_new, length, var):
    """Posterior mean of a zero-mean GP with RBF kernel (isotropic)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    x_new = np.asarray(x_new, float)
    L = length
    V = var
    diff = X[:, None, :] - X[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    K = V * np.exp(-sq / (2.0 * L * L))
    K += 1e-8 * np.eye(X.shape[0])
    try:
        alpha = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K, y, rcond=None)[0]
    kx = V * np.exp(-np.sum((x_new[None, :] - X) ** 2, axis=1) / (2.0 * L * L))
    return float(kx @ alpha)


class Optimizer:
    """Rotation-invariant hybrid: GP global search, CMA-ES, Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start = time.time()
        deadline = start + 280.0
        calls = 0

        def call(x):
            nonlocal calls
            calls += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # --- Phase 1: GP-assisted global search (randomized posterior-mean minimizer) ---
        X, y = [], []
        best_x, best_f = None, np.inf
        length = 1.5
        var = 1.0
        for i in range(budget):
            if calls >= budget or (time.time() > deadline - 180.0 and i > 25):
                break
            if i < 6:
                x = rng.uniform(-4.5, 4.5, d)
            else:
                # adapt GP hyperparameters to data scale
                span = np.ptp(np.array(X), axis=0) if X else np.ones(d)
                length = float(np.clip(np.mean(span) * 0.5, 0.3, 6.0))
                yv = np.array(y)
                var = float(np.clip(np.var(yv), 1e-4, 1e4))
                best_f_pred = np.inf
                xb = None
                for _ in range(40):
                    x0 = rng.uniform(-5, 5, d)
                    for _ in range(12):
                        grad = np.zeros(d)
                        h = 0.05
                        for j in range(d):
                            xpp = x0.copy(); xpp[j] += h
                            xpm = x0.copy(); xpm[j] -= h
                            grad[j] = (_gp_predict(X, y, xpp, length, var)
                                      - _gp_predict(X, y, xpm, length, var)) / (2 * h)
                        x0 = np.clip(x0 - 0.5 * grad, -5, 5)
                    fp = _gp_predict(X, y, x0, length, var)
                    if fp < best_f_pred:
                        best_f_pred, xb = fp, x0.copy()
                x = xb
            fx = call(x)
            X.append(x); y.append(fx)
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # --- Phase 2: CMA-ES around the best point ---
        if best_x is None:
            best_x = np.zeros(d)
        if calls < budget - 5 and time.time() < deadline - 20.0:
            x = best_x.copy()
            sigma = 1.5
            c1 = (2 + 2.0 / (d + 1.0)) / ((d + 1.0) ** 2 + 2)
            cmu = (2 + 2.0 / (d + 1.0) - d - c1) / ((d + 2.0) ** 2 + d)
            cs = (d + 1.0) / (d + 9.0)
            c_sigma = cs * (1.0 + 2.0 / max(d - 1.0, 1.0))
            damps = 1.0 + 2.0 * max(0.0, (d * c1 - 1.0) / 2.0) + c_sigma
            ps = np.zeros(d)
            C = np.eye(d)
            sqrtC = np.eye(d)
            for _ in range(30):
                if calls >= budget or time.time() > deadline - 15.0:
                    break
                N = 4
                ys, fvs = [], []
                for _ in range(N):
                    w = sqrtC @ rng.standard_normal(d)
                    yc = np.clip(x + sigma * w, -5, 5)
                    fyc = call(yc)
                    ys.append(yc); fvs.append(fyc)
                order = np.argsort(fvs)
                ys = [ys[k] for k in order]
                fvs = [fvs[k] for k in order]
                if fvs[0] < best_f:
                    best_f, best_x = fvs[0], ys[0].copy()
                # update step size
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mu_eff) / damps * (sqrtC @ (np.array(ys[0]) - x) / sigma)
                hsig = np.sqrt((1.0 - (1.0 - cs) ** (2 * (calls - i + 1))) ) if False else 0.0
                if np.linalg.norm(ps) < 1.4 * (1.0 - cs) * np.trace(sqrtC) / d:
                    sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / np.trace(sqrtC) - 1.0))
                sigma = float(np.clip(sigma, 1e-6, 8.0))
                # update covariance
                mu_eff = N / 2.0
                weights = np.log(N / 2.0 + 0.5) - np.log(np.arange(1, N + 1))
                weights /= np.sum(weights)
                ym = np.average([np.array(yy) for yy in ys], axis=0, weights=weights)
                x = ym
                delta = np.array(ys[0]) - ym
                C = (1.0 - c1 - cmu) * C + c1 * np.outer(delta, delta)
                C += cmu * np.sum(weights[:N], axis=0) * np.outer(
                    np.array(ys[0]) - ym, np.array(ys[0]) - ym) if False else C
                C = 0.5 * (C + C.T)
                # recompute sqrt
                try:
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.clip(evals, 1e-12, None)
                    sqrtC = evecs @ np.diag(np.sqrt(evals))
                except np.linalg.LinAlgError:
                    sqrtC = np.eye(d)
                    C = np.eye(d)

        # --- Phase 3: Nelder-Mead polish ---
        if best_x is not None and calls < budget - 2 and time.time() < deadline - 10.0:
            simplex = [best_x.copy()]
            for j in range(d):
                p = best_x.copy()
                p[j] += 0.5 if best_x[j] < 4.5 else -0.5
                simplex.append(p)
            simplexF = [call(p) for p in simplex]
            alpha, gamma, rho = 1.0, 2.0, 0.5
            for _ in range(60):
                if calls >= budget or time.time() > deadline - 5.0:
                    break
                order = np.argsort(simplexF)
                simplex = [simplex[k] for k in order]
                simplexF = [simplexF[k] for k in order]
                if simplexF[0] < best_f:
                    best_f, best_x = simplexF[0], simplex[0].copy()
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = call(xr)
                if fr < simplexF[0]:
                    if fr < alpha * simplexF[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = call(xe)
                        if fe < fr:
                            simplex[-1], simplexF[-1] = xe, fe
                        else:
                            simplex[-1], simplexF[-1] = xr, fr
                    else:
                        simplex[-1], simplexF[-1] = xr, fr
                elif fr < simplexF[-2]:
                    simplex[-1], simplexF[-1] = xr, fr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = call(xc)
                    if fc < simplexF[-1]:
                        simplex[-1], simplexF[-1] = xc, fc
                    else:
                        for k in range(1, d + 1):
                            simplex[k] = 0.5 * (simplex[k] + simplex[0])
                            simplexF[k] = call(simplex[k])

        return best_x, best_f
# EVOLVE-BLOCK-END
