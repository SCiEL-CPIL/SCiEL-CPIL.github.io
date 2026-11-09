"""Optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hybrid: fast (1+1)-ES local refinement + RBF-surrogate global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(1234)
        t0 = time.perf_counter()
        deadline = t0 + 300.0  # hard wall-clock safety margin

        # ---- initial point: center of the box ----
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # ---- observation store for the surrogate ----
        Xs = [x.copy()]
        Fs = [fx]

        def time_left():
            return deadline - time.perf_counter()

        # ---- local (1+1)-ES refinement ----
        def local_refine(x, fx, budget_left):
            sigma = 0.8
            for _ in range(budget_left):
                if time_left() < 5.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                else:
                    sigma *= 0.82
                if sigma < 1e-6:
                    break
            return x, fx

        # ---- RBF surrogate + EI acquisition ----
        def surrogate_next(x, fx):
            n = len(Fs)
            if n < 2:
                return None
            # use a subset for speed
            if n > 80:
                idx = list(range(n))
                rng.shuffle(idx)
                idx = idx[:80]
                X = np.array([Xs[i] for i in idx])
                F = np.array([Fs[i] for i in idx])
            else:
                X = np.array(Xs)
                F = np.array(Fs)
            F = F - F.mean()
            # RBF kernel
            L = 2.0
            K = np.exp(-0.5 * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1) / (L * L))
            reg = 1e-4 * np.eye(len(F))
            try:
                alpha = np.linalg.solve(K + reg, F)
            except np.linalg.LinAlgError:
                return None
            # candidate sampling for EI
            cands = rng.uniform(-5, 5, size=(2000, dim))
            # add a few near-best candidates
            cands[-20:] = best_x + rng.uniform(-0.5, 0.5, size=(20, dim))
            cands = np.clip(cands, -5, 5)
            # compute K(x, X)
            D2 = np.sum((cands[:, None, :] - X[None, :, :]) ** 2, axis=-1)
            Kx = np.exp(-0.5 * D2 / (L * L))
            mu = Kx @ alpha
            # EI with respect to current best
            f_best = min(F)
            improvement = (f_best - mu)
            # simple EI (deterministic-ish, using a small uncertainty proxy)
            # sigma_x^2 = k(x,x) - Kx @ solve(K, Kx^T); approximate via distance to X
            dmin = np.min(D2, axis=1)
            unc = np.exp(-0.5 * dmin / (L * L))  # in [0,1], larger when far
            sigma_x = 0.3 * (1.0 - unc) + 0.05
            acq = improvement - 0.1 * sigma_x
            j = int(np.argmax(acq))
            return cands[j]

        # ---- main loop ----
        calls = 1
        while calls < budget:
            if time_left() < 5.0:
                break
            # 1) local refinement on the best point
            x, fx = local_refine(x, fx, 40)
            calls += 40
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                Xs.append(x.copy())
                Fs.append(fx)
            # 2) occasionally try a surrogate jump
            if len(Fs) >= 6 and len(Fs) % 6 == 0:
                nx = surrogate_next(x, fx)
                if nx is not None:
                    nfx = func(nx)
                    calls += 1
                    Xs.append(nx.copy())
                    Fs.append(nfx)
                    if nfx < best_fx:
                        best_fx = nfx
                        best_x = nx.copy()
                    # refine the new point
                    x, fx = local_refine(nx, nfx, 20)
                    calls += 20
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        Xs.append(x.copy())
                        Fs.append(fx)
            if calls >= budget:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
