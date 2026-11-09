"""Optimizer for BBOB-style f13 (Sharp ridge) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start + PCA-anisotropic evolution for sharp-ridge minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin before the 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf

        def consider(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x
            return f

        # ---- Phase 1: broad random scan to locate a basin (fast early progress)
        phase1 = max(20, min(80, budget // 5))
        pts = []
        for _ in range(phase1):
            x = rng.uniform(-5.0, 5.0, dim)
            f = func(x)
            pts.append((f, x))
            consider(x, f)
            if time.time() > deadline:
                break

        # keep the K best from the scan
        pts.sort(key=lambda t: t[0])
        K = min(10, len(pts))
        top = np.array([p[1] for p in pts[:K]])
        vals = np.array([p[0] for p in pts[:K]])

        # ---- Phase 2: PCA-anisotropic evolution along the ridge
        # Ridge direction = principal component of the top solutions.
        def ridge_direction(matrix):
            m = matrix - matrix.mean(axis=0)
            try:
                _, Vt = np.linalg.svd(m, full_matrices=False)
            except np.linalg.LinAlgError:
                return None
            d = Vt[0]
            n = np.linalg.norm(d)
            if n < 1e-12:
                return None
            return d / n

        used = phase1
        while used < budget:
            if time.time() > deadline:
                break

            d = ridge_direction(top)
            if d is None:
                # fall back to isotropic steps
                step = 1.0
                for _ in range(min(50, budget - used)):
                    if time.time() > deadline:
                        break
                    x = np.clip(best_x + step * rng.standard_normal(dim), -5.0, 5.0)
                    f = func(x)
                    used += 1
                    consider(x, f)
                    if f < best_f:
                        top = np.vstack([x, top[:-1]])
                        vals = np.concatenate([[f], vals[:-1]])
                    step *= 0.95
                continue

            # large step along the ridge, small step across it
            along = 1.5
            across = 0.3
            improved = False
            for _ in range(min(60, budget - used)):
                if time.time() > deadline:
                    break
                nrm = rng.standard_normal(dim)
                proj = float(np.dot(nrm, d))
                perp = nrm - proj * d
                step = along * d + across * perp
                x = np.clip(best_x + step, -5.0, 5.0)
                f = func(x)
                used += 1
                if f < best_f:
                    best_f = f
                    best_x = x
                    improved = True
                    top = np.vstack([x, top[:-1]])
                    vals = np.concatenate([[f], vals[:-1]])
                    along *= 1.15
                    across *= 1.1
                else:
                    along *= 0.85
                    across *= 0.85

            # periodically re-sample broadly to escape a local ridge trap
            if not improved and used % 200 < 60 and used + 40 < budget:
                for _ in range(40):
                    if time.time() > deadline:
                        break
                    x = rng.uniform(-5.0, 5.0, dim)
                    f = func(x)
                    used += 1
                    consider(x, f)
                    if f < best_f:
                        top = np.vstack([x, top[:-1]])
                        vals = np.concatenate([[f], vals[:-1]])

        return best_x, best_f
# EVOLVE-BLOCK-END
