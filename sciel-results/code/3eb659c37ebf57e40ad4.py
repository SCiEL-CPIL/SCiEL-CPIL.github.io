# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def _ok():
            return time.time() < deadline and evaluated < budget

        def _eval(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def _nelder_mead(x0, alloc, seed_off):
            """Nelder-Mead simplex method with adaptive step size."""
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(100 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f

            # Initialize simplex
            simplex = [x.copy()]
            fvals = [fx]
            for i in range(dim):
                xi = x.copy()
                xi[i] += 0.5
                simplex.append(xi)
                fvals.append(_eval(xi))

            # Sort simplex
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            max_iter = alloc
            for _ in range(max_iter):
                if not _ok():
                    break

                # Sort
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflect
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = _eval(x_r)

                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = _eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = _eval(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = _eval(simplex[i])

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    fx = best_f

        def _cma_es(x0, alloc, seed_off):
            """CMA-ES with covariance adaptation."""
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(200 + seed_off)
            pop = max(4, int(np.ceil(4 + 3 * np.log(dim))))
            lam = pop
            sigma = 1.0
            mean = np.array(x0, dtype=float)
            C = np.eye(dim)

            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mue = 1.0 / np.sum(w ** 2)

            cs = (mue + 2) / (dim + mue + 5)
            cc = (4 + mue / dim) / (dim + 4 + 2 * mue / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mue)
            cmu = min(1.0 - c1, 2 * (mue - 2 + 2 / mue) / ((dim + 2) ** 2 + mue))

            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Initial population
            xs = []
            for _ in range(min(lam, alloc)):
                if not _ok():
                    break
                z = r.standard_normal(dim)
                x_new = np.clip(mean + sigma * z, -5.0, 5.0)
                f = _eval(x_new)
                xs.append((x_new, f))

            if len(xs) < mu:
                return

            # Main loop
            while _ok():
                if evaluated + lam > budget:
                    break

                xs = []
                L = None
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    pass

                for _ in range(lam):
                    if not _ok():
                        break
                    z = r.standard_normal(dim)
                    if L is not None:
                        x_new = np.clip(mean + sigma * (L @ z), -5.0, 5.0)
                    else:
                        x_new = np.clip(mean + sigma * z, -5.0, 5.0)
                    f = _eval(x_new)
                    xs.append((x_new, f))

                if len(xs) < mu:
                    break

                xs.sort(key=lambda t: t[1])

                old_mean = mean.copy()
                mean = np.zeros(dim)
                for i in range(mu):
                    mean += w[i] * xs[i][0]

                # Sigma update
                C_sinv = None
                try:
                    C_sinv = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    pass

                d = (mean - old_mean) / (sigma * np.sqrt(mue))
                if C_sinv is not None:
                    d = C_sinv @ d
                ps = (1 - cs) * ps + cs * d
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2)
                chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                if hsig < (1 + 2 / (dim + 1)) * chi:
                    sigma *= np.exp(cs / dim * (hsig - chi))
                sigma = max(sigma, 1e-10)

                # Covariance update
                dc = (mean - old_mean) / sigma
                pc = (1 - cc) * pc + cc * dc
                hcv = np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** 2)
                hc = hcv / ((1 + 2 / (dim + 1)) * chi)

                rank1 = np.outer(pc, pc)
                if hc < (1 + 2 / (dim + 1)) * chi:
                    rank1 += (1 - (1 - cc) ** 2) * (cc / (1 - cc)) * np.eye(dim)

                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    y = (xs[i][0] - old_mean) / sigma
                    rankmu += w[i] * np.outer(y, y)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2

                # Ensure PD
                try:
                    ev, evec = np.linalg.eigh(C)
                    ev = np.maximum(ev, 1e-12)
                    C = evec @ np.diag(ev) @ evec.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    sigma = max(sigma * 0.5, 0.01)

        def _plus_one_es(x0, alloc, seed_off):
            """(1+1)-ES for fine-tuning."""
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(300 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f
            sigma = 0.3

            while _ok():
                if evaluated >= budget:
                    break
                z = r.standard_normal(dim)
                x_try = np.clip(x + sigma * z, -5.0, 5.0)
                f_try = _eval(x_try)
                if f_try < fx:
                    fx = f_try
                    x = x_try
                    sigma = min(sigma * 1.15, 5.0)
                else:
                    sigma = max(sigma * 0.85, 1e-6)

        def _simplex_init():
            """Generate a diverse set of initial points."""
            pts = []
            pts.append(np.zeros(dim))
            for i in range(dim):
                v = np.zeros(dim)
                v[i] = 3.0
                pts.append(v)
                v[i] = -3.0
                pts.append(v)
            for _ in range(4):
                pts.append(rng.uniform(-4.0, 4.0, size=dim))
            return pts

        # === Strategy ===
        # Phase 1: Evaluate diverse initial points
        init_pts = _simplex_init()
        for pt in init_pts:
            if not _ok():
                break
            _eval(pt)

        # Phase 2: Nelder-Mead from best point (good for ridge structure)
        if best_x is not None and _ok():
            alloc_nm = budget // 3
            _nelder_mead(best_x, alloc_nm, seed_off=0)

        # Phase 3: CMA-ES for robust convergence
        if best_x is not None and _ok():
            alloc_cma = budget // 3
            _cma_es(best_x, alloc_cma, seed_off=0)

        # Phase 4: (1+1)-ES polish
        if best_x is not None and _ok():
            remaining = budget - evaluated
            _plus_one_es(best_x, remaining, seed_off=500)

        return best_x, best_f
# EVOLVE-BLOCK-END
