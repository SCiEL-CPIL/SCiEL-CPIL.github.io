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

        def _ok():
            return time.time() < deadline and evaluated < budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        evaluated = 0

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

        def _cma_es(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(42 + seed_off)
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
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(77 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f
            sigma = 0.3
            step_size = 0.8

            while _ok():
                if evaluated >= budget:
                    break
                z = r.standard_normal(dim)
                x_try = np.clip(x + sigma * z, -5.0, 5.0)
                f_try = _eval(x_try)
                if f_try < fx:
                    fx = f_try
                    x = x_try
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    sigma = max(sigma * step_size, 1e-6)

        def _powell_line(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(99 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f

            # Powell-like coordinate descent with Nelder-Mead refinement
            # First do simple coordinate search
            for _ in range(alloc // 3):
                if not _ok():
                    break
                improved = False
                for i in range(dim):
                    if not _ok():
                        break
                    step = 0.5
                    for _s in range(10):
                        if not _ok():
                            break
                        for sign in (1, -1):
                            x_try = x.copy()
                            x_try[i] += sign * step
                            x_try = np.clip(x_try, -5.0, 5.0)
                            f_try = _eval(x_try)
                            if f_try < fx:
                                fx = f_try
                                x = x_try
                                step *= 1.5
                                improved = True
                                break
                        else:
                            break
                    if not _ok():
                        break
                if not improved:
                    break

            # Nelder-Mead refinement
            if _ok():
                simplex = [x.copy() + 0.1 * r.standard_normal(dim) for _ in range(dim)]
                simplex[0] = x.copy()
                simplexF = [_eval(s) for s in simplex]

                for _ in range(alloc // 2):
                    if not _ok():
                        break
                    order = np.argsort(simplexF)
                    for idx in range(len(order)):
                        simplex[idx] = simplex[order[idx]]
                        simplexF[idx] = simplexF[order[idx]]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    worst = simplex[-1]
                    worstF = simplexF[-1]

                    # Reflection
                    alpha = 1.0
                    x_r = np.clip(centroid + alpha * (centroid - worst), -5.0, 5.0)
                    f_r = _eval(x_r)

                    if f_r < simplexF[0]:
                        # Expansion
                        gamma = 2.0
                        x_e = np.clip(centroid + gamma * (centroid - worst), -5.0, 5.0)
                        f_e = _eval(x_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplexF[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplexF[-1] = f_r
                    elif f_r < simplexF[-2]:
                        simplex[-1] = x_r
                        simplexF[-1] = f_r
                    else:
                        # Contraction
                        rho = 0.5
                        x_c = np.clip(centroid + rho * (worst - centroid), -5.0, 5.0)
                        f_c = _eval(x_c)
                        if f_c < worstF:
                            simplex[-1] = x_c
                            simplexF[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplexF[i] = _eval(simplex[i])

                # Track best
                for s, sf in zip(simplex, simplexF):
                    if sf < best_f:
                        best_f = sf
                        best_x = s.copy()

        # === Strategy ===
        # Phase 1: Broad exploration with CMA-ES
        n_starts = 3
        alloc1 = budget // 4

        for i in range(n_starts):
            if not _ok():
                break
            start = rng.uniform(-4.0, 4.0, size=dim)
            _cma_es(start, alloc1, seed_off=i * 100)

        # Phase 2: Powell line search from best
        if best_x is not None and _ok():
            alloc2 = budget // 4
            _powell_line(best_x, alloc2, seed_off=0)

        # Phase 3: CMA-ES refinement
        if best_x is not None and _ok():
            alloc3 = budget // 4
            _cma_es(best_x, alloc3, seed_off=500)

        # Phase 4: (1+1)-ES polish
        if best_x is not None and _ok():
            remaining = budget - evaluated
            _plus_one_es(best_x, remaining, seed_off=200)

        return best_x, best_f
# EVOLVE-BLOCK-END
