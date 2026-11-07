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
        deadline = t0 + 290.0

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

        def _powell_line(x, direction, alloc, seed_off):
            """1D search along a direction from x, returns best point and value."""
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return x, best_f

            r = np.random.default_rng(99 + seed_off)
            fx = best_f
            x_best = x.copy()

            # Golden section search along the line
            a = -3.0
            b = 3.0
            gr = (np.sqrt(5) - 1) / 2

            # First evaluate endpoints to bracket
            x_a = np.clip(x + a * direction, -5.0, 5.0)
            x_b = np.clip(x + b * direction, -5.0, 5.0)
            fa = _eval(x_a)
            fb = _eval(x_b)

            # Also evaluate at 0
            f0 = _eval(x.copy())

            # Determine bracket
            if f0 <= fa and f0 <= fb:
                a, b = -1.0, 1.0
                fa, fb = f0, f0
                x_a, x_b = x.copy(), x.copy()
            elif fa <= f0 and fa <= fb:
                a, b = -3.0, 0.0
            else:
                a, b = 0.0, 3.0

            c = a + (1 - gr) * (b - a)
            d = a + gr * (b - a)
            x_c = np.clip(x + c * direction, -5.0, 5.0)
            x_d = np.clip(x + d * direction, -5.0, 5.0)
            fc = _eval(x_c)
            fd = _eval(x_d)

            for _ in range(min(12, alloc)):
                if not _ok():
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = a + (1 - gr) * (b - a)
                    x_c = np.clip(x + c * direction, -5.0, 5.0)
                    fc = _eval(x_c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    x_d = np.clip(x + d * direction, -5.0, 5.0)
                    fd = _eval(x_d)

            # Pick best of all evaluated
            for xx, ff in [(x_a, fa), (x_b, fb), (x, f0), (x_c, fc), (x_d, fd)]:
                if ff < fx:
                    fx = ff
                    x_best = xx.copy()

            return x_best, fx

        def _powell(x0, alloc, seed_off):
            """Powell's method: coordinate descent + conjugate directions."""
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(55 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f

            # Initialize directions: standard basis
            directions = [np.eye(dim)[i] for i in range(dim)]

            for _outer in range(min(8, alloc // (dim + 1))):
                if not _ok():
                    break
                x_old = x.copy()
                f_old = fx
                max_delta = 0.0

                for i in range(dim):
                    if not _ok():
                        break
                    x, fx = _powell_line(x, directions[i], alloc // 10, seed_off + i)
                    max_delta = max(max_delta, abs(f_old - fx))

                # If significant improvement, add conjugate direction
                if max_delta > 1e-10:
                    new_dir = x - x_old
                    norm = np.linalg.norm(new_dir)
                    if norm > 1e-10:
                        new_dir /= norm
                        # Replace the direction that gave the largest improvement
                        # For simplicity, just append and keep top dim directions
                        directions.append(new_dir)
                        if len(directions) > dim:
                            directions = directions[1:]

            return x, fx

        def _plus_one_es(x0, alloc, seed_off):
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(77 + seed_off)
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
            """Generate a diverse set of initial points using simplex + random."""
            pts = []
            # Center
            pts.append(np.zeros(dim))
            # Simplex vertices
            for i in range(dim):
                v = np.zeros(dim)
                v[i] = 3.0
                pts.append(v)
                v[i] = -3.0
                pts.append(v)
            # Additional random points
            for _ in range(4):
                pts.append(rng.uniform(-4.0, 4.0, size=dim))
            return pts

        # === Strategy ===
        # Phase 1: Evaluate a diverse set of initial points to get a good starting position
        init_pts = _simplex_init()
        for pt in init_pts:
            if not _ok():
                break
            _eval(pt)

        # Phase 2: CMA-ES from best point (main workhorse)
        if best_x is not None and _ok():
            alloc_cma = budget // 2
            _cma_es(best_x, alloc_cma, seed_off=0)

        # Phase 3: Powell's method for ridge alignment
        if best_x is not None and _ok():
            alloc_powell = budget // 5
            _powell(best_x, alloc_powell, seed_off=0)

        # Phase 4: (1+1)-ES polish
        if best_x is not None and _ok():
            remaining = budget - evaluated
            _plus_one_es(best_x, remaining, seed_off=200)

        # Phase 5: One more CMA-ES restart from best if budget remains
        if best_x is not None and _ok() and budget - evaluated > 30:
            _cma_es(best_x, budget - evaluated, seed_off=500)

        return best_x, best_f
# EVOLVE-BLOCK-END
