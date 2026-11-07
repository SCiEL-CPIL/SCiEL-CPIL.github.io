"""Optimizer for BBOB-style f12 (Bent cigar) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES + valley-direction fitting + 1-D search for the bent cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _fit_valley(self, points, values, dim):
        """Return (t, coeffs) of a polynomial fit, or None."""
        n = len(points)
        if n < 5:
            return None
        idx = np.argsort(values)
        k = min(12, n)
        best = idx[:k]
        P = points[best]
        V = values[best]
        m = P.shape[0]
        if m < 5:
            return None
        # t parameter = index in the sorted-by-value order
        t = np.arange(m, dtype=float)
        # Fit each coordinate to a low-order polynomial in t
        deg = 3
        coeffs = np.zeros((dim, deg + 1))
        valid = True
        for d in range(dim):
            try:
                c = np.polyfit(t, P[:, d], deg)
                coeffs[d] = np.poly1d(c).coef
            except Exception:
                valid = False
                break
        if not valid:
            return None
        return t, coeffs

    def _eval_poly(self, coeffs, t, dim):
        """Evaluate polynomial fit at scalar t -> vector."""
        out = np.zeros(dim)
        for d in range(dim):
            out[d] = np.polyval(coeffs[d], t)
        return out

    def _golden_1d(self, f1d, a, b, iters=60, tol=1e-9):
        """Golden-section minimization of a scalar function on [a, b]."""
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f1d(c)
        fd = f1d(d)
        for _ in range(iters):
            if b - a < tol:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f1d(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f1d(d)
        return 0.5 * (a + b)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # evaluator kills at 330 s wall-clock
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            calls_used += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        def time_ok():
            return time.time() < deadline

        # ---- Phase 0: initial samples to estimate the valley ----
        rng = np.random.default_rng(12345)
        points = []
        values = []

        # Origin
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        points.append(x0.copy())
        values.append(f0)

        # Sample along random directions at a few radii to map the valley
        n_dir = 12
        for i in range(n_dir):
            if calls_used >= budget or not time_ok():
                break
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            for radius in (1.0, 2.5, 4.0):
                if calls_used >= budget or not time_ok():
                    break
                x = radius * direction
                x = np.clip(x, -5, 5)
                fx = evaluate(x)
                points.append(x.copy())
                values.append(fx)

        # ---- Phase 1: CMA-ES for global search ----
        from cma import CMA, CMAOptions

        x0 = best_x.copy()
        # Initial sigma: start moderate, let CMA-ES adapt
        sigma0 = 2.0
        popsize = 8 + int(3 * np.log(dim + 1))  # ~12 for dim=5
        opts = {
            'popsize': popsize,
            'seed': 7,
            'maxfevals': budget - calls_used,
            'bounds': [(-5.0,) * dim, (5.0,) * dim],
            'verbose': -9,
            'tolfun': 1e-12,
            'tolfunint': 1e-12,
            'tolfunup': 1e-9,
        }
        es = CMA(x0, sigma0, opts)
        while not es.stop() and calls_used < budget and time_ok():
            for x in es.ask():
                if calls_used >= budget or not time_ok():
                    break
                fx = evaluate(x)
                es.tell(x, fx)

        # ---- Phase 2: valley fitting + 1-D search ----
        if calls_used < budget and time_ok():
            fit = self._fit_valley(np.array(points), np.array(values), dim)
            if fit is not None:
                t_vals, coeffs = fit
                t_min = t_vals.min()
                t_max = t_vals.max()

                # Extend the search range slightly
                t_lo = t_min - 2.0
                t_hi = t_max + 2.0

                def f1d(t):
                    if calls_used >= budget or not time_ok():
                        return float('inf')
                    x = self._eval_poly(coeffs, t, dim)
                    x = np.clip(x, -5, 5)
                    return evaluate(x)

                # Golden-section search on the valley curve
                t_best = self._golden_1d(f1d, t_lo, t_hi, iters=40)
                x_best = self._eval_poly(coeffs, t_best, dim)
                x_best = np.clip(x_best, -5, 5)
                fx_best = evaluate(x_best)

                # ---- Phase 3: local CMA-ES refinement around the valley point ----
                if calls_used < budget and time_ok() and fx_best < best_f + 1.0:
                    sigma_local = 0.5
                    opts2 = {
                        'popsize': 6,
                        'seed': 99,
                        'maxfevals': budget - calls_used,
                        'bounds': [(-5.0,) * dim, (5.0,) * dim],
                        'verbose': -9,
                        'tolfun': 1e-14,
                        'tolfunint': 1e-14,
                    }
                    es2 = CMA(x_best.copy(), sigma_local, opts2)
                    while not es2.stop() and calls_used < budget and time_ok():
                        for x in es2.ask():
                            if calls_used >= budget or not time_ok():
                                break
                            fx = evaluate(x)
                            es2.tell(x, fx)

        # ---- Phase 4: final local (1+1)-ES polish ----
        if calls_used < budget and time_ok():
            x = best_x.copy()
            fx = evaluate(x)
            sigma = 0.3
            no_improve = 0
            max_no_improve = 30
            while calls_used < budget and time_ok() and no_improve < max_no_improve:
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    sigma *= 1.2
                else:
                    no_improve += 1
                    sigma *= 0.8
                sigma = np.clip(sigma, 1e-5, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
