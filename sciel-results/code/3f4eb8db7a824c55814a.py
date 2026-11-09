"""Multi-start CMA-ES + Nelder-Mead portfolio for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nm_polish(func, x0, dim, budget, rng, lo=-5.0, hi=5.0):
    """Nelder-Mead with box projection; returns (best_x, best_f, calls_used)."""
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    f0 = func(x0)
    calls = 1
    # Initial simplex
    xk = [x0.copy()]
    for j in range(dim):
        xi = x0.copy()
        step = 0.25 if x0[j] < 0 else -0.25
        xi[j] += step
        xi = np.clip(xi, lo, hi)
        xk.append(xi)
    fx = [f0] + [func(xi) for xi in xk[1:]]
    calls += dim

    for _ in range(budget - calls):
        if calls >= budget:
            break
        order = np.argsort(fx)
        xk = [xk[i] for i in order]
        fx = [fx[i] for i in order]
        if calls >= budget:
            break

        # centroid of all but worst
        centroid = np.mean(xk[:-1], axis=0)
        worst = xk[-1]
        # reflection
        xr = np.clip(centroid + 1.0 * (centroid - worst), lo, hi)
        fr = func(xr)
        calls += 1
        if fr < fx[-2]:
            # expansion
            xe = np.clip(centroid + 2.0 * (xr - centroid), lo, hi)
            fe = func(xe)
            calls += 1
            if fe < fr:
                xk[-1], fx[-1] = xe, fe
            else:
                xk[-1], fx[-1] = xr, fr
        elif fr <= fx[0]:
            xk[-1], fx[-1] = xr, fr
        else:
            # contraction
            xc = np.clip(centroid + 0.5 * (worst - centroid), lo, hi)
            fc = func(xc)
            calls += 1
            if fc < fx[-1]:
                xk[-1], fx[-1] = xc, fc
            else:
                # shrink toward best
                for i in range(1, dim + 1):
                    xk[i] = np.clip(xk[0] + 0.5 * (xk[i] - xk[0]), lo, hi)
                    fx[i] = func(xk[i])
                    calls += 1
                    if calls >= budget:
                        break
        if calls >= budget:
            break

    order = np.argsort(fx)
    best_i = order[0]
    return xk[best_i], fx[best_i], calls


def _cmaes(func, x0, dim, budget, sigma0, rng, lo=-5.0, hi=5.0, deadline=None):
    """CMA-ES with box projection. Returns (best_x, best_f, calls_used)."""
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    # Population size
    lam = int(4 + 3 * np.log(dim))
    lam = max(lam, 4)
    if lam > budget // 2:
        lam = max(4, budget // 2)

    # Initial step size: adapt to box
    sigma = min(sigma0, 1.5)
    if sigma < 1e-4:
        sigma = 1e-4

    mean = x0.copy()
    C = np.eye(dim)
    Cinv = np.eye(dim)
    D = 1.0  # scale of C (sqrt of det)

    # Adaptation parameters
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / weights.sum()
    mueff = 1.0 / np.sum(weights ** 2)
    cs = (mueff + 2) / (dim + mueff + 5)
    ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
    c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (2 - (2 / ((dim + 1.3) ** 2 + mueff)))
    cmu = min(1 - c1,
              2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)
    calls = 0
    best_x = x0.copy()
    best_f = func(x0)
    calls = 1
    best_hist = [best_f]

    while calls < budget:
        if deadline is not None and time.time() > deadline:
            break
        # Evolve distribution
        B, Dmat = np.linalg.eigh(C)
        Dmat = np.sqrt(np.clip(Dmat, 1e-20, None))
        D = Dmat
        B = B  # eigenvectors
        # Sample
        ys = []
        xs = []
        for _ in range(lam):
            z = rng.standard_normal(dim)
            y = (Dmat @ B.T) @ z
            x = mean + sigma * y
            x = np.clip(x, lo, hi)
            ys.append(y)
            xs.append(x)
        ys = np.array(ys)
        xs = np.array(xs)
        fs = np.array([func(x) for x in xs])
        calls += lam
        best_hist.append(fs.min())

        # Update best
        i_best = int(np.argmin(fs))
        if fs[i_best] < best_f:
            best_f = fs[i_best]
            best_x = xs[i_best].copy()

        # Sort
        order = np.argsort(fs)
        xs = xs[order]
        ys = ys[order]
        fs = fs[order]

        # Weighted mean of selected
        y_w = np.sum(weights[:, None] * ys[:mu], axis=0)
        # Update path sigma
        p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (Dmat @ B.T) @ y_w
        hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (calls + 1)) / chiN) < (1.4 + 2 / (dim + 1))
        # Update path covariance
        p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) * y_w
        # Rank-one + rank-mu update
        C = (1 + c1 * hsig * (hsig - 1) * cs * (2 - cs) + c1 - cmu) * C \
            + c1 * np.outer(p_c, p_c) \
            + cmu * np.sum(weights[:, None, None] * np.outer(ys[:mu], ys[:mu]), axis=0)
        # Ensure symmetry / PSD
        C = 0.5 * (C + C.T)
        # Recompute eigen for next iter (defer)
        # Sigma update
        sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
        sigma = np.clip(sigma, 1e-8, 10.0)
        # Move mean
        mean = mean + sigma * y_w
        mean = np.clip(mean, lo, hi)

        # Stagnation: if no improvement for a long time, break out to restart
        if len(best_hist) >= 30:
            recent = best_hist[-30:]
            if recent[0] - recent[-1] < 1e-9:
                break

    return best_x, best_f, calls


class Optimizer:
    """
    Multi-start CMA-ES + Nelder-Mead portfolio for Katsuura (f23).
    1) Diversified warm-start (random + center + stratified) to locate a basin.
    2) CMA-ES with adaptive covariance on the best basin.
    3) Nelder-Mead polish for high-precision convergence.
    4) Multi-restart CMA-ES if budget/time permits.
    5) Deadline guard with safety margin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._start_time = None
        self._deadline_margin = 8.0  # seconds to spare before 330s hard limit

    def __call__(self, func):
        self._start_time = time.time()
        total_budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(20240517)

        deadline = self._start_time + 330.0 - self._deadline_margin

        def _time_ok():
            return time.time() < deadline

        # ---------- Phase 1: Diversified warm-start ----------
        candidates = []
        # Pure random
        for _ in range(6):
            candidates.append(rng.uniform(-5, 5, dim))
        # Center
        candidates.append(np.zeros(dim))
        # Small offsets around center
        for _ in range(4):
            candidates.append(0.5 * rng.standard_normal(dim))
        # Stratified / spread points
        for _ in range(5):
            x = np.zeros(dim)
            for j in range(dim):
                x[j] = rng.uniform(-5, 5)
            candidates.append(x)

        best_x = None
        best_f = np.inf
        calls_made = 0
        for c in candidates:
            if not _time_ok():
                break
            c = np.clip(c, -5, 5)
            fc = func(c)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        remaining = total_budget - calls_made
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # ---------- Phase 2: CMA-ES on the best basin ----------
        # Reserve budget for polish and possible restarts
        cma_budget = int(remaining * 0.60)
        nm_budget = int(remaining * 0.25)
        restart_budget = remaining - cma_budget - nm_budget
        cma_budget = max(20, cma_budget)
        nm_budget = max(10, nm_budget)

        cx, cf, cma_calls = _cmaes(func, best_x, dim, cma_budget, 1.0, rng,
                                   deadline=deadline)
        calls_made += cma_calls
        if cf < best_f:
            best_f = cf
            best_x = cx.copy()

        if not _time_ok():
            return best_x, best_f

        # ---------- Phase 3: Nelder-Mead polish ----------
        if nm_budget > 0 and _time_ok():
            nx, nf, nm_calls = _nm_polish(func, best_x, dim, nm_budget, rng)
            calls_made += nm_calls
            if nf < best_f:
                best_f = nf
                best_x = nx.copy()

        # ---------- Phase 4: Multi-restart CMA-ES if budget remains ----------
        remaining2 = total_budget - calls_made
        if remaining2 > 30 and _time_ok():
            # A few short CMA-ES runs from perturbed starts
            n_restarts = min(3, remaining2 // 30)
            for r in range(n_restarts):
                if not _time_ok():
                    break
                pert = best_x + 1.0 * rng.standard_normal(dim)
                pert = np.clip(pert, -5, 5)
                rb = min(remaining2 - r * 30, 40)
                if rb < 10:
                    break
                rx, rf, rcalls = _cmaes(func, pert, dim, rb, 0.8, rng,
                                        deadline=deadline)
                calls_made += rcalls
                if rf < best_f:
                    best_f = rf
                    best_x = rx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
