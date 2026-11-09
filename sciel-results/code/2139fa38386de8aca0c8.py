"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cmaes(func, x0, sigma0, budget, dim, deadline, best_x, best_f, seed=0):
    """CMA-ES. Returns (best_x, best_f, calls_used). best_x/best_f are the
    global best seen so far (across this and prior phases)."""
    rng = np.random.default_rng(seed)
    n = dim
    x = np.asarray(x0, dtype=float).copy()
    fx = func(x)
    calls = 1
    if fx < best_f:
        best_f, best_x = fx, x.copy()
    sigma = sigma0
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    cc = (4 + n / 3.0) / ((n + 2) ** 2 + 4)
    cs = (2 + 2.0 / max(n - 1.0, 1.0)) / (n + 4 + 2.0 / max(n - 1.0, 1.0))
    cmu = 2 / ((n + 1.3) ** 2 + n)
    damps = 1 + 2 * max(0.0, np.sqrt((1 + np.sqrt(n)) - 1) - 1) + cs
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    while calls < budget:
        if time.time() > deadline:
            break
        m = max(2, min(2 + int(np.floor(0.35 * n)), budget - calls))
        if m < 2:
            break
        # sample population
        pop = []
        for _ in range(m):
            B = np.linalg.cholesky(C)
            z = rng.standard_normal(n)
            pop.append(x + sigma * (B @ z))
        # evaluate
        fvals = np.empty(m)
        for i in range(m):
            if time.time() > deadline:
                break
            fvals[i] = func(pop[i])
            calls += 1
        order = np.argsort(fvals)
        mu = m // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # weighted mean
        xnew = sum(w[i] * pop[order[i]] for i in range(mu))
        # update CMA
        y = (xnew - x) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C @ y
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1)) / chi_n <
                (1.4 + 2.0 / (n + 1)) * cc)
        pc = (1 - cc) * pc + hsig * cc * y
        rank_mu = np.outer(y, y)
        for i in range(mu):
            rank_mu += 2 * w[i] * np.outer(pop[order[i]] - xnew, pop[order[i]] - xnew) / sigma ** 2
        delta = (2 * hsig - 1) * (1 - cc) * np.outer(pc, pc)
        C = (1 + cmu * (mueff + 2) - cs) * C + rank_mu + delta
        sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
        x = xnew
        if np.isfinite(fvals[order[0]]) and fvals[order[0]] < best_f:
            best_f, best_x = fvals[order[0]], pop[order[0]].copy()
        # stop criteria
        if sigma < 1e-12:
            break
        if np.ptp(fvals[:mu]) < 1e-12:
            break
    return best_x, best_f, calls


class Optimizer:
    """Random basin scan + CMA-ES local search (rotation-invariant)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # --- Phase A: random scan to locate a promising basin ---
        n_scan = max(8, min(budget // 6, 80))
        rng = np.random.default_rng(1234)
        cands = [np.zeros(dim), np.array([1.0] * dim), np.array([-1.0] * dim),
                 np.array([2.5] * dim), np.array([-2.5] * dim)]
        while len(cands) < n_scan:
            cands.append(rng.uniform(-4, 4, dim))
        for c in cands:
            if time.time() > deadline:
                break
            fc = func(np.clip(c, -5, 5))
            calls += 1
            if fc < best_f:
                best_f, best_x = fc, np.clip(c, -5, 5).copy()

        # --- Phase B: CMA-ES local search from the best point ---
        remaining = budget - calls
        if remaining > 5:
            best_x, best_f, c2 = _cmaes(func, best_x, 0.8, remaining, dim,
                                         deadline, best_x, best_f, seed=7)
            calls += c2

        # --- Phase C: fine polish if budget remains ---
        remaining = budget - calls
        if remaining > 4:
            best_x, best_f, c3 = _cmaes(func, best_x, 0.15, remaining, dim,
                                         deadline, best_x, best_f, seed=99)
            calls += c3

        return best_x, best_f
# EVOLVE-BLOCK-END
