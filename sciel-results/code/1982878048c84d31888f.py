"""Derivative-free Broyden (DFB) quasi-Newton with global restarts for Bent Cigar."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """DFB quasi-Newton + CMA-ES-lite global restarts for f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0          # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls = 0

        def now_ok():
            nonlocal calls
            return calls < budget and time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- DFB local minimizer ----------
        def dfb(x0, f0, iters):
            nonlocal calls
            x = x0.copy()
            f = f0
            # initialize B (Hessian approx) and g (gradient approx)
            B = np.eye(dim)
            g = np.zeros(dim)
            # finite-difference gradient estimate
            h0 = 1e-2
            for i in range(dim):
                xp = x.copy(); xp[i] += h0
                xm = x.copy(); xm[i] -= h0
                fp = evaluate(np.clip(xp, -5, 5))
                fm = evaluate(np.clip(xm, -5, 5))
                g[i] = (fp - fm) / (2 * h0)
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-12:
                return x, f
            # scale B to match g magnitude
            B = B * (gnorm / np.sqrt(dim))

            for it in range(iters):
                if not now_ok():
                    break
                # Newton step: solve B p = -g
                try:
                    p = np.linalg.solve(B, -g)
                except np.linalg.LinAlgError:
                    p = -g
                pnorm = np.linalg.norm(p)
                if pnorm < 1e-14:
                    break
                # limit step size
                max_step = 4.0
                if pnorm > max_step:
                    p = p * (max_step / pnorm)
                # line search (Armijo backtracking)
                alpha = 1.0
                c1 = 1e-4
                improved = False
                for _ in range(30):
                    x_new = np.clip(x + alpha * p, -5, 5)
                    f_new = evaluate(x_new)
                    if f_new < f - c1 * alpha * gnorm * np.linalg.norm(p):
                        improved = True
                        break
                    alpha *= 0.5
                if not improved:
                    # accept best along the line if any improvement
                    x_new = np.clip(x + p, -5, 5)
                    f_new = evaluate(x_new)
                    if f_new >= f:
                        break
                    alpha = 1.0
                # update
                s = x_new - x
                f = f_new
                x = x_new
                # finite-difference gradient at new point
                h = 1e-3
                g_new = np.zeros(dim)
                for i in range(dim):
                    xp = x.copy(); xp[i] += h
                    xm = x.copy(); xm[i] -= h
                    g_new[i] = (evaluate(np.clip(xp, -5, 5)) -
                                evaluate(np.clip(xm, -5, 5))) / (2 * h)
                y = g_new - g
                # Broyden rank-1 update: B <- B + (y - B s) s^T / (s^T s)
                Bs = B @ s
                if np.linalg.norm(s) > 1e-14:
                    B = B + np.outer(y - Bs, s) / (s @ s)
                # check convergence
                if np.linalg.norm(y) < 1e-10:
                    break
                g = g_new
                gnorm = np.linalg.norm(g)
            return x, f

        # ---------- CMA-ES-lite global sampler ----------
        def cma_global(iters):
            nonlocal calls
            mu = 4
            pop = 2 * mu + 4
            sigma = 2.5
            mean = np.zeros(dim)
            C = np.eye(dim)
            best_local = (None, np.inf)
            for it in range(iters):
                if not now_ok():
                    break
                # sample population
                for _ in range(pop):
                    if not now_ok():
                        break
                    # sample from N(mean, sigma^2 C)
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                    z = rng.standard_normal(dim)
                    x = np.clip(mean + sigma * (L @ z), -5, 5)
                    f = evaluate(x)
                    if f < best_local[1]:
                        best_local = (x.copy(), f)
                if not now_ok():
                    break
                # simple update: move mean toward best, shrink sigma
                if best_local[0] is not None:
                    mean = 0.7 * mean + 0.3 * best_local[0]
                    sigma *= 0.9
            return best_local

        # ---------- Strategy: alternate global exploration + DFB exploitation ----------
        # Phase 1: global search to find a good basin
        global_iters = max(1, budget // 20)
        gx, gf = cma_global(global_iters)
        if gx is None:
            gx = rng.uniform(-5, 5, dim)
            gf = evaluate(gx)

        # Phase 2: DFB local refinement from the best point
        remaining = budget - calls
        if remaining > 2 * dim + 10:
            dfb_iters = max(1, remaining // (2 * dim + 5))
            dx, df = dfb(gx, gf, dfb_iters)

        # Phase 3: if time/budget left, do a few more global+local cycles
        cycles = 3
        for _ in range(cycles):
            if not now_ok():
                break
            # small global perturbation
            if calls + 2 * dim + 10 > budget:
                break
            px = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
            pf = evaluate(px)
            if pf < best_f:
                gx, gf = px.copy(), pf
            remaining = budget - calls
            if remaining > 2 * dim + 10:
                dfb_iters = max(1, remaining // (2 * dim + 5))
                dfb(gx, gf, dfb_iters)

        return best_x, best_f
# EVOLVE-BLOCK-END
