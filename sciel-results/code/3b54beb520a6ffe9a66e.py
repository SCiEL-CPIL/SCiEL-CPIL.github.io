"""Rastrigin-aware population optimizer with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population search + Rastrigin integer-snapping + local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin before 300s
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(0)

        calls = 0

        def f(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        def time_ok():
            return calls < budget and time.time() < deadline

        # ---- candidate pool ----
        best_x = None
        best_f = np.inf

        def consider(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        # ---- integer-snap helper: snap coords to nearest integer, clamp to box ----
        def snap(x):
            xs = np.round(x)
            return np.clip(xs, -5, 5)

        # ---- 1. broad random population ----
        pop = 40
        for i in range(pop):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            consider(x, f(x))

        # ---- 2. Rastrigin-aware: evaluate integer lattice around a few seeds ----
        # Integer lattice in [-5,5]^5 is huge (11^5=161051), so we sample.
        # Evaluate a structured set: all integer points is too many; sample.
        # Use the current best as anchor and its neighbors, plus random integers.
        if best_x is not None:
            anchor = best_x
        else:
            anchor = np.zeros(dim)

        # evaluate integer points within radius around anchor (L-inf radius 2)
        seen = set()
        base = np.round(anchor).astype(int)
        offsets = np.array(np.ndindex(*[(3,)] * dim)) - 1  # -1,0,1 per dim
        for off in offsets:
            pt = base + off
            pt = np.clip(pt, -5, 5)
            key = tuple(pt)
            if key in seen:
                continue
            seen.add(key)
            if not time_ok():
                break
            consider(pt, f(pt))

        # random integer points
        for _ in range(200):
            if not time_ok():
                break
            pt = rng.integers(-5, 6, dim)
            key = tuple(pt)
            if key in seen:
                continue
            seen.add(key)
            consider(pt, f(pt))

        # ---- 3. population-based refinement around best (CMA-like) ----
        if best_x is not None:
            mu = 20
            sigma = 1.0
            for gen in range(40):
                if not time_ok():
                    break
                # sample population around best
                offs = sigma * rng.standard_normal((mu, dim))
                cands = best_x + offs
                cands = np.clip(cands, -5, 5)
                vals = np.array([f(c) for c in cands])
                # also snap each candidate and evaluate
                for c, v in zip(cands, vals):
                    sc = snap(c)
                    if not time_ok():
                        break
                    consider(sc, f(sc))
                # update best from raw population
                idx = int(np.argmin(vals))
                consider(cands[idx], vals[idx])
                # adapt sigma: shrink if little improvement
                sigma *= 0.9
                if sigma < 1e-4:
                    break

        # ---- 4. local coordinate descent from best ----
        if best_x is not None:
            cur = best_x.copy()
            cur_f = best_f
            step = 0.5
            for _ in range(30):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    for sgn in (1.0, -1.0):
                        trial = cur.copy()
                        trial[d] = cur[d] + sgn * step
                        trial = np.clip(trial, -5, 5)
                        tf = f(trial)
                        if tf < cur_f - 1e-12:
                            cur, cur_f = trial, tf
                            improved = True
                    # snap this coordinate to integer
                    trial = cur.copy()
                    trial[d] = round(cur[d])
                    trial = np.clip(trial, -5, 5)
                    tf = f(trial)
                    if tf < cur_f - 1e-12:
                        cur, cur_f = trial, tf
                        improved = True
                consider(cur, cur_f)
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        # ---- 5. final fine search around best ----
        if best_x is not None:
            for _ in range(300):
                if not time_ok():
                    break
                x = best_x + 0.2 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                consider(x, f(x))

        # ---- 6. final snap of best ----
        if best_x is not None:
            consider(snap(best_x), f(snap(best_x)))

        return best_x, best_f
# EVOLVE-BLOCK-END
