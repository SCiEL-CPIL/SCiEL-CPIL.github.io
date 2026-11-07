"""Rewritten optimizer: differential-evolution population search with adaptive F,
best-1 exploitation, and a late local-polish phase. Structural change from (1+1)-ES."""
import numpy as np
import time


class Optimizer:
    """DE/rand/1/bin population optimizer with adaptive mutation, best-1 bias,
    and a final local polish. Robust to hidden rotation/shift/offset."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        start = time.time()
        deadline = start + 285.0  # safety margin well before 300s kill
        dim = self.dim
        budget = self.budget
        LO, HI = -5.0, 5.0

        calls = 0

        def clip(x):
            return np.clip(x, LO, HI)

        # ---- initial population ----
        lam = 20
        # ensure lam <= remaining budget
        while lam > 2 and lam > budget - 1:
            lam //= 2

        def fresh():
            return rng.uniform(LO, HI, dim)

        pop = [fresh() for _ in range(lam)]
        fvals = np.empty(lam)
        for i in range(lam):
            if calls >= budget:
                break
            fvals[i] = func(pop[i])
            calls += 1
            if time.time() > deadline:
                break
        # fill any unfilled (shouldn't happen) with best-known
        if calls < lam:
            # fallback: fill with zeros
            for i in range(calls, lam):
                pop[i] = np.zeros(dim)
                fvals[i] = fvals[:calls].min() if calls else 0.0

        order = np.argsort(fvals)
        best_i = int(order[0])
        best_x = pop[best_i].copy()
        best_f = float(fvals[best_i])

        # re-sort population by fitness (best first) for selection convenience
        idx = order
        pop = [pop[i].copy() for i in idx]
        fvals = fvals[idx].copy()

        F = 0.8  # mutation scale
        CR = 0.5  # crossover rate
        no_improve = 0

        # main DE loop
        while calls < budget - 1:
            if time.time() > deadline:
                break

            # adapt F: explore more when stagnating
            if no_improve > 8:
                F = min(1.2, F * 1.3)
            elif no_improve < 2:
                F = max(0.3, F * 0.9)

            gen_best_i = 0
            gen_best_f = fvals[0]
            n_eval_this_gen = 0

            for i in range(lam):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # pick two distinct random indices different from i
                r1, r2 = rng.choice(lam, size=2, replace=False)
                while r1 == i:
                    r1 = rng.integers(0, lam)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, lam)

                v = pop[i] + F * (pop[int(r1)] - pop[int(r2)])
                v = clip(v)

                # binary crossover
                mask = rng.random(dim) < CR
                if not mask.any():
                    j = rng.integers(0, dim)
                    mask[j] = True
                trial = np.where(mask, v, pop[i])

                ft = func(trial)
                calls += 1
                n_eval_this_gen += 1

                if ft < fvals[i]:
                    pop[i] = trial
                    fvals[i] = ft
                    if ft < gen_best_f:
                        gen_best_f = ft
                        gen_best_i = i

            # best/1 exploitation: one extra candidate from global best
            if calls < budget:
                if time.time() > deadline:
                    break
                r1, r2 = rng.choice(lam, size=2, replace=False)
                v = best_x + F * (pop[int(r1)] - pop[int(r2)])
                v = clip(v)
                ft = func(v)
                calls += 1
                if ft < gen_best_f:
                    gen_best_f = ft
                    gen_best_i = int(r1)  # placeholder; we compare to best below
                    # update population slot with the best candidate
                    # find the worst slot to replace
                    worst = int(np.argmax(fvals))
                    if ft < fvals[worst]:
                        pop[worst] = v
                        fvals[worst] = ft

            # update global best from this generation
            gmin_i = int(np.argmin(fvals))
            if fvals[gmin_i] < best_f:
                best_f = float(fvals[gmin_i])
                best_x = pop[gmin_i].copy()
                no_improve = 0
            else:
                no_improve += 1

            # keep population sorted-ish: replace worst with a perturbation of best
            # (maintains diversity + exploitation balance)
            worst = int(np.argmax(fvals))
            if fvals[worst] > best_f + 1e-6:
                # occasionally re-seed worst from best with noise to keep basin coverage
                if rng.random() < 0.5:
                    newp = clip(best_x + 0.5 * F * rng.standard_normal(dim))
                    fn = func(newp)
                    calls += 1
                    if fn < fvals[worst]:
                        pop[worst] = newp
                        fvals[worst] = fn

        # ---- local polish phase: refine around global best ----
        # Use any remaining budget to do a decaying-step local search.
        remaining = budget - calls
        if remaining > 0:
            step = 0.5
            cx = best_x.copy()
            cf = best_f
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                cand = clip(cx + step * rng.standard_normal(dim))
                fc = func(cand)
                calls += 1
                if fc < cf:
                    cx = cand
                    cf = fc
                    if fc < best_f:
                        best_f = fc
                        best_x = cx.copy()
                    step *= 0.9
                else:
                    step *= 0.7
                if step < 1e-4:
                    step = 1e-4

        return best_x, best_f
