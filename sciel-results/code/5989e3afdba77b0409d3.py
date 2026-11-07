"""Rewritten optimizer for BBOB-style Rastrigin (f15) on [-5,5]^5.

Design: multi-start population search (Differential-Evolution style) with a
Nelder-Mead local polish on the current best, plus an explicit wall-clock
deadline guard. Rastrigin is highly multimodal, so a single (1+1)-ES trajectory
gets trapped in a local basin.  Here we keep a small population that jumps
between basins (large mutation + random reseed on stagnation) and, whenever the
global best improves, we spend a bounded number of evaluations polishing it with
a Nelder-Mead simplex.  Early progress (many distinct basins sampled quickly)
is rewarded by the AOCC score.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Population (DE-style) global search + Nelder-Mead local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _clip(x):
        return np.clip(x, -5.0, 5.0)

    def _nm_polish(self, func, x0, budget_left, deadline, rng):
        """Nelder-Mead simplex. Returns (best_x, best_f, calls_used)."""
        x0 = self._clip(x0).astype(float)
        n = self.dim
        # initial simplex: standard NM with small spread
        simplex = [x0.copy()]
        for i in range(n):
            pt = x0.copy()
            if x0[i] != 0.0:
                pt[i] += 0.05
            else:
                pt[i] += 0.05
            simplex.append(self._clip(pt))
        fvals = [func(p) for p in simplex]
        calls = n + 1
        best_i = int(np.argmin(fvals))
        best_x = simplex[best_i].copy()
        best_f = fvals[best_i]
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while calls < budget_left and (time.time() < deadline):
            fvals.sort()
            simplex.sort(key=lambda p, fv=fvals: fv)  # not used; keep aligned
            # re-sort properly
            order = np.argsort(fvals)
            fvals = [fvals[i] for i in order]
            simplex = [simplex[i] for i in order]
            best_i = 0
            best_x = simplex[0].copy()
            best_f = fvals[0]

            centroid = np.mean(simplex[:-1], axis=0)
            # reflect
            xr = self._clip(centroid + alpha * (centroid - simplex[-1]))
            fr = func(xr)
            calls += 1
            if fr < fvals[0]:
                # expand
                xe = self._clip(centroid + gamma * (centroid - simplex[-1]))
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # contract
                if fr < fvals[-1]:
                    xc = self._clip(centroid + rho * (simplex[-1] - centroid))
                    fc = func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # shrink
                        for i in range(1, n + 1):
                            simplex[i] = self._clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget_left or time.time() >= deadline:
                                break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = self._clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget_left or time.time() >= deadline:
                            break

        return best_x, best_f, calls

    # ---- main -----------------------------------------------------------
    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        # safety margin: stop ~20s before the 330s kill
        deadline = start + 310.0

        rng = np.random.default_rng(0)

        # initial population of small size for fast basin coverage
        pop_size = 12
        pop = [self._clip(rng.uniform(-5, 5, dim)) for _ in range(pop_size)]
        fvals = [func(p) for p in pop]
        calls = pop_size

        best_i = int(np.argmin(fvals))
        best_x = pop[best_i].copy()
        best_f = fvals[best_i]

        # bookkeeping for stagnation
        last_improve_calls = calls
        no_improve = 0
        F = 0.85          # DE mutation scale
        CR = 0.9          # crossover rate

        # Nelder-Mead polish budget per improvement (bounded)
        nm_budget = 40

        while calls < budget and (time.time() < deadline):
            # if close to deadline, just return best
            if time.time() > deadline - 5.0:
                break

            # pick a random base and a random donor
            r1 = rng.integers(0, pop_size)
            r2 = rng.integers(0, pop_size)
            base = best_i  # bias toward the best (exploitation)
            # occasionally use a random base (exploration)
            if rng.random() < 0.3:
                base = rng.integers(0, pop_size)
            v = self._clip(pop[base] + F * (pop[r1] - pop[r2]))
            # binomial crossover
            jrand = rng.integers(0, dim)
            trial = np.empty(dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[base][j]
            trial = self._clip(trial)
            ft = func(trial)
            calls += 1

            if ft <= fvals[base]:
                pop[base] = trial
                fvals[base] = ft
                if ft < best_f - 1e-12:
                    best_f = ft
                    best_x = trial.copy()
                    best_i = base
                    no_improve = 0
                    last_improve_calls = calls
                    # polish the new best with Nelder-Mead if we have budget
                    if calls < budget and (time.time() < deadline) and (budget - calls) > nm_budget:
                        px, pf, used = self._nm_polish(
                            func, best_x, budget - calls - 5, deadline, rng
                        )
                        calls += used
                        if pf < best_f:
                            best_f = pf
                            best_x = px.copy()
                            # also seed the population with the polished point
                            pop[base] = self._clip(px)
                            fvals[base] = pf
            else:
                no_improve += 1

            # stagnation: if no global improvement for a while, reseed some members
            if no_improve >= 15:
                for k in range(3):
                    idx = rng.integers(0, pop_size)
                    # large random jump (escape basin)
                    jump = self._clip(rng.uniform(-5, 5, dim))
                    fjump = func(jump)
                    calls += 1
                    if fjump < fvals[idx]:
                        pop[idx] = jump
                        fvals[idx] = fjump
                        if fjump < best_f:
                            best_f = fjump
                            best_x = jump.copy()
                            best_i = idx
                            no_improve = 0
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
