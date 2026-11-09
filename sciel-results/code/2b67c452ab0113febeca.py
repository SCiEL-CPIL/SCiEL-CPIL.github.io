# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II + random-restart multi-objective search for WFG7 (3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 300.0 - 20.0          # safety margin before the 330s kill
        budget = int(self.budget)
        dim = int(self.dim)
        n_obj = int(self.n_obj)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        if np.any(span <= 0):
            span = np.where(span <= 0, 1.0, span)

        # deterministic RNG (evaluator sets np.random.seed, but we use our own for stability)
        rng = np.random.default_rng(12345)

        def rand_pop(n):
            return lo + span * rng.random((n, dim))

        # ---- initial population: a mix of random + a few greedy local improvements ----
        P = rand_pop(24)
        F = np.array([problem(x) for x in P], dtype=float)
        used = P.shape[0]

        # a handful of coordinate-descent refinements to seed the front early
        for _ in range(6):
            if used >= budget:
                break
            if time.time() - t0 > deadline:
                break
            base = P[rng.integers(len(P))].copy()
            for d in range(dim):
                if used >= budget:
                    break
                if time.time() - t0 > deadline:
                    break
                best = base.copy()
                best_f = F[rng.integers(len(P))]  # placeholder, will be overwritten
                # try a few perturbations of this coordinate
                for _ in range(4):
                    cand = best.copy()
                    cand[d] = best[d] + span[d] * rng.standard_normal() * 0.05
                    cand[d] = min(max(cand[d], lo[d]), hi[d])
                    f = problem(cand)
                    used += 1
                    if np.all(f <= best_f):
                        best, best_f = cand, f
            if used >= budget:
                break
            # keep the refined point in the population (replace a random slot)
            idx = rng.integers(len(P))
            P[idx] = best
            F[idx] = best_f

        # ---- main loop ----
        pop_size = len(P)
        while used < budget:
            if time.time() - t0 > deadline:
                break

            # non-dominated sort (cheap, small pop)
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            front = [i for i in range(n) if not dominated[i]]

            # pick parents from the non-dominated front
            if len(front) == 0:
                front = list(range(n))
            a = P[front[rng.integers(len(front))]]
            b = P[front[rng.integers(len(front))]]

            # crossover + mutation (batched, vectorized)
            m = 6
            mask = rng.random((m, dim)) < 0.5
            children = np.where(mask, a, b)
            # gaussian mutation, scaled, plus occasional larger jump
            sigma = 0.05 + 0.05 * (used / max(1, budget))
            children += span * sigma * rng.standard_normal((m, dim))
            # a couple of random-restart children to spread the front
            children[rng.integers(m)] = lo + span * rng.random(dim)
            children[rng.integers(m)] = lo + span * rng.random(dim)
            children = np.clip(children, lo, hi)

            for c in children:
                if used >= budget:
                    break
                if time.time() - t0 > deadline:
                    break
                fc = problem(c)
                used += 1
                # replace the worst (largest sum of objectives) individual
                worst = int(np.argmax(np.sum(F, axis=1)))
                P[worst] = c
                F[worst] = fc

        return P
# EVOLVE-BLOCK-END
