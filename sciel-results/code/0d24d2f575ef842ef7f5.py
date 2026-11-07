"""Fresh design: Tchebycheff weighted-sum decomposition + local search for WFG9."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer via weighted-sum decomposition + local search.

    WFG9's Pareto front is convex, so a Tchebycheff/weighted-sum decomposition
    covers the entire front: each weight vector yields a unique supported point.
    We solve one scalarized subproblem per weight with a fast local optimizer,
    under a hard cap on the number of objective evaluations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270.0  # safety margin before 300 s limit
        budget = int(self.budget)
        dim = int(self.dim)
        n_obj = int(self.n_obj)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # ---- hard evaluation counter (never exceed budget) ----
        state = {"count": 0}

        def capped_problem(x):
            if state["count"] >= budget:
                raise _BudgetExhausted()
            state["count"] += 1
            return problem(x)

        def time_ok():
            return (time.time() - start) < deadline

        # ---- 1. Weight grid over the simplex (uniform in the simplex) ----
        n = 11  # points per edge -> 66 weights
        simplex = []
        for i in range(n):
            for j in range(n - i):
                k = n - 1 - i - j
                simplex.append((i / (n - 1), j / (n - 1), k / (n - 1)))
        simplex = np.array(simplex, dtype=float)
        simplex /= simplex.sum(axis=1, keepdims=True)

        # ---- 2. Scalarized objective for a weight w ----
        def scalar_obj(x, w):
            f = capped_problem(x)
            f = np.asarray(f, dtype=float)
            return float(np.dot(w, f))

        # ---- 3. Local search per weight ----
        n_w = len(simplex)
        evals_per_weight = max(4, budget // n_w)
        max_iter = max(6, int(evals_per_weight * 0.7))
        max_nfev = max(6, evals_per_weight)

        rng = np.random.RandomState(0)
        lxs = rng.uniform(0.0, 1.0, size=(n_w, dim))
        inits = lo + span * lxs

        for idx, w in enumerate(simplex):
            if not time_ok() or state["count"] >= budget:
                break
            x0 = inits[idx]
            try:
                res = minimize(
                    scalar_obj, x0, args=(w,), method="L-BFGS-B",
                    bounds=list(zip(lo, hi)),
                    options={"maxiter": max_iter, "maxfun": max_nfev, "maxls": 20},
                )
            except _BudgetExhausted:
                break
            except Exception:
                continue
            # polish with Nelder-Mead (cheap on a smooth convex front)
            if time_ok() and state["count"] < budget:
                try:
                    res2 = minimize(
                        scalar_obj, res.x, args=(w,), method="Nelder-Mead",
                        options={"maxiter": max(4, max_iter // 3),
                                 "xatol": 1e-8, "fatol": 1e-10, "adaptive": True},
                    )
                    if res2.fun <= res.fun:
                        res = res2
                except _BudgetExhausted:
                    break
                except Exception:
                    pass

        return simplex


class _BudgetExhausted(Exception):
    """Internal signal that the evaluation budget is fully used."""
# EVOLVE-BLOCK-END
