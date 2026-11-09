"""Hybrid DE + L-BFGS-B optimizer for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """Hybrid Differential Evolution + L-BFGS-B optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _de_local(self, func, pop, budget, rng):
        """
        Differential Evolution with adaptive F and CR,
        plus periodic L-BFGS-B refinement on best point.
        """
        n = self.dim
        pop_size = len(pop)
        calls_used = 0
        best_f = np.inf
        best_x = None

        # Evaluate initial population
        for i in range(pop_size):
            x = np.clip(pop[i], -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if calls_used >= budget:
            return best_x, best_f, calls_used

        # Adaptive parameters
        F = 0.8
        CR = 0.5
        min_F = 0.3
        max_F = 1.0
        min_CR = 0.1
        max_CR = 0.9

        # Track progress for adaptation
        no_improve = 0
        prev_best = best_f

        while calls_used < budget:
            # Adaptive parameter adjustment
            if no_improve > 5:
                F = np.clip(F * 0.9, min_F, max_F)
                CR = np.clip(CR * 1.1, min_CR, max_CR)
            elif best_f < prev_best - 1e-10:
                no_improve = 0
                # Slightly increase exploration
                F = np.clip(F * 1.05, min_F, max_F)
                CR = np.clip(CR * 0.95, min_CR, max_CR)
            else:
                no_improve += 1

            prev_best = best_f

            # Generate offspring
            new_pop = [None] * pop_size
            for i in range(pop_size):
                if calls_used >= budget:
                    break

                # Select three distinct indices
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if len({r1, r2, r3, i}) == 4:
                        break

                # DE/1/bin mutation
                v = pop[r1].copy()
                mask = rng.random(n) < CR
                if not mask.any():
                    jrand = rng.integers(n)
                    mask[jrand] = True
                v[mask] = pop[r2][mask] + F * (pop[r3][mask] - pop[r2][mask])
                v = np.clip(v, -5, 5)

                f_v = func(v)
                calls_used += 1

                # Selection
                if f_v <= func(np.clip(pop[i], -5, 5)):
                    new_pop[i] = v
                    if f_v < best_f:
                        best_f = f_v
                        best_x = v.copy()
                else:
                    new_pop[i] = pop[i]

            if new_pop[-1] is None:
                new_pop[-1] = pop[-1]
            pop = new_pop

            # Periodic L-BFGS-B refinement every 20 generations or when stuck
            gen = calls_used // pop_size
            if (gen % 20 == 0 and gen > 0) or (no_improve > 15 and gen % 5 == 0):
                if budget - calls_used >= 10:
                    # Try L-BFGS-B from best point
                    try:
                        res = scipy_minimize(
                            func,
                            best_x.copy(),
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * n,
                            options={'maxiter': 50, 'maxfun': 15, 'gtol': 1e-10}
                        )
                        calls_used += res.nfev
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5, 5).copy()
                    except Exception:
                        pass

                    if calls_used >= budget:
                        break

        return best_x, best_f, calls_used

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Multi-start strategy:
        # 1. Evaluate a diverse initial set
        # 2. Run DE on the best few
        # 3. Final L-BFGS-B polish

        # Phase 1: Initial sampling (10% of budget)
        init_budget = max(10, budget // 10)
        init_points = []

        # Origin and structured points
        init_points.append(np.zeros(n))
        init_points.append(np.full(n, 3.0))  # Rosenbrock minimum is near (1,...,1) but rotated
        init_points.append(np.full(n, -3.0))

        # Random points
        while len(init_points) < init_budget:
            init_points.append(rng.uniform(-4.5, 4.5, n))

        # Evaluate and sort
        scored = []
        for x in init_points:
            x = np.clip(x, -5, 5)
            f = func(x)
            scored.append((f, x.copy()))

        scored.sort(key=lambda t: t[0])
        calls_used = len(scored)

        # Keep top K for DE
        K = min(8, len(scored))
        de_pop = [scored[i][1].copy() for i in range(K)]
        best_f = scored[0][0]
        best_x = scored[0][1].copy()

        if calls_used >= budget:
            return best_x, best_f

        # Phase 2: DE with local refinement (80% of budget)
        de_budget = int(budget * 0.8) - calls_used
        if de_budget > 10:
            best_x, best_f, de_calls = self._de_local(
                func, de_pop, de_budget, rng
            )
            calls_used += de_calls

        # Phase 3: Final L-BFGS-B polish (10% of budget)
        if budget - calls_used >= 5:
            try:
                res = scipy_minimize(
                    func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 100, 'maxfun': budget - calls_used - 1, 'gtol': 1e-12}
                )
                calls_used += res.nfev
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
