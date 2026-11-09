"""Halton space-filling global search + Nelder-Mead local refinement for BBOB f18."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Global Halton exploration followed by local Nelder-Mead exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(z):
            nonlocal best_x, best_f, calls
            z = clip(z)
            f = func(z)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = z.copy()
            return f

        # Phase 1: Global exploration with Halton sequence
        # Use ~60% of budget for broad space-filling coverage
        n_global = int(budget * 0.6)
        # Prime bases for Halton sequence
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]

        def halton(index, base):
            """Generate Halton sequence value for given index and base."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # Generate Halton points in [0,1]^dim, map to [-5,5]^dim
        for i in range(1, n_global + 1):
            if calls >= budget or time.time() > deadline - 60.0:
                break
            point = np.array([halton(i, primes[d]) for d in range(dim)])
            # Map [0,1] to [-5,5]
            x_point = -5.0 + 10.0 * point
            evaluate(x_point)

        # Phase 2: Local refinement with Nelder-Mead on best point
        if best_x is not None and calls < budget - 10 and time.time() < deadline - 30.0:
            try:
                def wrapped(z):
                    nonlocal calls
                    if calls >= budget:
                        return best_f
                    calls += 1
                    return func(clip(z))

                # Use remaining budget for NM iterations
                remaining = budget - calls
                res = minimize(
                    wrapped,
                    best_x,
                    method="Nelder-Mead",
                    options={
                        "maxiter": min(remaining, 200),
                        "xatol": 1e-8,
                        "fatol": 1e-10,
                        "adaptive": True,
                    },
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = clip(res.x)
            except Exception:
                pass

        # Phase 3: If time/budget remains, do additional local refinement
        # from the second-best Halton point to potentially escape local minima
        if best_x is not None and calls < budget - 10 and time.time() < deadline - 20.0:
            try:
                # Find a diverse starting point from Halton exploration
                # Re-evaluate a few points around the best for better local search
                rng = np.random.default_rng(42)
                for attempt in range(3):
                    if calls >= budget - 5 or time.time() > deadline - 10.0:
                        break
                    # Perturb best point slightly
                    perturbed = best_x + rng.normal(0, 0.1, dim)
                    perturbed = clip(perturbed)
                    f_pert = evaluate(perturbed)
                    
                    if f_pert < best_f:
                        # Continue local search from this new point
                        def wrapped2(z):
                            nonlocal calls
                            if calls >= budget:
                                return best_f
                            calls += 1
                            return func(clip(z))
                        
                        remaining = budget - calls
                        if remaining > 10:
                            res2 = minimize(
                                wrapped2,
                                perturbed,
                                method="Nelder-Mead",
                                options={
                                    "maxiter": min(remaining // 2, 50),
                                    "xatol": 1e-8,
                                    "fatol": 1e-10,
                                    "adaptive": True,
                                },
                            )
                            if res2.fun < best_f:
                                best_f = res2.fun
                                best_x = clip(res2.x)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
