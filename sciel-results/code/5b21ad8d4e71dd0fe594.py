"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Optimizer for BBOB f18 (Schaffers F7 ill-conditioned).

    Strategy:
      Phase 1: Random sampling to find diverse seeds.
      Phase 2: Multiple parallel (1+1)-ES runs from the best seeds, with
               adaptive step-size (1/5 rule variant) and periodic restarts
               to escape local minima. This portfolio approach exploits
               multiple basins simultaneously and is robust to the
               ill-conditioned funnel structure of f18.

    Entry point and return interface are kept exactly as required.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit

        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---------------- Phase 1: random sampling ----------------
        n_samples = min(50, max(10, total_budget // 10))
        samples = []
        for _ in range(n_samples):
            if time.time() > deadline - 2.0 or calls >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            try:
                f = func(x)
            except Exception:
                continue
            calls += 1
            samples.append((f, x.copy()))
            record(x, f)

        if not samples:
            return best_x, best_f

        # Sort samples by fitness
        samples.sort(key=lambda t: t[0])

        # ---------------- Phase 2: Multiple (1+1)-ES runs from best seeds ----------------
        # Select distinct seeds from the top samples
        seeds = []
        for fx, x in samples[:10]:
            if all(np.linalg.norm(x - sx) > 0.5 for _, sx in seeds):
                seeds.append((fx, x))
            if len(seeds) >= 5:
                break
        if not seeds:
            seeds = [samples[0]]

        remaining = total_budget - calls
        if remaining <= 0 or time.time() >= deadline:
            return best_x, best_f

        # Allocate budget across runs
        n_runs = len(seeds)
        per_run_budget = max(1, remaining // n_runs)

        for fx0, x0 in seeds:
            if calls >= total_budget or time.time() >= deadline - 5.0:
                break

            x = x0.copy()
            fx = fx0
            sigma = 1.0
            successes = 0
            total = 0
            stall_count = 0
            max_stall = 20
            restart_sigma = 2.0

            for _ in range(per_run_budget):
                if calls >= total_budget or time.time() >= deadline - 2.0:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                try:
                    fy = func(y)
                except Exception:
                    calls += 1
                    continue
                calls += 1
                total += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.1
                    stall_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    stall_count += 1

                # 1/5 success rule adaptation every 1+dim trials
                if total >= 1 + dim:
                    rate = successes / total
                    if rate > 0.3:
                        sigma *= 1.2
                    elif rate < 0.1:
                        sigma *= 0.7
                    successes = 0
                    total = 0

                # Keep sigma in a sane range
                sigma = float(np.clip(sigma, 1e-4, 10.0))

                # Restart if stalled
                if stall_count >= max_stall:
                    sigma = max(restart_sigma, sigma * 2.0)
                    restart_sigma *= 1.5
                    stall_count = 0
                    max_stall = min(max_stall + 10, 80)

        # ---------------- Phase 3: Local refinement ----------------
        if calls < total_budget - 5 and time.time() < deadline - 5:
            remaining_calls = total_budget - calls
            for _ in range(min(remaining_calls, 25)):
                if time.time() > deadline - 2:
                    break
                for scale in [0.1, 0.05, 0.01]:
                    if calls >= total_budget:
                        break
                    x_try = best_x + scale * rng.standard_normal(dim)
                    x_try = np.clip(x_try, -5, 5)
                    try:
                        f_try = func(x_try)
                    except Exception:
                        calls += 1
                        continue
                    calls += 1
                    record(x_try, f_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
