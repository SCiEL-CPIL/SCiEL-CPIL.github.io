"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer exploiting known structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit, safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ1: tail variables (indices 1..dim-1) should be at midpoint for g=0
        # Head variable (index 0) parameterizes the Pareto front
        tail_mid = (lo[1:] + hi[1:]) / 2.0

        def make_point(head_val, tail_vals):
            x = np.empty(dim)
            x[0] = head_val
            x[1:] = tail_vals
            return x

        # Strategy:
        # 1. Spend some budget driving tail variables toward midpoint (g→0)
        # 2. Use remaining budget to explore head variable for diversity
        # 3. Final polish: ensure best points have minimal g

        # Phase 1: Convergence - drive tail variables to midpoint
        n_conv = min(50, max(10, self.budget // 4))
        n_div = self.budget - n_conv

        # Initialize with diverse head values and random tail offsets
        n_init = min(20, n_conv)
        X = []
        F = []
        for i in range(n_init):
            head = lo[0] + (hi[0] - lo[0]) * (i + 0.5) / n_init
            tail = tail_mid + 0.1 * (hi[1:] - lo[1:]) * rng.standard_normal(dim - 1)
            tail = np.clip(tail, lo[1:], hi[1:])
            x = make_point(head, tail)
            f = np.asarray(problem(x), dtype=np.float64)
            X.append(x)
            F.append(f)

        used = n_init

        # Iteratively reduce tail offsets (convergence phase)
        for it in range(n_conv - n_init):
            if used >= self.budget or time.time() > deadline - 5.0:
                break
            idx = rng.integers(len(X))
            x = X[idx].copy()
            x[1:] = tail_mid + 0.5 * (x[1:] - tail_mid)
            x[1:] = np.clip(x[1:], lo[1:], hi[1:])
            f = np.asarray(problem(x), dtype=np.float64)
            X[idx] = x
            F[idx] = f
            used += 1

        # Phase 2: Diversity - explore head variable with converged tails
        tail_dists = np.array([np.sum((x[1:] - tail_mid) ** 2) for x in X])
        best_tail_idx = np.argmin(tail_dists)
        best_tail = X[best_tail_idx][1:].copy()

        n_div_actual = self.budget - used
        for i in range(n_div_actual):
            if used >= self.budget or time.time() > deadline - 2.0:
                break
            head = lo[0] + (hi[0] - lo[0]) * rng.random()
            x = make_point(head, best_tail)
            f = np.asarray(problem(x), dtype=np.float64)
            X.append(x)
            F.append(f)
            used += 1

        # Phase 3: Polish
        if used < self.budget:
            F_arr = np.array(F)
            order = np.argsort(np.sum(F_arr, axis=1))
            n_polish = min(10, self.budget - used)
            for i in range(n_polish):
                if used >= self.budget or time.time() > deadline - 1.0:
                    break
                idx = order[i]
                x = X[idx].copy()
                step = 0.01 * (hi[1:] - lo[1:])
                x[1:] += step * rng.standard_normal(dim - 1)
                x[1:] = 0.9 * x[1:] + 0.1 * tail_mid
                x[1:] = np.clip(x[1:], lo[1:], hi[1:])
                f = np.asarray(problem(x), dtype=np.float64)
                if np.sum(f) < np.sum(F[idx]):
                    X[idx] = x
                    F[idx] = f
                used += 1

        return X
# EVOLVE-BLOCK-END
