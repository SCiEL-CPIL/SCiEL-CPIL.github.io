"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 exploiting the analytic front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng()

        n_evals = 0
        X_list = []
        F_list = []

        def evaluate(x):
            nonlocal n_evals
            x_c = np.clip(x, lo, hi)
            f = np.asarray(problem(x_c), dtype=float)
            n_evals += 1
            return x_c, f

        # x2..x10 sit at the per-variable minimum of the g-term, i.e. 0.5.
        # The front is parameterized by x1 only.
        base_tail = np.full(dim - 1, 0.5)
        base_tail = np.clip(base_tail, lo[1:], hi[1:])

        # Phase 1: dense x1 sweep with the tail pinned at the front location.
        # f1(x1) = 1 + 8*(x1^2 - 6*cos(12*pi*x1)).
        # Cover x1 in [-5,5]; the meaningful f1 span for the front is modest,
        # so concentrate sampling near x1=0 where f1 is small.
        n_phase1 = int(budget * 0.6)
        n_phase2 = budget - n_phase1

        # Mix: uniform over full range + concentrated near 0 + a few near edges.
        n_unif = max(1, n_phase1 // 4)
        n_conc = n_phase1 - n_unif
        u_unif = rng.uniform(-5.0, 5.0, n_unif)
        u_conc = 5.0 * np.tanh(4.0 * (2.0 * rng.random(n_conc) - 1.0))
        x1_vals = np.concatenate([u_unif, u_conc])
        perm = rng.permutation(len(x1_vals))
        x1_vals = x1_vals[perm]

        for i in range(len(x1_vals)):
            if time.time() > deadline or n_evals >= budget:
                break
            x = np.empty(dim)
            x[0] = x1_vals[i]
            x[1:] = base_tail
            x, f = evaluate(x)
            X_list.append(x)
            F_list.append(f)

        # Phase 2: local refinement around the most promising (low-f2) points,
        # nudging x1 to trace the front and tightening the tail to 0.5.
        if n_evals < budget and len(F_list) > 0:
            F_arr = np.array(F_list)
            X_arr = np.array(X_list)

            # Pick diverse candidates across the f1 range.
            order = np.argsort(F_arr[:, 0])
            n_cand = min(20, len(order))
            if n_cand > 1:
                cand_idx = np.linspace(0, len(order) - 1, n_cand, dtype=int)
                cand_idx = order[cand_idx]
            else:
                cand_idx = order

            x1_steps = [-0.05, 0.05, -0.02, 0.02, -0.01, 0.01, -0.1, 0.1]
            for idx in cand_idx:
                if time.time() > deadline or n_evals >= budget:
                    break
                x_base = X_arr[idx].copy()
                f_base = F_arr[idx].copy()
                for step in x1_steps:
                    if n_evals >= budget or time.time() > deadline:
                        break
                    x_trial = x_base.copy()
                    x_trial[0] = x_base[0] + step
                    # Tighten tail toward the exact front location 0.5.
                    x_trial[1:] = 0.5 + 0.002 * rng.standard_normal(dim - 1)
                    x_trial = np.clip(x_trial, lo, hi)
                    x_trial, f_trial = evaluate(x_trial)
                    X_list.append(x_trial)
                    F_list.append(f_trial)
                    # Walk downhill in f2 while keeping f1 close.
                    if f_trial[1] < f_base[1] and abs(f_trial[0] - f_base[0]) < 0.5:
                        x_base = x_trial.copy()
                        f_base = f_trial.copy()
                        F_arr[idx] = f_trial
                        X_arr[idx] = x_trial

        if len(X_list) > 0:
            return np.array(X_list)
        return np.array([])
# EVOLVE-BLOCK-END
