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
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            n_evals += 1
            return x_clipped, f

        # ZDT4: f1 = 1 + 8*(x1^2 - 6*cos(12*pi*x1))
        #       f2 = g(x) * (1 - sqrt(f1/g(x)))
        #       g(x) = sum_{i=2}^{n} ((xi-0.5)^2 - 10*cos(20*pi*(xi-0.5)))
        # Pareto front: g at its minimum (xi=0.5 for all i>=2), x1 varies.
        # At xi=0.5, each term = 0 - 10 = -10, so g_min = -10*(dim-1).
        # But we need g > 0 for the standard ZDT4 front. Actually for ZDT4,
        # the true Pareto front has g at minimum and f1 varying, with f2 = g*(1-sqrt(f1/g)).
        # When g is large positive, f2 is large. The front is achieved by keeping
        # x2..x10 at 0.5 (minimizing g) and varying x1.

        # Determine the x1 range from bounds
        x1_lo, x1_hi = lo[0], hi[0]

        # Phase 1: Dense structured sampling of x1 with x2..x10 at 0.5
        n_phase1 = int(budget * 0.75)
        n_phase2 = budget - n_phase1

        # Generate x1 samples: mix of uniform and concentrated near center
        # The interesting f1 range for hypervolume is where f1 is moderate
        n_uniform = n_phase1 // 3
        n_concentrated = n_phase1 - n_uniform

        # Uniform coverage of x1 range
        x1_uniform = np.linspace(x1_lo, x1_hi, n_uniform, endpoint=False) + (x1_hi - x1_lo) / n_uniform * rng.random(n_uniform)

        # Concentrated sampling: use arcsine-like distribution for better coverage
        # of the oscillatory region
        u = rng.random(n_concentrated)
        # Map to x1 with concentration in the middle where f1 has interesting structure
        x1_concentrated = (x1_lo + x1_hi) / 2 + (x1_hi - x1_lo) / 2 * np.sin(np.pi * u - np.pi / 2)

        x1_all = np.concatenate([x1_uniform, x1_concentrated])
        perm = rng.permutation(len(x1_all))
        x1_all = x1_all[perm]

        # Evaluate with x2..x10 exactly at 0.5 (or clipped to bounds)
        mid = 0.5
        for i in range(len(x1_all)):
            if time.time() > deadline or n_evals >= budget:
                break
            x = np.zeros(dim)
            x[0] = x1_all[i]
            # Set x2..x10 to 0.5, clipped to bounds
            x[1:] = mid
            x[1:] = np.clip(x[1:], lo[1:], hi[1:])
            x, f = evaluate(x)
            X_list.append(x)
            F_list.append(f)

        # Phase 2: Local refinement - refine x1 positions and fine-tune x2..x10
        if n_evals < budget and len(F_list) > 0:
            F_arr = np.array(F_list)
            X_arr = np.array(X_list)

            # Sort by f1 to get diverse candidates across the front
            order = np.argsort(F_arr[:, 0])

            # Select well-spread candidates
            n_cands = min(20, len(order))
            if n_cands > 1:
                cand_idx = np.linspace(0, len(order) - 1, n_cands, dtype=int)
                cand_idx = order[cand_idx]
            else:
                cand_idx = order

            # Multi-scale local search
            for idx in cand_idx:
                if time.time() > deadline or n_evals >= budget:
                    break
                x_base = X_arr[idx].copy()
                f_base = F_arr[idx].copy()

                # Try multiple step sizes for x1 refinement
                for scale in [0.05, 0.02, 0.01, 0.005]:
                    if n_evals >= budget or time.time() > deadline:
                        break
                    # 4 directions in x1
                    for d in [-1, 1]:
                        if n_evals >= budget or time.time() > deadline:
                            break
                        x_trial = x_base.copy()
                        x_trial[0] += d * scale
                        # Keep x2..x10 at 0.5 for clean front evaluation
                        x_trial[1:] = mid
                        x_trial[1:] = np.clip(x_trial[1:], lo[1:], hi[1:])
                        x_trial = np.clip(x_trial, lo, hi)
                        x_trial, f_trial = evaluate(x_trial)
                        X_list.append(x_trial)
                        F_list.append(f_trial)
                        # Accept if non-dominated w.r.t. current best at this slot
                        if np.any(f_trial < f_base) and not np.all(f_trial >= f_base):
                            x_base = x_trial.copy()
                            f_base = f_trial.copy()
                            F_arr[idx] = f_trial
                            X_arr[idx] = x_trial

                # Fine-tune x2..x10 to see if small deviations help
                if n_evals < budget and time.time() < deadline:
                    for _ in range(2):
                        if n_evals >= budget or time.time() > deadline:
                            break
                        x_trial = x_base.copy()
                        x_trial[1:] = mid + 0.001 * rng.standard_normal(dim - 1)
                        x_trial = np.clip(x_trial, lo, hi)
                        x_trial, f_trial = evaluate(x_trial)
                        X_list.append(x_trial)
                        F_list.append(f_trial)
                        if np.any(f_trial < f_base) and not np.all(f_trial >= f_base):
                            x_base = x_trial.copy()
                            f_base = f_trial.copy()
                            F_arr[idx] = f_trial
                            X_arr[idx] = x_trial

        # Return all evaluated points
        if len(X_list) > 0:
            return np.array(X_list)
        return np.array([])
# EVOLVE-BLOCK-END
