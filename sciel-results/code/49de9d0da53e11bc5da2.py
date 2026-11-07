"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Sobol exploration + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best
        best_x = np.array([0.0] * dim)
        best_f = np.inf

        # Wrap func to track calls and best
        call_count = [0]
        best_f_track = [np.inf]
        best_x_track = [np.zeros(dim)]

        def tracked_func(x):
            call_count[0] += 1
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            if fx < best_f_track[0]:
                best_f_track[0] = fx
                best_x_track[0] = x.copy()
            return fx

        # Sobol sequence for global exploration
        # We'll use scipy's QMC or a simple Halton/Sobol generator
        # Let's use scipy.stats.qmc or a manual Sobol
        # For simplicity and speed, use scipy.stats.qmc.Sobol
        try:
            from scipy.stats.qmc import Sobol
            sobol = Sobol(d=dim, scramble=True, seed=42)
            n_sobol = min(200, budget // 3)
            points = sobol.random(n_sobol)
            # Map [0,1]^d to [-5,5]^d
            points = points * 10.0 - 5.0
        except ImportError:
            # Fallback to random
            rng = np.random.default_rng(42)
            n_sobol = min(200, budget // 3)
            points = rng.uniform(-5, 5, size=(n_sobol, dim))

        # Evaluate Sobol points
        for i in range(n_sobol):
            if call_count[0] >= budget - 50:  # reserve some budget
                break
            if time.time() > deadline - 30:
                break
            x = points[i]
            fx = tracked_func(x)

        # Sort by function value and take top candidates for local refinement
        # We need to re-evaluate or track the values
        # Let's collect evaluated points
        eval_points = []
        for i in range(n_sobol):
            if call_count[0] >= budget - 50:
                break
            if time.time() > deadline - 30:
                break
            # Already evaluated, but we need the values
            # Let's just re-evaluate top few by re-running tracked_func
            # Actually, let's keep track properly

        # Better approach: collect all evaluated points and their values
        # Reset and redo properly
        call_count[0] = 0
        best_f_track[0] = np.inf
        best_x_track[0] = np.zeros(dim)

        eval_records = []  # list of (x, f)

        def tracked_func2(x):
            call_count[0] += 1
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            eval_records.append((x.copy(), fx))
            if fx < best_f_track[0]:
                best_f_track[0] = fx
                best_x_track[0] = x.copy()
            return fx

        # Phase 1: Sobol global exploration
        try:
            from scipy.stats.qmc import Sobol
            sobol = Sobol(d=dim, scramble=True, seed=42)
            n_sobol = min(150, budget // 4)
            points = sobol.random(n_sobol)
            points = points * 10.0 - 5.0
        except ImportError:
            rng = np.random.default_rng(42)
            n_sobol = min(150, budget // 4)
            points = rng.uniform(-5, 5, size=(n_sobol, dim))

        for i in range(n_sobol):
            if call_count[0] >= budget - 100:
                break
            if time.time() > deadline - 60:
                break
            x = points[i]
            tracked_func2(x)

        # Sort by function value
        eval_records.sort(key=lambda t: t[1])

        # Phase 2: Local refinement on top candidates
        n_top = min(5, len(eval_records))
        for i in range(n_top):
            if call_count[0] >= budget - 20:
                break
            if time.time() > deadline - 20:
                break

            x0 = eval_records[i][0]
            f0 = eval_records[i][1]

            # Nelder-Mead with bounded tracking
            def nm_func(x):
                return tracked_func2(x)

            # Use Nelder-Mead
            try:
                res = minimize(
                    nm_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'maxfev': min(30, budget - call_count[0] - 10),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                    }
                )
            except Exception:
                pass

        # Return best
        return best_x_track[0], best_f_track[0]
# EVOLVE-BLOCK-END
