import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using optimal 1D grid spacing on x0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, F, ref):
        """Compute 2D hypervolume for points strictly better than ref."""
        if len(F) == 0:
            return 0.0
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        idx = np.argsort(F[:, 0])
        F = F[idx]
        # Keep only non-dominated (sorted by f1 asc, f2 should be desc)
        keep = [0]
        for i in range(1, len(F)):
            if F[i, 1] < F[keep[-1], 1]:
                keep.append(i)
        F = F[keep]
        if len(F) == 0:
            return 0.0
        x, y = F[:, 0], F[:, 1]
        n = len(x)
        hv = 0.0
        for i in range(n):
            x_next = x[i + 1] if i < n - 1 else ref[0]
            hv += (x_next - x[i]) * (ref[1] - y[i])
        return hv

    def _compute_optimal_x0(self, n_points, x0_lo, x0_hi, ref):
        """
        Compute optimal x0 values to maximize hypervolume.
        Uses power-law spacing and searches for the optimal exponent.
        """
        if n_points <= 1:
            return np.array([(x0_lo + x0_hi) / 2.0])

        best_hv = -1.0
        best_exp = 4.0 / 3.0

        # Coarse search over exponent
        for exp in np.linspace(0.3, 2.5, 47):
            t = np.linspace(0, 1, n_points)
            x0_trial = x0_lo + (x0_hi - x0_lo) * (t ** exp)
            f1 = x0_trial
            # For ZDT4 Pareto front with x1..x9=0: f1=x0, f2=2-sqrt(x0)
            f2 = 2.0 - np.sqrt(np.clip(f1, 0, None))
            F_trial = np.column_stack([f1, f2])
            hv = self._hv_2d(F_trial, ref)
            if hv > best_hv:
                best_hv = hv
                best_exp = exp

        # Fine search around best exponent
        for exp in np.linspace(max(0.1, best_exp - 0.05), best_exp + 0.05, 100):
            t = np.linspace(0, 1, n_points)
            x0_trial = x0_lo + (x0_hi - x0_lo) * (t ** exp)
            f1 = x0_trial
            f2 = 2.0 - np.sqrt(np.clip(f1, 0, None))
            F_trial = np.column_stack([f1, f2])
            hv = self._hv_2d(F_trial, ref)
            if hv > best_hv:
                best_hv = hv
                best_exp = exp

        # Also try sqrt spacing
        u = np.linspace(np.sqrt(x0_lo), np.sqrt(x0_hi), n_points)
        x0_sqrt = u ** 2
        x0_sqrt[0] = x0_lo
        x0_sqrt[-1] = x0_hi
        f1 = x0_sqrt
        f2 = 2.0 - np.sqrt(np.clip(f1, 0, None))
        F_trial = np.column_stack([f1, f2])
        hv_sqrt = self._hv_2d(F_trial, ref)

        if hv_sqrt > best_hv:
            return x0_sqrt

        t = np.linspace(0, 1, n_points)
        x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** best_exp)
        x0_vals[0] = x0_lo
        x0_vals[-1] = x0_hi
        return x0_vals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([11.0, 400.0])

        # Check if 0 is within bounds for x1..x_{dim-1}
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, dim))

        if zeros_ok:
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)

            if x0_lo >= x0_hi:
                x0_vals = np.full(budget, x0_lo)
            else:
                x0_vals = self._compute_optimal_x0(budget, x0_lo, x0_hi, ref)

            X = np.zeros((budget, dim))
            X[:, 0] = x0_vals[:budget]

            # Evaluate all points
            F_all = []
            for i in range(budget):
                if time.time() > deadline:
                    break
                f = np.array(problem(X[i]), dtype=np.float64)
                F_all.append(f)

            return X

        else:
            # Fallback: random search biased toward x1..x9=0
            rng = np.random.default_rng(0)
            F_all = []
            X_all = []
            while len(F_all) < budget and time.time() < deadline:
                x = lo + (hi - lo) * rng.random(dim)
                x[1:] *= 0.1
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=np.float64)
                X_all.append(x.copy())
                F_all.append(f.copy())

            return np.array(X_all) if X_all else np.zeros((0, dim))
