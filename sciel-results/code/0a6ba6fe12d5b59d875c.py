import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290

        # Use a deterministic space-filling sequence (Halton) for initial exploration
        # Then use local optimization (Nelder-Mead) on promising points

        # Number of initial points
        n_initial = min(20, budget)
        if n_initial < 2:
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Generate Halton sequence points
        X_init = self._halton_sequence(n_initial, dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = n_initial

        # Archive of non-dominated points
        archive_F = F_init.copy()

        # Check time
        if time.time() > deadline:
            return X_init

        # Now use remaining budget for local optimization
        remaining_budget = budget - used

        # Select top non-dominated points for local refinement
        # We'll pick a few points and run Nelder-Mead on them
        # Strategy: pick the point with best hypervolume contribution or just the first few non-dominated

        # Get non-dominated points from initial set
        nd_indices = self._get_nondominated_indices(F_init)

        # We'll allocate remaining budget to local optimization
        # Each Nelder-Mead call uses about 2*dim + 1 evaluations
        nm_evals_per_run = 2 * dim + 1
        n_nm_runs = min(len(nd_indices), max(1, remaining_budget // nm_evals_per_run))

        # Shuffle to diversify
        if len(nd_indices) > n_nm_runs:
            np.random.shuffle(nd_indices)
            nd_indices = nd_indices[:n_nm_runs]

        for idx in nd_indices:
            if used + nm_evals_per_run > budget:
                break
            if time.time() > deadline:
                break

            x0 = X_init[idx].copy()

            # Define scalar objective: sum of objectives (simple)
            # But better: use a weighted sum or hypervolume contribution
            # For simplicity, use sum of objectives
            def scalar_obj(x):
                fx = problem(x)
                return np.sum(fx)

            # Nelder-Mead
            try:
                result = minimize(scalar_obj, x0, method='Nelder-Mead',
                                  bounds=list(zip(lo, hi)),
                                  options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
                if result.success or result.fun < np.sum(F_init[idx]):
                    F_new = np.array(problem(result.x), dtype=np.float64)
                    # Only add if it's non-dominated by the archive
                    if self._is_nondominated(F_new, archive_F):
                        archive_F = np.vstack([archive_F, F_new])
                    used += nm_evals_per_run
            except Exception:
                # If Nelder-Mead fails, just do a few random perturbations
                for _ in range(min(5, budget - used)):
                    x_pert = x0 + (hi - lo) * 0.05 * np.random.random(dim)
                    x_pert = np.clip(x_pert, lo, hi)
                    F_pert = np.array(problem(x_pert), dtype=np.float64)
                    if self._is_nondominated(F_pert, archive_F):
                        archive_F = np.vstack([archive_F, F_pert])
                    used += 1
                    if time.time() > deadline:
                        break

        # Use any remaining budget for random sampling
        while used < budget and time.time() < deadline:
            x_rand = lo + (hi - lo) * np.random.random(dim)
            F_rand = np.array(problem(x_rand), dtype=np.float64)
            if self._is_nondominated(F_rand, archive_F):
                archive_F = np.vstack([archive_F, F_rand])
            used += 1

        return archive_F

    def _halton_sequence(self, n, dim, lo, hi):
        """Generate n points using Halton sequence in [lo, hi]."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # Use primes as bases
            h = np.zeros(n)
            for i in range(1, n):
                f = 1.0 / base
                n_val = i
                while n_val > 0:
                    f /= base
                    h[i] += f * (n_val % base)
                    n_val //= base
            points[:, d] = h
        # Scale to [lo, hi]
        points = lo + (hi - lo) * points
        return points

    def _get_nondominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = F.shape[0]
        nd = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(i)
        return nd

    def _is_nondominated(self, F_new, archive_F):
        """Check if F_new is non-dominated with respect to archive_F."""
        for f in archive_F:
            if np.all(f <= F_new) and np.any(f < F_new):
                return False
        return True
