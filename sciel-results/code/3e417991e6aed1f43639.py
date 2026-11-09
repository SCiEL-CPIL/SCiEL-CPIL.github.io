"""Rewritten optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Sobol + adaptive 1-D ridge tracking for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        # Sobol sequence for low-discrepancy initialization
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        except ImportError:
            class _SobolFallback:
                def __init__(self, d, seed=0):
                    self.d = d
                    self._rng = np.random.default_rng(seed)
                    self._count = 0
                def random(self, m):
                    self._count += m
                    return self._rng.random((m, self.d))
            sobol = _SobolFallback(n)

        # Phase 1: Sparse Halton/Sobol grid to locate the ridge basin
        n_init = min(48, budget // 5)
        sobol.random(1)  # skip origin
        points = sobol.random(n_init) * 10.0 - 5.0

        evals = []  # list of (x, f)
        calls = 0
        best_f = np.inf
        best_x = None

        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            x = points[i]
            f = func(x)
            calls += 1
            evals.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            return None, np.inf

        # Phase 2: Estimate ridge direction from top-k points
        # Sort by function value and take top k
        evals_sorted = sorted(evals, key=lambda t: t[1])
        k = min(10, len(evals_sorted))
        top_points = np.array([e[0] for e in evals_sorted[:k]])
        top_fvals = np.array([e[1] for e in evals_sorted[:k]])
        
        # Weighted centroid of top points
        weights = np.exp(-(top_fvals - top_fvals.min()) / max(top_fvals.max() - top_fvals.min() + 1e-10, 1e-10))
        weights /= weights.sum()
        centroid = (weights[:, None] * top_points).sum(axis=0)
        
        # Estimate principal direction via PCA on top points (weighted)
        # Center the points
        centered = top_points - centroid
        # Weighted covariance
        W = np.diag(weights)
        cov = centered.T @ W @ centered
        # Eigenvector with smallest eigenvalue = ridge direction (slowest variation)
        eigvals, eigvecs = np.linalg.eigh(cov)
        ridge_dir = eigvecs[:, 0]  # smallest eigenvalue = direction of least change = along ridge
        ridge_dir = ridge_dir / (np.linalg.norm(ridge_dir) + 1e-15)

        # Phase 3: 1-D minimization along the ridge direction
        # Start from centroid and search along ridge_dir
        def line_search_1d(start_x, direction, budget_remaining):
            """Minimize func along start_x + t * direction for t in [-5, 5]"""
            nonlocal calls, best_f, best_x
            
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
            a, b = -5.0, 5.0  # search range for t
            
            # Evaluate endpoints
            fa = func(np.clip(start_x + a * direction, -5, 5))
            calls += 1
            fb = func(np.clip(start_x + b * direction, -5, 5))
            calls += 1
            
            if fa < fb:
                b = (a + b) / 2
                fb = fa
                fa = func(np.clip(start_x + a * direction, -5, 5))
                calls += 1
            else:
                a = (a + b) / 2
                fa = fb
                fb = func(np.clip(start_x + b * direction, -5, 5))
                calls += 1
            
            # Iterative golden section
            max_iter = 25
            for _ in range(max_iter):
                if calls >= budget_remaining or elapsed() > deadline:
                    break
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc = func(np.clip(start_x + c * direction, -5, 5))
                calls += 1
                fd = func(np.clip(start_x + d * direction, -5, 5))
                calls += 1
                
                if fc < fd:
                    b = d
                    fb = fd
                else:
                    a = c
                    fa = fc
                
                if abs(b - a) < 1e-6:
                    break
            
            # Final evaluation at midpoint
            t_mid = (a + b) / 2
            x_best_line = np.clip(start_x + t_mid * direction, -5, 5)
            f_best_line = func(x_best_line)
            calls += 1
            return x_best_line, f_best_line

        # Do 1-D search along ridge from centroid
        x_line, f_line = line_search_1d(centroid, ridge_dir, budget)
        if f_line < best_f:
            best_f = f_line
            best_x = x_line.copy()

        # Phase 4: Refine ridge direction and do more 1-D searches
        # Add the new point to our set and re-estimate ridge
        if calls < budget - 20 and elapsed() < deadline:
            evals.append((best_x.copy(), best_f))
            evals_sorted = sorted(evals, key=lambda t: t[1])
            k2 = min(12, len(evals_sorted))
            top_points2 = np.array([e[0] for e in evals_sorted[:k2]])
            top_fvals2 = np.array([e[1] for e in evals_sorted[:k2]])
            
            weights2 = np.exp(-(top_fvals2 - top_fvals2.min()) / max(top_fvals2.max() - top_fvals2.min() + 1e-10, 1e-10))
            weights2 /= weights2.sum()
            centroid2 = (weights2[:, None] * top_points2).sum(axis=0)
            
            centered2 = top_points2 - centroid2
            W2 = np.diag(weights2)
            cov2 = centered2.T @ W2 @ centered2
            eigvals2, eigvecs2 = np.linalg.eigh(cov2)
            ridge_dir2 = eigvecs2[:, 0]
            ridge_dir2 = ridge_dir2 / (np.linalg.norm(ridge_dir2) + 1e-15)

            # 1-D search from best_x along refined ridge
            x_line2, f_line2 = line_search_1d(best_x, ridge_dir2, budget)
            if f_line2 < best_f:
                best_f = f_line2
                best_x = x_line2.copy()

        # Phase 5: Final polish with Nelder-Mead from the best point
        if calls < budget - 10 and elapsed() < deadline:
            def objective_nm(x):
                nonlocal calls
                if calls >= budget:
                    return best_f
                f = func(x)
                calls += 1
                return f

            try:
                result = minimize(
                    objective_nm,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 40, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 6: If budget remains, do one more ridge refinement cycle
        if calls < budget - 15 and elapsed() < deadline:
            evals.append((best_x.copy(), best_f))
            evals_sorted = sorted(evals, key=lambda t: t[1])
            k3 = min(15, len(evals_sorted))
            top_points3 = np.array([e[0] for e in evals_sorted[:k3]])
            top_fvals3 = np.array([e[1] for e in evals_sorted[:k3]])
            
            weights3 = np.exp(-(top_fvals3 - top_fvals3.min()) / max(top_fvals3.max() - top_fvals3.min() + 1e-10, 1e-10))
            weights3 /= weights3.sum()
            centroid3 = (weights3[:, None] * top_points3).sum(axis=0)
            
            centered3 = top_points3 - centroid3
            W3 = np.diag(weights3)
            cov3 = centered3.T @ W3 @ centered3
            eigvals3, eigvecs3 = np.linalg.eigh(cov3)
            ridge_dir3 = eigvecs3[:, 0]
            ridge_dir3 = ridge_dir3 / (np.linalg.norm(ridge_dir3) + 1e-15)

            x_line3, f_line3 = line_search_1d(best_x, ridge_dir3, budget)
            if f_line3 < best_f:
                best_f = f_line3
                best_x = x_line3.copy()

            # Final Nelder-Mead polish
            if calls < budget - 5 and elapsed() < deadline:
                def objective_final(x):
                    nonlocal calls
                    if calls >= budget:
                        return best_f
                    f = func(x)
                    calls += 1
                    return f

                try:
                    result = minimize(
                        objective_final,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-9, 'fatol': 1e-12}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
