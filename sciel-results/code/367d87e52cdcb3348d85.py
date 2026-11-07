"""Bayesian-optimization style optimizer for f22 (Gallagher 21 peaks)."""
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

class Optimizer:
    """Multi-start + local CMA-ES-like refinement for 21-peak f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def _call(self, func, x):
        """Count a function call."""
        self.calls += 1
        return float(func(x))

    def _cma_es(self, func, x0, budget, rng, sigma0=1.0):
        """A simple population-based CMA-ES variant."""
        n = self.dim
        pop = 4 + int(2 * np.log(n))
        c = 1 / (n + 4)
        d = 1 + 2 * max(0, np.sqrt((pop - 1) / (n + 1)) - 1) / (n + 2)
        c_sigma = (pop + 1.3) / (n + 2)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop - 1) / (n + 1)) - 1)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = 1.0 / (n + 4)
        ds = 1 + 2 * max(0, np.sqrt((pop - 1) / (n + 1)) - 1)
        C = np.eye(n)
        # Eigen decomposition for step
        eigvals, eigvecs = np.linalg.eigh(C)
        inv_sqrt_C = (eigvecs * (1.0 / np.sqrt(np.clip(eigvals, 1e-12, None)))) @ eigvecs.T

        best_x = x0.copy()
        best_f = self._call(func, x0)
        mean = x0.copy()
        iters = budget // pop

        for it in range(iters):
            # Generate population
            ws = np.exp(-0.5 * (np.arange(pop) / (pop / 2)) ** 2)
            ws /= np.sum(ws)
            mu = np.sum(ws[:pop // 2 + 1])
            mu_eff = 1.0 / mu
            wi = ws / mu

            Y = np.zeros((pop, n))
            for i in range(pop):
                z = rng.standard_normal(n)
                y = mean + sigma * (inv_sqrt_C @ z)
                Y[i] = np.clip(y, -5, 5)

            # Evaluate
            F = np.array([self._call(func, Y[i]) for i in range(pop)])
            order = np.argsort(F)
            Ys = Y[order]
            Fs = F[order]

            # Update mean
            new_mean = Ys[:pop // 2 + 1].T @ wi[:pop // 2 + 1]
            new_mean = np.clip(new_mean, -5, 5)

            # Update path
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (sigma * (new_mean - mean)))
            pc = (1 - c) * pc + np.sqrt(c * (2 - c) * mu_eff) * (new_mean - mean)
            mean = new_mean

            # Update covariance
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) / np.sqrt(1 - (1 - c) ** (2 * (it + 1)))
            h_sigma = min(1.4, h_sigma)
            C = (1 + c_sigma * d_sigma * h_sigma - c) * C + c * np.outer(ps, ps) - c * np.outer(pc, pc) * (1 - h_sigma) / c + c * (np.outer(mean, mean) - np.outer(mean, mean))
            # Simpler: just adapt step size based on success rate
            sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) - np.sqrt(n) * (1 + 1.0 / (n + 1))) / (n + 2))
            sigma = np.clip(sigma, 1e-8, 5.0)

            # Update best
            for i in range(pop):
                if Fs[i] < best_f:
                    best_f = Fs[i]
                    best_x = Ys[i].copy()

            # Re-estimate eigenvectors every few iterations
            if it % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                inv_sqrt_C = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

        return best_x, best_f

    def __call__(self, func):
        deadline = time.time() + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Random scan to find promising basins
        scan_budget = min(100, self.budget // 5)
        scan_points = []
        for i in range(scan_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            fx = self._call(func, x)
            scan_points.append((fx, x))

        scan_points.sort()
        
        # Phase 2: Refine top candidates with CMA-ES
        remaining = self.budget - self.calls
        top_k = min(5, len(scan_points))
        per_candidate = max(10, remaining // max(1, top_k))

        best_overall_x = scan_points[0][1].copy()
        best_overall_f = scan_points[0][0]

        for k in range(top_k):
            if time.time() > deadline:
                break
            if self.calls >= self.budget:
                break
            start_x = scan_points[k][1].copy()
            start_f = scan_points[k][0]
            alloc = min(per_candidate, self.budget - self.calls)
            if alloc < 10:
                break

            bx, bf = self._cma_es(func, start_x, alloc, rng, sigma0=0.8)
            if bf < best_overall_f:
                best_overall_f = bf
                best_overall_x = bx.copy()

        # Phase 3: Use remaining budget for final local refinement
        remaining = self.budget - self.calls
        if remaining > 20 and time.time() < deadline:
            bx, bf = self._cma_es(func, best_overall_x, remaining, rng, sigma0=0.3)
            if bf < best_overall_f:
                best_overall_f = bf
                best_overall_x = bx.copy()

        self.best_x = best_overall_x
        self.best_f = best_overall_f
        return best_overall_x, best_overall_f
