"""Fresh design: Multi-Start CMA-ES for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class CMAES:
    """Minimal CMA-ES implementation."""

    def __init__(self, x0, sigma0, budget, dim, rng, func, box_lo, box_hi, deadline=None):
        self.dim = dim
        self.x0 = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.budget = budget
        self.rng = rng
        self.func = func
        self.box_lo = box_lo
        self.box_hi = box_hi
        self.deadline = deadline

        # Parameters
        self.mu = max(1, int(self.dim / 2))
        self.lam = 2 * self.mu
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / self.weights.sum()
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmua = min(1 - self.c1,
                        2 * ((self.mueff + 2) / (self.dim + 2) ** 2 - 1) /
                        ((self.dim + 1.3) ** 2 + self.mueff))
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        self.mean = self.x0.copy()
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        self.best_x = self.x0.copy()
        self.best_f = self.func(self.x0)
        self.calls = 1

    def step(self):
        if self.deadline is not None and time.time() > self.deadline:
            return False
        if self.calls >= self.budget:
            return False

        # Sample
        B = self.rng.standard_normal((self.lam, self.dim))
        # Cholesky of C
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            L = np.eye(self.dim)
        Y = self.sigma * (B @ L.T)
        X = self.mean[None, :] + Y
        X = np.clip(X, self.box_lo, self.box_hi)

        # Evaluate
        F = np.array([self.func(x) for x in X])
        self.calls += self.lam

        # Update best
        for i in range(self.lam):
            if F[i] < self.best_f:
                self.best_f = F[i]
                self.best_x = X[i].copy()

        # Sort
        order = np.argsort(F)
        Y = Y[order]
        X = X[order]
        F = F[order]

        # Weighted mean
        m_new = self.weights @ X

        # Update step-size control
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * L.T @ (m_new - self.mean) / self.sigma
        if np.linalg.norm(self.ps) / self.chiN < (1.4 + 2 / (self.dim + 1)) * self.cc + 1:
            self.sigma *= np.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / self.chiN - 1))

        # Rank-mu update
        yw = (m_new - self.mean) / (self.sigma + 1e-30)
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        Hpc = np.outer(self.pc, self.pc) / self.cc

        # Rank-1 update
        C1 = self.c1 * (np.outer(yw, yw) + (self.cc * (2 - self.cc)) * self.C - Hpc)

        # Rank-mu update
        Yw = Y[:self.mu] / self.sigma
        Cmua = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            w = self.weights[i] * self.cmua
            Cmua += w * np.outer(Yw[i], Yw[i])

        self.C = (1 + self.c1 + self.cmua - self.c1) * self.C + C1 + Cmua
        # Symmetrize
        self.C = 0.5 * (self.C + self.C.T)
        self.mean = m_new
        return True


class Optimizer:
    """Multi-Start CMA-ES for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        box_lo = -5.0
        box_hi = 5.0
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        # Track total calls
        total_calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        # Phase 1: Multi-start coarse CMA-ES
        # Use ~65% of budget for exploration
        n_starts = 5
        budget_per_start = max(10, int((budget * 0.65) / n_starts))

        # Initial starts: center + random
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if time.time() > deadline - 60:  # Leave 60s for polish
                break
            if total_calls >= budget:
                break
            remaining = budget - total_calls
            cma_budget = min(budget_per_start, remaining)
            if cma_budget < 10:
                break
            cma = CMAES(s, 1.5, cma_budget, dim, rng, func, box_lo, box_hi, deadline)
            while cma.step():
                pass
            total_calls += cma.calls
            if cma.best_f < best_f:
                best_f = cma.best_f
                best_x = cma.best_x.copy()

        # Phase 2: Polish from best point
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            polish_budget = min(max(20, int(budget * 0.30)), remaining)
            if polish_budget >= 10:
                cma2 = CMAES(best_x, 0.1, polish_budget, dim, rng, func, box_lo, box_hi, deadline)
                while cma2.step():
                    pass
                total_calls += cma2.calls
                if cma2.best_f < best_f:
                    best_f = cma2.best_f
                    best_x = cma2.best_x.copy()

        # Phase 3: Final ultra-tight polish
        if best_x is not None and total_calls < budget and time.time() < deadline - 10:
            remaining = budget - total_calls
            final_budget = min(max(5, int(budget * 0.05)), remaining)
            if final_budget >= 5:
                cma3 = CMAES(best_x, 0.01, final_budget, dim, rng, func, box_lo, box_hi, deadline)
                while cma3.step():
                    pass
                total_calls += cma3.calls
                if cma3.best_f < best_f:
                    best_f = cma3.best_f
                    best_x = cma3.best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
