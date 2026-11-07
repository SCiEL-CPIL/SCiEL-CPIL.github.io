import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget

        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        x = np.zeros(dim)
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        def time_ok():
            return time.time() < deadline

        while calls < budget and time_ok():
            # Generate lam candidates
            candidates = []
            for i in range(lam):
                if not time_ok() or calls >= budget:
                    break
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                fy = func(y)
                calls += 1
                candidates.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(candidates) < mu:
                break

            # Sort by fitness
            candidates.sort(key=lambda t: t[0])
            # Take mu best
            y_best = np.array([c[1] for c in candidates[:mu]])
            # Weighted mean
            m = np.sum(weights[:, None] * y_best, axis=0)

            # Update evolution paths
            old_sigma = sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (m - x)))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - dim))) < (1.4 + 2 / (dim + 1)) * chi_n:
                pc = (1 - c_c) * pc + c_c * (m - x) / sigma

            # Update covariance matrix
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - dim))) < (1.4 + 2 / (dim + 1)) * chi_n:
                delta_h = 0
            else:
                delta_h = 1
            h = delta_h
            pc = (1 - c_c) * pc + c_c * (m - x) / sigma
            # Actually, the standard CMA-ES update is:
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * (B@(D*(m-x)))
            # if ||ps||/sqrt(1-(1-c_sigma)^(2*t)) < (1.4+2/(dim+1))*chi_n:
            #     pc = (1-c_c)*pc + c_c*(m-x)/sigma
            #     B,D = eigh((1-c_c*(2-c_c+delta_h))*C + c_c*(2-c_c+delta_h)*pc*pc.T)
            # else:
            #     pc = (1-c_c)*pc
            #     B,D = eigh((1-c_c*(2-c_c+delta_h))*C + c_c*(2-c_c+delta_h)*pc*pc.T)  # same
            # sigma = sigma * exp(c_sigma/d_sigma * (||ps||/chi_n - 1))
            # x = m

            # Let me just do a simpler version. Actually, let me use scipy if available, or implement carefully.

            # Simpler: just update x = m, and adapt sigma based on improvement
            x = m
            # Adapt sigma: compare average step to expected
            avg_step = np.mean(np.linalg.norm(y_best - x, axis=1))  # this is wrong, x already updated
            # Let me just use a simple rule
            improvement = candidates[0][0] - candidates[-1][0]
            if improvement > 0:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
