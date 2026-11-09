"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES and (1+1)-ES with adaptive sigma for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # Safety margin before 300s

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        def cma_es(start_x, max_calls, pop_size=None):
            """CMA-ES with adaptive step size and rotation."""
            nonlocal best_x, best_f
            if pop_size is None:
                pop_size = max(4, int(4 + 3 * np.log(dim)))
            
            # Initialize
            x_mean = np.clip(start_x, -5, 5).copy()
            f_mean = safe_func(x_mean)
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()
            
            sigma = 1.5
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # CMA-ES parameters
            cmu = 1 / (4 + dim / 2)
            ccs = (1 + 2 * max(0, (dim - 1) / (dim + 2)) + cmu) / (dim + 4 + 2 * cmu)
            cc = (4 + dim / dim) / (dim + 4 + 2 * cmu)
            cs = (1 + 2 * max(0, (dim - 1) / (dim + 2)) + cmu) / (dim + 4 + 2 * cmu)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            used = 1
            stagnation = 0
            max_stagnation = 3 * pop_size
            
            while used < max_calls and call_count < budget and time.time() < deadline:
                # Generate population
                Y = np.array([x_mean + sigma * (C_inv_sqrt @ rng.standard_normal(dim)) for _ in range(pop_size)])
                Y = np.clip(Y, -5, 5)
                
                # Evaluate
                F = np.array([safe_func(y) for y in Y])
                used += pop_size
                
                # Sort
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]
                
                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = Y[0].copy()
                
                # Update mean
                w = np.exp(-np.arange(1, pop_size + 1) / (0.35 * (dim + 2)))
                w /= w.sum()
                mu_eff = 1 / (w @ w)
                
                x_new = (w[:pop_size] * Y).sum(axis=0)
                
                # Update covariance
                yv = (Y[:pop_size] - x_mean) / sigma
                C = (1 - cmu * mu_eff) * C + cmu * (w[:pop_size] * (yv.T @ yv)).sum(axis=0)
                
                # Update step size
                ps = (1 - cs) * ps + cs / chi_n * (C_inv_sqrt @ (x_new - x_mean) / sigma)
                hsig = (1 - np.exp(-1 / ccs)) ** 0.5
                hsig = 1 - np.exp(-1 / ccs)
                h = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (used + 1) / (dim * ccs))) < (1 + 2 / (dim + 1)) * chi_n else 0
                pc = (1 - cc) * pc + cc * h * (x_new - x_mean) / sigma
                C = (1 + cmu * mu_eff * h * (2 - cc) - cc) * C + cmu * mu_eff * h * (pc @ pc)
                
                # Update step size
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 0.01, 10.0)
                
                # Update C_inv_sqrt
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = (eigvecs @ np.diag(1 / np.sqrt(eigvals))) @ eigvecs.T
                
                x_mean = x_new
                f_mean = F[0]
                
                # Stagnation check
                if F[0] >= best_f:
                    stagnation += 1
                else:
                    stagnation = 0
                
                if stagnation >= max_stagnation:
                    break
            
            return x_mean, f_mean

        def run_es_15_rule(start_x, max_calls, initial_sigma=1.5):
            """(1+1)-ES with 1/5 success rule for sigma adaptation."""
            nonlocal best_x, best_f
            x = np.clip(start_x, -5, 5).copy()
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = initial_sigma
            used = 1
            success_count = 0
            window_size = 10
            success_window = 0
            stagnation = 0
            max_stagnation = 40

            while used < max_calls and call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                used += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    success_window += 1
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    success_window += 1
                    stagnation += 1

                # 1/5 success rule: adjust sigma based on success rate in window
                if success_window >= window_size:
                    success_rate = success_count / success_window
                    if success_rate > 0.2:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0
                    success_count = 0
                    success_window = 0

                # Safety: keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

                # Stagnation check
                if stagnation >= max_stagnation:
                    break

            return x, fx

        # Phase 1: CMA-ES for global exploration
        if call_count < budget and time.time() < deadline:
            start1 = np.zeros(dim)
            budget1 = int(budget * 0.4)
            cma_es(start1, budget1, pop_size=10)

        # Phase 2: (1+1)-ES with 1/5 rule for fine-tuning
        if call_count < budget and time.time() < deadline:
            if best_x is not None:
                start2 = best_x.copy()
            else:
                start2 = rng.uniform(-4, 4, dim)
            budget2 = int(budget * 0.3)
            run_es_15_rule(start2, budget2, initial_sigma=0.8)

        # Phase 3: Final polish with (1+1)-ES
        if call_count < budget and time.time() < deadline:
            if best_x is not None:
                start3 = best_x.copy()
            else:
                start3 = rng.uniform(-5, 5, dim)
            budget3 = budget - call_count
            run_es_15_rule(start3, budget3, initial_sigma=0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
