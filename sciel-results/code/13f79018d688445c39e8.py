"""CMA-ES based optimizer for shifted/rotated Rosenbrock-style f8."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive restarts and boundary handling for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Diverse Initial Points
        candidates = []
        for _ in range(15):
            candidates.append(rng.uniform(-4, 4, dim))
        candidates.append(np.zeros(dim))
        candidates.append(np.ones(dim))
        candidates.append(np.full(dim, -1))
        candidates.append(np.full(dim, 2))
        candidates.append(np.full(dim, 0.5))
        candidates.append(np.full(dim, -0.5))

        scored_candidates = []
        for cand in candidates:
            if call_count >= budget or time.time() > deadline - 50:
                break
            fx = safe_call(cand)
            if np.isfinite(fx):
                scored_candidates.append((fx, cand))
        
        scored_candidates.sort(key=lambda x: x[0])

        # Phase 2: CMA-ES Implementation
        def run_cmaes(x0, max_calls, sigma0=None):
            nonlocal call_count, best_x, best_f
            n = dim
            if sigma0 is None:
                sigma0 = 0.5
            lambda_ = max(4, int(np.ceil(4 + 3 * np.log(n))))
            w = np.zeros(lambda_)
            sum_w = 0
            for i in range(lambda_):
                w[i] = np.log((i + 1) / (lambda_ / 2.0))
            w /= np.sum(w)
            w *= lambda_
            mu = int(np.ceil(lambda_ / 2))
            mu_w = np.sum(w[:mu])
            c_alpha = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (n + 1) - 1)) + 0.01
            c_sigma = (mu_w + 2) / (n + mu_w + 5)
            c_c = (4 + mu_w / n) / (n + 4 + 2 * mu_w / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (n + 1) - 1)) + c_sigma
            c_1 = 2 / ((n + 1.3) ** 2 + mu_w)
            c_mu = min(1 - c_1, 2 * ((mu_w - 2 + 1. / mu_w) / ((n + 2) ** 2 + mu_w)))
            
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            
            steps = 0
            max_steps = max_calls // lambda_
            stall_count = 0
            last_best_f = best_f
            
            for gen in range(max_steps):
                if call_count >= budget or time.time() > deadline - 20:
                    break
                
                # Generate offspring
                ys = []
                xs = []
                fs = []
                
                for k in range(lambda_):
                    if call_count >= budget or time.time() > deadline - 20:
                        break
                    z = rng.standard_normal(n)
                    # Cumulative transformation
                    y = B @ (D * z)
                    x = x_mean + sigma * y
                    x = np.clip(x, -5.0, 5.0)
                    fx = safe_call(x)
                    if not np.isfinite(fx):
                        continue
                    ys.append(y)
                    xs.append(x)
                    fs.append(fx)
                
                if len(fs) < 2:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                ys = np.array([ys[i] for i in order])
                xs = np.array([xs[i] for i in order])
                fs = np.array([fs[i] for i in order])
                
                # Update mean
                x_mean_new = np.sum(w[:mu][:, np.newaxis] * xs[:mu], axis=0)
                
                # Update step size
                y_mean = np.sum(w[:mu][:, np.newaxis] * ys[:mu], axis=0)
                norm_y_mean = np.linalg.norm(y_mean)
                c_hat = (1 - c_sigma) ** (1. / (steps + 1.))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) / norm_y_mean * y_mean
                h_sig = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (steps + 1)) < (1.4 + 2. / (n + 1)) * np.sqrt(n + 1 - 1) else 0
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(y_mean) / np.sqrt(2 * n) - 1))
                
                # Update evolution paths
                p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_w) * y_mean
                C = (1 + c_1 * h_sig * (2 - h_sig) - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(w[:mu][:, np.newaxis] * np.outer(ys[:mu], ys[:mu]), axis=0)
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    B, D = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-12))
                except Exception:
                    B, D = np.eye(n), np.ones(n)
                    C = np.eye(n)
                
                x_mean = x_mean_new
                
                # Check for improvement
                if fs[0] < last_best_f - 1e-8:
                    stall_count = 0
                    last_best_f = fs[0]
                else:
                    stall_count += 1
                
                if stall_count > 10:
                    break

        # Run CMA-ES from the best starting point
        if scored_candidates:
            best_start_f, best_start_x = scored_candidates[0]
            cma_budget = min(300, budget - call_count - 50)
            if cma_budget > 10:
                run_cmaes(best_start_x, cma_budget)

        # Phase 3: Local Optimization with L-BFGS-B
        max_local_starts = min(3, len(scored_candidates))
        for i in range(max_local_starts):
            if call_count >= budget or time.time() > deadline - 30:
                break
            fx0, x0 = scored_candidates[i]
            try:
                bounds = [(-5.0, 5.0)] * dim
                result = minimize(
                    lambda x: safe_call(x),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': max(1, budget - call_count - 10),
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                    }
                )
            except Exception:
                pass

        # Phase 4: Final Polish with Nelder-Mead
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            try:
                remaining = max(1, budget - call_count - 5)
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = np.inf
        
        return best_x, best_f
# EVOLVE-BLOCK-END
