"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        # Wrap func to count calls and guard against exceptions
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def safe_call(x):
            if call_count[0] >= budget:
                return best_f[0]
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                call_count[0] += 1
                if not np.isfinite(f):
                    f = 1e10
                if f < best_f[0]:
                    best_f[0] = f
                    best_x[0] = x.copy()
                return f
            except Exception:
                return 1e10

        # CMA-ES Parameters
        # For dim=5, lambda=12 is a good choice
        lam = 12
        mu = lam // 2  # number of weighted individuals
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_mat = (2 + 2 * min(1, (mu_eff - 1) / (dim + 1))) / (dim**1.8 + mu_eff + 13)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial state
        rng = np.random.default_rng(42)
        mean = rng.uniform(-4.0, 4.0, dim)
        sigma = 1.0  # Initial step size
        C = np.eye(dim)
        ps = np.zeros(dim)  # step size control path
        pc = np.zeros(dim)  # evolution path for covariance
        
        # Helper to decompose C into B and D
        def decompose_C(C):
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            return B, D

        B, D = decompose_C(C)
        mean_prev = mean.copy()

        # Main Loop
        while call_count[0] < budget and time.time() < deadline:
            # 1. Sample lam points
            ys = []
            xs = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x = mean + sigma * y
                x = np.clip(x, -5.0, 5.0)
                ys.append(y)
                xs.append(x)
            
            # 2. Evaluate
            fs = []
            for x in xs:
                f = safe_call(x)
                fs.append(f)
                
            if time.time() > deadline or call_count[0] >= budget:
                break

            # 3. Sort by fitness
            order = np.argsort(fs)
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            ys_sorted = ys[order]
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # 4. Update mean
            mean_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            
            # 5. Update step size control
            y_sigma = np.sum(w[:, None] * ys_sorted[:mu], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # 6. Update covariance control
            # h_sigma check
            norm_ps = np.linalg.norm(ps)
            # Approximation for the expected norm of ps
            # E[||ps||] = sqrt(1 - (1 - c_sigma)^(2 * gen)) * sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            # We use a simpler check often found in implementations
            h_sigma = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (call_count[0] // lam + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((1 - c_sigma) / (1 + c_sigma)) else 0.0
            
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - mean_prev) / sigma
            
            # 7. Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_mat * (mu_eff - 2 + delta_h) - c_mat) * C
            C += c_mat * (np.outer(pc, pc) - delta_h * C)
            
            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # 8. Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(mu_eff) - 1))
            
            # 9. Update decomposition
            B, D = decompose_C(C)
            mean = mean_new
            mean_prev = mean_new

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
