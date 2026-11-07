import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, calls, max_iters=50, step0=0.5):
        """Nelder-Mead for final polish."""
        d = self.dim
        x = np.array(x0, dtype=float)
        f0 = func(x)
        simplex = [x.copy()]
        fs = [f0]
        for i in range(d):
            xi = x.copy()
            xi[i] += step0
            simplex.append(xi)
            fs.append(func(xi))
        best_x = x.copy()
        best_f = f0
        
        for _ in range(max_iters):
            if calls[0] >= self.budget:
                break
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            if fr < fs[0]:
                # Extension
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fs[-1] = fe
                else:
                    simplex[-1] = xr
                    fs[-1] = fr
                continue
            if fr < fs[-2]:
                simplex[-1] = xr
                fs[-1] = fr
                continue
            
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fs[-1]:
                simplex[-1] = xc
                fs[-1] = fc
                continue
            
            # Shrink
            for i in range(1, d + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                fs[i] = func(simplex[i])
                
        return best_x, best_f

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        calls = [0]
        budget = self.budget
        
        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # --- CMA-ES Implementation ---
        # Parameters
        lambda_ = 12  # Population size - slightly larger for better Rastrigin exploration
        mu = 6        # Number of parents
        # Weights
        w = np.log(mu / 2 + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Initial state
        m = np.zeros(dim)
        sigma = 1.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Pre-compute Cholesky (will update)
        L = np.linalg.cholesky(C)
        
        # CMA-ES Loop
        # Reserve ~40 calls for final Nelder-Mead polish
        cma_budget = budget - 40
        
        for gen in range(100): # Max generations safeguard
            if calls[0] >= cma_budget or time.time() > deadline - 30:
                break
            
            # Sample population
            xs = []
            fs = []
            for k in range(lambda_):
                z = np.random.randn(dim)
                x = m + sigma * (L @ z)
                # Optional: Clip to box to avoid wasted calls? 
                # func clips, so it's safe, but CMA-ES stats might be skewed.
                # We'll let func clip.
                xs.append(x)
                fs.append(safe_func(x))
            
            # Sort
            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update best
            for x, f in zip(xs, fs):
                record(x, f)
                
            # Update mean
            m_new = sum(w[i] * xs[i] for i in range(mu))
            
            # Update covariance matrix
            # Delta sigma
            sigma_new = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            
            # Update C
            # v_k = (x_k - m) / sigma
            v = [(xs[i] - m) / sigma for i in range(mu)]
            # Rank-mu update
            C_mu = sum(w[i] * np.outer(v[i], v[i]) for i in range(mu))
            # Rank-1 update
            pc_new = (1 - c_c) * pc + c_c * ((m_new - m) / sigma)
            C_1 = (1 - (c_1 + c_mu)) * C + c_1 * (1 + (1 - c_sigma) * d_sigma * np.outer(pc, pc) - c_sigma * np.outer(pc, pc)) + c_mu * (C_mu - C)
            
            # Symmetrize and ensure positive definite
            C = (C_1 + C_1.T) / 2
            # Add small regularization if needed
            C += 1e-10 * np.eye(dim)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((m_new - m) / sigma)
            
            # Update L
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: restart or add identity
                C = C + 1e-3 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
            m = m_new
            sigma = sigma_new
            pc = pc_new
            
            # Stopping criterion
            if sigma < 1e-6:
                break

        # --- Coordinate-wise refinement before NM ---
        # Quick 1D scan along each axis to escape shallow local minima
        if calls[0] < budget - 20:
            for axis in range(dim):
                if calls[0] >= budget - 20:
                    break
                for delta in [0.3, -0.3, 0.6, -0.6, 1.0, -1.0]:
                    if calls[0] >= budget - 20:
                        break
                    x_try = best_x.copy()
                    x_try[axis] += delta
                    f_try = safe_func(x_try)
                    if f_try < best_f:
                        best_x = x_try.copy()
                        best_f = f_try

        # --- Final Polish with Nelder-Mead ---
        if calls[0] < budget - 5:
            px, pf = self._nelder_mead(safe_func, best_x, calls, max_iters=60, step0=0.8)
            record(px, pf)

        return best_x, best_f
