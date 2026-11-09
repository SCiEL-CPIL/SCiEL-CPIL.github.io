import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0 # 5s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi
        
        rng = np.random.default_rng(42)
        
        # 1. Scout Phase: Find a good starting point
        # Use ~10% of budget for scouting, min 20, max 30
        scout_budget = min(30, max(20, budget // 10))
        
        best_x = None
        best_f = np.inf
        scout_points = []
        
        # Sample uniformly, including origin
        x0 = np.zeros(d)
        fx0 = func(x0)
        scout_points.append((x0, fx0))
        if fx0 < best_f:
            best_f = fx0
            best_x = x0.copy()
            
        for _ in range(scout_budget - 1):
            x = rng.uniform(lo, hi, d)
            fx = func(x)
            scout_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            scout_budget = 1

        # 2. CMA-ES Phase
        # Parameters for CMA-ES
        # Population size lambda
        lam = 4 + int(3 * np.log(d))
        if lam < 8:
            lam = 8
        # Weighted mean: weights for the best mu candidates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # State variables
        m = best_x.copy()
        
        # Estimate initial sigma from scout points
        # Use the spread of the best half of the scout points
        if len(scout_points) > 2:
            scout_xs = np.array([p[0] for p in scout_points])
            scout_fs = np.array([p[1] for p in scout_points])
            # Use spread of points with good fitness
            good_mask = scout_fs < np.percentile(scout_fs, 50)
            if np.sum(good_mask) >= 2:
                good_xs = scout_xs[good_mask]
                # Heuristic: average distance from mean of good points
                mean_good = np.mean(good_xs, axis=0)
                dists = np.linalg.norm(good_xs - mean_good, axis=1)
                sigma = max(0.5, np.mean(dists) * 0.5)
            else:
                sigma = 1.0
        else:
            sigma = 1.0
            
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        remaining_budget = budget - scout_budget
        best_f_cma = best_f
        best_x_cma = best_x.copy()
        
        # Pre-compute constants
        chi_n = np.sqrt(d) * (1 - 1.0/(4*d) + 1.0/(21*d*d))
        
        # Main loop
        while remaining_budget >= lam:
            # Check time limit
            if time.time() - self.start_time > self.deadline:
                break
                
            # Sample candidates
            # Compute Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite
                C = np.eye(d)
                L = np.eye(d)
                
            # Sample standard normals
            Z = rng.standard_normal((lam, d))
            # Transform: x = m + sigma * (Z @ L.T)
            Y = Z @ L.T
            X = m[None, :] + sigma * Y
            X = np.clip(X, lo, hi)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            remaining_budget -= lam
            
            # Update best found globally
            min_idx = np.argmin(F)
            if F[min_idx] < best_f_cma:
                best_f_cma = F[min_idx]
                best_x_cma = X[min_idx].copy()
                
            # Sort indices by fitness
            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            F_sorted = F[sort_idx]
            
            # Update mean m
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution path ps
            # We need C^{-1/2}.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small eigenvalues
            eigvals = np.clip(eigvals, 1e-16, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            step = (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            # Update covariance C
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (d + 1)) * chi_n else 0
            
            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc)
                
            # Add rank-mu update
            Y_delta = X_sorted[:mu] - m # Use old m for delta
            Y_delta = Y_delta / sigma
            rank_mu_term = np.sum(weights[:, None, None] * Y_delta[:, :, None] * Y_delta[:, None, :], axis=0)
            C += c_mu * rank_mu_term
            
            # Enforce symmetry and positive definiteness (regularization)
            C = (C + C.T) / 2.0
            # Simple regularization if needed
            if not np.all(np.linalg.eigvalsh(C) > 0):
                C = C + 1e-4 * np.eye(d)
                
            m = m_new
            
        return best_x_cma, best_f_cma
