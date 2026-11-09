"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive CMA-ES optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Helper to evaluate and track
        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # CMA-ES parameters
        # For dim=5, lambda = 4 + floor(3*log(dim)) = 4 + 9 = 13
        lam = 4 + int(3 * np.log(dim))
        
        # Initial sigma: heuristic based on box size
        # BBOB f4 has oscillations. A moderate initial step size.
        sigma = 2.0
        
        # Mean
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Step sizes for evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        # Standard CMA-ES rates
        mu = lam // 2  # selection mass
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Precompute constants
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial population
        def sample_population():
            nonlocal m, C, sigma
            # Generate standard normal samples
            Y = rng.standard_normal((lam, dim))
            # Transform to X = m + sigma * B @ Y
            # B is the square root of C
            B = np.linalg.cholesky(C)
            X = m[None, :] + sigma * (B @ Y.T).T
            return X, Y

        # Main loop
        while call_count < budget and time.time() < deadline:
            # Sample population
            X, Y = sample_population()
            
            # Evaluate
            F = np.array([eval_func(x) for x in X])
            
            # Check for NaN/Inf
            valid = np.isfinite(F)
            if not np.all(valid):
                # If some are invalid, replace them with best + small noise
                # Or just break if too many invalid
                if np.sum(valid) < mu:
                    break
                # For simplicity, if any invalid, we might need to handle ranking carefully.
                # Standard CMA-ES assumes finite values.
                # We'll replace invalids with a large value to push them to the end of sorting.
                F[~valid] = np.inf
            
            # Sort
            indices = np.argsort(F)
            x_sorted = X[indices]
            f_sorted = F[indices]
            y_sorted = Y[indices]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
            
            # Selection
            x_selected = x_sorted[:mu]
            y_selected = y_sorted[:mu]
            
            # Update mean
            m_new = m + np.sum(weights[:, None] * x_selected, axis=0)
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * np.dot(B.T, np.dot(weights, y_selected.T)).T[0]
            
            # Check hsig condition
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**((call_count + 1) / (cs * dim))) < (1.4 + 2 / (dim + 1)) * chiN:
                pc = (1 - cc) * pc + cc / sigma * (np.dot(B.T, m_new) - np.dot(B.T, m))
                hsig = 1.0
            else:
                hsig = 0.0
            
            # Update covariance
            v = 1 / sigma * (m_new - m)
            C = (1 + c1 * hsig * (cc - cc - 1) + cmu * hsig - c1) * C \
                + c1 * (hsig * (pc * pc.T + (1 - cc) * (1 - hsig) * C) - cc * pc * pc.T) \
                + cmu * np.sum(weights[:, None] * (x_selected - m)[:, :, None] * (x_selected - m)[:, None, :], axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness (add small identity if needed)
            try:
                np.linalg.cholesky(C)
            except:
                C = C + 1e-4 * np.eye(dim)
            
            # Update sigma (Adaptation rule)
            # Simple adaptation: if progress is slow, decrease sigma; if fast, increase?
            # Standard CMA-ES uses a target value for the norm of ps.
            # Here we use a simpler heuristic or the standard one.
            # Standard: sigma = sigma * exp(cs / (5 * dim) * (||ps||/chiN - 1))
            sigma = sigma * np.exp(cs / (5 * dim) * (np.linalg.norm(ps) / chiN - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update state
            m = m_new
            C = C
            
            # Convergence check: if simplex size is very small
            if sigma < 1e-8:
                break
                
            # Optional: Restart if stuck? 
            # For now, rely on the global nature of CMA-ES with appropriate sigma.
            # If we detect no improvement for a long time, we could randomize m.
            # Let's add a simple stagnation detection.
            # If best_f hasn't improved in last 50 calls, perturb m.
            if call_count % 50 == 0:
                # Check if we are stuck
                pass # Simple heuristic omitted for now to keep code clean, CMA-ES usually handles this.

        return best_x, best_f

# EVOLVE-BLOCK-END
