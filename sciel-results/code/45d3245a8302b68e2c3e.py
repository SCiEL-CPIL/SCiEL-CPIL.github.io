"""CMA-ES optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    This is a structural change from local geometric search (Nelder-Mead) 
    to population-based adaptive search. CMA-ES is robust to discontinuities 
    and non-smooth landscapes like the Step Ellipsoidal function by adapting 
    the search distribution based on successful samples, allowing it to 
    "jump" over flat regions and refine in complex basins.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, keep 15s safety margin
        budget = self.budget
        dim = self.dim

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            try:
                x = np.asarray(x, dtype=float)
                x = np.clip(x, -5.0, 5.0)
                f = func(x)
                f = float(f)
            except Exception:
                return np.inf
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # --- CMA-ES Parameters ---
        # Population size
        lam = int(4 + 3 * np.log(dim))  # Standard recommendation
        mu = lam // 2  # Number of selected individuals
        
        # Weights for the selected individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)  # Equivalent population size for update
        
        # Learning rates
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0)**2 + mueff))
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute Cholesky factor
        # C = B B^T / D^2 ? No, C is the covariance matrix.
        # We need to sample x = x_mean + sigma * B * z / D ?
        # Standard CMA-ES: x = x_mean + sigma * B * z
        # Where C = B B^T.
        
        # Initial evaluation
        f_mean = safe_call(x_mean)
        
        # Main loop
        while calls_made < budget and time.time() < deadline:
            # 1. Sample population
            xs = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Sample from N(0, C) using B and D
                # C = B B^T, so sqrt(C) = B * D^{-1}? 
                # Actually, if C = B B^T, then sampling is B @ z.
                # But standard CMA-ES maintains C such that C = B B^T / D^2?
                # Let's stick to the standard formulation:
                # C is the covariance matrix.
                # We need the Cholesky decomposition or eigen decomposition to sample.
                # For simplicity and speed in 5D, we can use np.linalg.eigh or just keep C and sample via Cholesky.
                
                # However, standard CMA-ES updates C directly.
                # To sample, we need a square root of C.
                # Let's use Cholesky decomposition each iteration (cheap for 5x5).
                try:
                    L = np.linalg.cholesky(C)
                    x_sample = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reinitialize or add small identity
                    C = C + 1e-10 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                    x_sample = x_mean + sigma * (L @ z)
                
                xs.append(x_sample)
            
            # 2. Evaluate population
            fs = [safe_call(x) for x in xs]
            
            # 3. Sort by fitness
            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # 4. Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * xs[i]
            x_mean_new = x_mean + sigma * y_w
            
            # 5. Update covariance matrix
            # Rank-one update
            y_c = np.zeros(dim)
            for i in range(mu):
                y_c += w[i] * (xs[i] - x_mean) / sigma
            
            # Path covariance
            pc = (1.0 - cc) * pc + cc * (np.sqrt(cc * (2.0 - cc) * mueff) / sigma) * (x_mean_new - x_mean)
            
            # Path step size
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (C @ (x_mean_new - x_mean) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs)**(2 * (calls_made - 1)))) / \
                   (np.sqrt(1.0 - (1.0 - cc)**(2 * (calls_made - 1))) * np.trace(C) / dim)
            
            # Update C
            delta_sig = 1.0 + 2.0 * max((hsig - 1.0) / (dim + 2.0), 0.0)
            
            C = (1.0 + c1 * (cc * (2.0 - cc) * mueff / dim - 1.0) + cmu * np.sum(w[:mu]**2) - c1) * C + \
                c1 * np.outer(pc, pc) + cmu * np.sum([w[i] * np.outer((xs[i] - x_mean) / sigma, (xs[i] - x_mean) / sigma) for i in range(mu)], axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            C += 1e-12 * np.eye(dim)
            
            # Update sigma
            sigma = sigma * delta_sig * np.exp(cs / (dim + 2.0) * (np.linalg.norm(ps)**2 / dim - 1.0))
            
            # Update mean
            x_mean = x_mean_new
            
            # Convergence check
            if sigma < 1e-12:
                break
            if np.linalg.norm(x_mean_new - x_mean) < 1e-12:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
