import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~9 for dim=5
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Cholesky factor of C
        Cholesky_C = np.eye(dim)
        
        best_x = x_mean
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal calls
            calls += 1
            return func(x)
        
        # Initial evaluation
        # We can start with a random point or center. Center is good for Sphere.
        # But let's do a small random perturbation to avoid symmetry issues if optimum is exactly 0?
        # Actually, for Sphere, 0 is a very good start if shift is small.
        # The problem says minimizer in [-4, 4].
        
        # Let's do a quick random search for the first few calls to get a decent start?
        # No, CMA-ES handles the global aspect reasonably well if sigma is large.
        
        # Initial population
        ys = []
        fs = []
        for _ in range(lam):
            y = x_mean + sigma * (Cholesky_C @ np.random.randn(dim))
            y = np.clip(y, -5, 5)
            fy = evaluate(y)
            ys.append(y)
            fs.append(fy)
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                
        # Sort
        order = np.argsort(fs)
        ys = np.array([ys[i] for i in order])
        fs = np.array([fs[i] for i in order])
        
        # Main loop
        while calls < budget:
            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            
            # Update covariance matrix
            y = (x_new_mean - x_mean) / sigma
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) * y
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cholesky_C @ y
            
            delta_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls + 1) / (dim * damps))) - 1)
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls + 1) / (dim * damps))) - np.sqrt(dim)))
            
            # Rank 1 update
            C = (1 + c1 * (mueff - 1) + c1) * C - c1 * np.outer(y, y)
            # Rank mu update
            C += cmu * np.sum(weights[:mu] * np.outer(ys[:mu] - x_mean, ys[:mu] - x_mean) / sigma**2, axis=0)
            
            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            # Re-calculate Cholesky
            try:
                Cholesky_C = np.linalg.cholesky(C)
            except:
                # If not PD, add small identity
                C = C + 1e-6 * np.eye(dim)
                Cholesky_C = np.linalg.cholesky(C)
                
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5) # Keep mean in bounds? CMA-ES usually doesn't clip mean, but here we are bounded.
            
            # Generate new population
            ys = []
            fs = []
            for _ in range(lam):
                y = x_mean + sigma * (Cholesky_C @ np.random.randn(dim))
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                ys.append(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
            order = np.argsort(fs)
            ys = np.array([ys[i] for i in order])
            fs = np.array([fs[i] for i in order])
            
            # Termination check (optional, budget is hard limit)
            if sigma < 1e-12:
                break
                
        return best_x, best_f
