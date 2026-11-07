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

        def cma_es(mean, sigma, max_generations):
            """Run CMA-ES from mean with initial sigma."""
            n = dim
            # Population size
            lam = int(4 + 3 * np.log(n))
            mu = lam // 2
            
            # Weights for recombination
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            
            # Variance-effective population size
            mueff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # Initial state
            C = np.eye(n)
            Csqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            # Initial population
            z = np.random.randn(lam, n)
            pop = mean + sigma * (z @ Csqrt.T)
            fvals = np.array([safe_call(x) for x in pop])
            
            for gen in range(max_generations):
                if time.time() > deadline or call_count[0] >= budget:
                    break
                
                # Sort population
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                
                # Update best
                # safe_call already updates best_x, best_f
                
                # Check convergence
                if np.std(fvals[:mu]) < 1e-12 * (1 + np.abs(fvals[0])):
                    break
                    
                # Recombination
                w = weights
                new_mean = np.dot(w, pop[:mu])
                
                # Update step size control
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (Csqrt @ (new_mean - mean))
                
                # Hessian drift control
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**((gen + 1) * cs)) < (1.4 + 2 / (n + 1)) * chi_n)
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma
                
                # Rank-one and rank-mu updates
                delta_sigma = np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = sigma * delta_sigma
                
                # Update covariance matrix
                C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C)
                C += cmu * np.dot(w[:mu], (pop[:mu] - mean) / sigma).T @ (np.dot(w[:mu], (pop[:mu] - mean) / sigma))
                
                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                try:
                    Csqrt, _ = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                except np.linalg.LinAlgError:
                    # Fallback if not positive definite
                    C = np.eye(n)
                    Csqrt = np.eye(n)
                    sigma *= 0.5
                    
                mean = new_mean
                
                # Generate new population
                z = np.random.randn(lam, n)
                pop = mean + sigma * (z @ Csqrt.T)
                fvals = np.array([safe_call(x) for x in pop])
                
                # If sigma becomes too small or too large, adjust
                if sigma < 1e-15:
                    break

            return mean, sigma

        # Initialize CMA-ES
        # Start from center or best of a few random points
        rng = np.random.default_rng(42)
        
        # Small screening to find a good start
        n_screen = 10
        screen_points = []
        for _ in range(n_screen):
            x = rng.uniform(-4.0, 4.0, dim)
            f = safe_call(x)
            screen_points.append((f, x))
            if call_count[0] >= budget:
                break
                
        if screen_points:
            screen_points.sort(key=lambda t: t[0])
            start_mean = screen_points[0][1]
        else:
            start_mean = np.zeros(dim)
            
        start_sigma = 2.0
        
        # Run CMA-ES
        # Estimate generations: budget / lam
        lam = int(4 + 3 * np.log(dim))
        max_gens = max(10, (budget - call_count[0]) // lam)
        
        cma_es(start_mean, start_sigma, max_gens)
        
        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
