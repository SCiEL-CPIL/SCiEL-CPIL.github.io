import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320  # Seconds, evaluator kills at 330, leave margin
        n = self.dim
        lam = 10  # Standard population size for n=5
        
        # Check if budget allows at least one full generation
        if lam > self.budget:
            lam = max(2, self.budget // 2)

        # Global best tracking
        best_x_global = np.zeros(n)
        best_f_global = float('inf')
        
        # Initial evaluation at 0
        fx0 = func(np.zeros(n))
        if fx0 < best_f_global:
            best_f_global = fx0
            best_x_global = np.zeros(n)
        
        calls = 1
        
        # Time check helper
        def time_ok():
            return (time.time() - start_time) < time_limit

        # Restart loop
        max_restarts = 5
        current_restart = 0
        rng = np.random.default_rng(42)
        
        while current_restart < max_restarts and calls < self.budget and time_ok():
            # Initialize CMA-ES parameters
            # Start from the global best
            x = best_x_global.copy()
            sigma = 2.0 # Larger initial sigma for exploration in restarts
            
            # Initialize covariance matrix to identity
            C = np.eye(n)
            # Eigen-decomposition for sampling
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))

            # Cumulation paths
            pc = np.zeros(n)
            ps = np.zeros(n)
            # Weighting
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            # Learning rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

            # Boundary handling: clip
            def clip(y):
                return np.clip(y, -5, 5)

            # Initial evaluation of the restart start point (if not already done)
            # If current_restart > 0, x is best_x_global which was already evaluated.
            # But CMA-ES needs fx for the mean? No, CMA-ES updates based on offspring.
            # We need to ensure the first generation is valid.
            
            fx = best_f_global # We know the value of x
            # However, CMA-ES internal state assumes x is the current mean.
            # We will generate offspring from x.
            
            gen_in_restart = 0
            best_f_restart = fx
            best_x_restart = x.copy()
            stall_gens = 0
            threshold = 50 # Generations to wait before restart

            while calls < self.budget and time_ok():
                gen_in_restart += 1
                if gen_in_restart > 1000: break # Safety

                # Sample population
                offspring = []
                fvals = []
                
                # Generate candidates
                for i in range(lam):
                    if calls >= self.budget:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = clip(y)
                    offspring.append(y)

                # Evaluate
                improved = False
                for y in offspring:
                    if calls >= self.budget:
                        break
                    fy = func(y)
                    calls += 1
                    fvals.append(fy)
                    if fy < best_f_global:
                        best_f_global = fy
                        best_x_global = y.copy()
                    if fy < best_f_restart:
                        best_f_restart = fy
                        best_x_restart = y.copy()
                        improved = True

                if len(fvals) < lam:
                    break

                if improved:
                    stall_gens = 0
                else:
                    stall_gens += 1

                # Sort by fitness
                order = np.argsort(fvals)
                offspring = [offspring[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update mean
                x_new = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)

                # Update covariance matrix
                if sigma < 1e-10:
                    sigma = 1e-10
                    
                y_inv = np.linalg.solve(C, (x_new - x) / sigma)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_inv
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen_in_restart))
                        < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_inv

                # Update C
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                rank_one = np.outer(pc, pc)
                rank_mu = np.sum([
                    weights[i] * np.outer(
                        np.linalg.solve(C, (offspring[i] - x) / sigma),
                        np.linalg.solve(C, (offspring[i] - x) / sigma)
                    ) for i in range(mu)
                ], axis=0)

                C = (1 + c1 * delta_hsig + cmu - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += (1e-10 - np.min(eigvals)) * np.eye(n)

                # Update step size (CSA)
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
                sigma = np.clip(sigma, 1e-8, 100)

                # Update mean
                x = x_new

                # Update eigen-decomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))

                # Check restart conditions
                if sigma < 1e-6 or stall_gens > threshold:
                    break

            current_restart += 1

        return best_x_global, best_f_global
