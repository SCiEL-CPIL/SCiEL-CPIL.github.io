"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Model-and-optimize with CMA-ES polish for Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)

        N = self.dim
        bounds_low = -5.0
        bounds_high = 5.0

        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")

            x = np.clip(x, bounds_low, bounds_high)

            val = func(x)
            self.calls += 1

            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()

            return val

        # MODEL-AND-OPTIMIZE STRATEGY
        # Estimate x* using N+1 points to solve linear system
        D = 4.0
        points = [np.zeros(N)]
        for i in range(N):
            p = np.zeros(N)
            p[i] = D
            points.append(p)

        values = []
        for p in points:
            try:
                v = wrapped_func(p)
                values.append(v)
            except (ValueError, TimeoutError, Exception):
                break

        # Fallback if model estimation fails
        if len(values) < N + 1:
            rng = np.random.default_rng(42)
            for _ in range(10):
                if self.calls >= self.budget: break
                x = rng.uniform(bounds_low, bounds_high, N)
                try:
                    wrapped_func(x)
                except Exception:
                    break
            return (self.best_x, self.best_f)

        # Solve for x*
        A = np.zeros((N, N))
        b = np.zeros(N)

        x0 = points[0]
        f0 = values[0]

        for i in range(1, N + 1):
            xi = points[i]
            fi = values[i]

            diff = xi - x0
            norm_sq_diff = np.dot(xi, xi) - np.dot(x0, x0)
            val_diff = fi - f0

            A[i-1] = 2.0 * diff
            b[i-1] = norm_sq_diff - val_diff

        try:
            x_est = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_est = np.zeros(N)

        x_start = np.clip(x_est, bounds_low, bounds_high)

        # Evaluate start point if not already done
        try:
            if not np.allclose(x_start, self.best_x):
                wrapped_func(x_start)
        except Exception:
            pass

        # CMA-ES POLISH PHASE
        # Implement lightweight CMA-ES for remaining budget
        if self.calls < self.budget and time.time() < deadline:
            try:
                self._run_cmaes(wrapped_func, x_start, N)
            except Exception:
                pass

        return (self.best_x, self.best_f)

    def _run_cmaes(self, func, x0, dim):
        """Lightweight CMA-ES implementation."""
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial step size
        sigma = 1.0
        
        # Mean
        mean = x0.copy()
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Precompute matrices
        B, D = np.linalg.eigh(C)
        Cinv = B @ np.diag(1.0 / D) @ B.T
        
        # Step size control
        cs = (lambda_ + 2) ** -0.5
        ds = 1 + 2 * max(0, cs - 1 / (lambda_ + 2)) + cs
        hsig = np.sqrt((lambda_ + 2 - 2) / (dim + lambda_ + 2 - 2))
        
        # Evolution path for sigma
        ps = np.zeros(dim)
        
        # Evolution path for C
        pc = np.zeros(dim)
        
        # Damping parameters
        cmu = 2 / ((dim + 1.3)**2 + lambda_)
        cc = (4 + cmu - dim) / (dim + 4 - cmu)
        
        # Weights
        w = np.log(np.log(lambda_) + 1) - np.log(np.arange(1, lambda_ + 1))
        w = w / w.sum()
        mu = int(np.floor(lambda_ / 2))
        mueff = np.sum(w[:mu]) ** 2 / np.sum(w[:mu] ** 2)
        
        # Main loop
        while self.calls < self.budget and time.time() < deadline - 5:
            # Generate population
            offspring = []
            for i in range(lambda_):
                z = np.random.randn(dim)
                # Transform z using C^{1/2}
                y = B @ (D * z)
                x = mean + sigma * y
                x = np.clip(x, -5.0, 5.0)
                offspring.append(x)
            
            # Evaluate offspring
            fvals = []
            for x in offspring:
                try:
                    f = func(x)
                    fvals.append(f)
                except (ValueError, TimeoutError):
                    break
            
            if len(fvals) < lambda_:
                break
            
            # Sort by fitness
            indices = np.argsort(fvals)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_fvals = [fvals[i] for i in indices]
            
            # Update best
            if sorted_fvals[0] < self.best_f:
                self.best_f = sorted_fvals[0]
                self.best_x = sorted_offspring[0].copy()
            
            # Update mean
            new_mean = np.sum(w[:mu, None] * np.array(sorted_offspring[:mu]), axis=0)
            
            # Update evolution path for sigma
            sigma_inv = 1.0 / sigma
            ps = (1 - cs) * ps + cs * sigma_inv * (B.T @ np.diag(1.0 / D) @ (new_mean - mean))
            
            # Update step size
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (self.calls // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update evolution path for C
            pc = (1 - cc) * pc + cc * (new_mean - mean) / sigma
            
            # Update covariance
            delta_h = (1 + 2 * max(0, np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (self.calls // lambda_)) / chi_n - 1) * cc * (2 - cc))
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(w[:mu, None, None] * np.outer(np.array(sorted_offspring[:mu]) - mean, np.array(sorted_offspring[:mu]) - mean) / (sigma ** 2), axis=0)
            
            C = (1 + cmu * (delta_h - 1) - cc * (1 - (1 - cc) ** (self.calls // lambda_))) * C + cmu * (rank_one + rank_mu)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                B, D = np.linalg.eigh(C)
                D = np.maximum(D, 1e-20)
                Cinv = B @ np.diag(1.0 / D) @ B.T
            except np.linalg.LinAlgError:
                break
            
            mean = new_mean
        
        return (self.best_x, self.best_f)

# EVOLVE-BLOCK-END
