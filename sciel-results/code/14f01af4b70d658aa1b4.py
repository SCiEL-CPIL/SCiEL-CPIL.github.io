"""CMA-ES optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f11.
    Adapts the covariance matrix to the discus shape, which should outperform
    fixed-direction or simple isotropic search methods.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        X = []
        Y = []
        evals = 0

        def _eval(x):
            nonlocal evals
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = func(x)
            X.append(x.copy())
            Y.append(float(f))
            evals += 1
            return f

        # --- Phase 1: Initial exploration to find a good starting point ---
        # Center
        _eval(np.zeros(dim))

        # Axis probes: both directions, several magnitudes
        for i in range(dim):
            for d in [1.0, 2.0, 3.0, 4.0]:
                for sign in [1.0, -1.0]:
                    if evals >= budget:
                        break
                    p = np.zeros(dim)
                    p[i] = sign * d
                    _eval(p)
                if evals >= budget:
                    break
            if evals >= budget:
                break

        # Random points for diversity
        n_rand = min(25, budget - evals)
        for _ in range(n_rand):
            if evals >= budget:
                break
            _eval(rng.uniform(-4.5, 4.5, dim))

        if evals >= budget:
            arr = np.array(X)
            yarr = np.array(Y)
            bi = int(np.argmin(yarr))
            return arr[bi], yarr[bi]

        # --- Phase 2: CMA-ES ---
        # Initialize CMA-ES with the best point found so far
        best_idx = int(np.argmin(Y))
        x_mean = np.array(X[best_idx])
        sigma = 1.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        Cinv = np.eye(dim)  # Inverse covariance matrix
        
        # Parameters
        lambda_pop = 4 + int(3 * np.log(dim))  # Population size
        mu = lambda_pop // 2  # Number of selected points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Loop
        while evals < budget and time.time() < deadline:
            # Generate population
            population = []
            for i in range(lambda_pop):
                # Sample from multivariate normal
                # Using Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset
                    C = np.eye(dim)
                    Cinv = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    L = np.linalg.cholesky(C)
                
                z = rng.normal(size=dim)
                x_new = x_mean + sigma * (L @ z)
                population.append(x_new)
            
            # Evaluate population
            f_values = []
            for x in population:
                f = _eval(x)
                f_values.append(f)
                if evals >= budget:
                    break
            
            if len(f_values) < lambda_pop:
                break
            
            # Sort by fitness
            indices = np.argsort(f_values)
            population = [population[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * population[i]
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (Cinv @ (x_new_mean - x_mean))
            
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals - len(X)))) > 4.4 * chi_n + 4:
                sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            h_sig = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (evals - len(X))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (population[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * (h_sig - 1) - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            
            # Update inverse covariance
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                Cinv = np.eye(dim)
                C = np.eye(dim)
            
            # Update mean
            x_mean = x_new_mean
            
            # Check convergence
            if sigma < 1e-12:
                break

        # Final best
        all_x = np.array(X)
        all_y = np.array(Y)
        bi = int(np.argmin(all_y))
        best_x_final = all_x[bi]
        best_f_final = all_y[bi]

        return best_x_final, best_f_final
# EVOLVE-BLOCK-END
