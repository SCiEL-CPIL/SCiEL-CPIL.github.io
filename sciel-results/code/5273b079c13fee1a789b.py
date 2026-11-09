# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """GP-guided search with local CMA-ES refinement for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # Data storage for GP
        X_data = []
        F_data = []

        def add_data(x, f):
            X_data.append(x)
            F_data.append(f)

        # GP kernel: Matern 3/2
        def matern32_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
            X1 = np.atleast_2d(X1)
            X2 = np.atleast_2d(X2)
            d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            d2 = np.maximum(d2, 0.0)
            r = np.sqrt(d2) / length_scale
            K = (sigma_f**2) * (1 + r + r**2 / 3) * np.exp(-r)
            return K

        def gp_predict(X_test, X_train, y_train, length_scale=1.0, sigma_f=1.0, noise=1e-6):
            K = matern32_kernel(X_train, X_train, length_scale, sigma_f)
            K += noise * np.eye(len(X_train))
            K_star = matern32_kernel(X_test, X_train, length_scale, sigma_f)
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
            K_star_inv = K_star @ K_inv
            y_mean = np.mean(y_train)
            mu = y_mean + K_star_inv @ (y_train - y_mean)
            K_ss = matern32_kernel(X_test, X_test, length_scale, sigma_f)
            var = np.diag(K_ss) - np.sum(K_star_inv**2, axis=1)
            var = np.maximum(var, 1e-12)
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, best_f_current, xi=0.01):
            imp = best_f_current - mu - xi
            sigma = np.maximum(sigma, 1e-8)
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0.0)

        # Phase 1: Initial random sampling to build GP model
        n_init = min(30, budget)
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = evaluate(x)
            if fx is None:
                break
            add_data(x, fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return None, np.inf

        # GP parameters
        length_scale = 1.5
        sigma_f = max(1.0, (max(F_data) - min(F_data)) / 2.0 if len(F_data) > 1 else 1.0)

        # Phase 2: GP-guided search with local CMA-ES refinement
        gp_step = 0
        cma_step = 0
        use_gp = True
        gp_interval = 5  # Every 5 GP steps, do CMA-ES refinement
        cma_population = 10

        while calls < budget and time.time() < deadline:
            if use_gp:
                # GP-guided exploration
                if len(X_data) < 5:
                    x = rng.uniform(-4.5, 4.5, dim)
                else:
                    # Generate candidate points
                    n_candidates = 50
                    X_cand = rng.uniform(-4.5, 4.5, (n_candidates, dim))
                    
                    # Also add points around current best
                    if best_x is not None:
                        X_around_best = best_x + rng.normal(0, 0.5, (20, dim))
                        X_around_best = np.clip(X_around_best, -4.5, 4.5)
                        X_cand = np.vstack([X_cand, X_around_best])
                    
                    X_train = np.array(X_data)
                    y_train = np.array(F_data)
                    
                    mu, sigma = gp_predict(X_cand, X_train, y_train, length_scale, sigma_f)
                    ei = expected_improvement(mu, sigma, best_f, xi=0.005)
                    
                    # Select best candidate by EI
                    best_idx = np.argmax(ei)
                    x = X_cand[best_idx]
                
                fx = evaluate(x)
                if fx is None:
                    break
                add_data(x, fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                gp_step += 1
                if gp_step >= gp_interval:
                    use_gp = False
                    gp_step = 0
            else:
                # Local CMA-ES refinement around best point
                if best_x is None:
                    use_gp = True
                    continue
                
                # Simple CMA-ES: generate population around best
                sigma_local = 0.3
                offspring = []
                fitnesses = []
                
                for i in range(cma_population):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = rng.standard_normal(dim)
                    xi = best_x + sigma_local * y
                    xi = np.clip(xi, -5.0, 5.0)
                    fx = evaluate(xi)
                    if fx is None:
                        break
                    offspring.append(xi)
                    fitnesses.append(fx)
                    if fx < best_f:
                        best_f = fx
                        best_x = xi.copy()
                
                if len(offspring) < cma_population:
                    use_gp = True
                    cma_step = 0
                    continue
                
                # Update best_x to weighted mean of best half
                indices = np.argsort(fitnesses)
                mu_half = len(offspring) // 2
                weights = np.ones(mu_half) / mu_half
                best_x_new = np.sum([weights[i] * offspring[indices[i]] for i in range(mu_half)], axis=0)
                
                # Only move if improved
                if fitnesses[indices[0]] < best_f + 1e-10:
                    best_x = best_x_new
                
                cma_step += 1
                if cma_step >= 3:
                    use_gp = True
                    cma_step = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
