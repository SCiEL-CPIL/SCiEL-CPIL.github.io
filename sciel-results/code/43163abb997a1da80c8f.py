"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Single-run high-fidelity CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 325.0  # Safety margin for 330s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return (time.time() - start_time) > (deadline - 5.0)

        # 1. Initial Random Search
        # Use a portion of the budget to find a good starting point.
        # For dim=5, 50 samples is usually enough to find a decent basin.
        init_budget = min(50, max(10, budget // 10))
        
        for i in range(init_budget):
            if check_deadline() or calls_made >= budget:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = rng.uniform(-5.0, 5.0, dim)
            best_f = func(x)
            best_x = x.copy()
            calls_made = 1

        # 2. CMA-ES
        remaining = budget - calls_made
        if remaining > 10:
            n = dim
            # CMA-ES Parameters
            mu = max(1, n // 2) 
            lam = 2 * mu  # Population size
            
            if remaining >= lam:
                # Weights
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights**2)
                
                # Learning rates
                c_sigma = (mueff + 2) / (n + mueff + 5)
                c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
                c_1 = 2 / ((n + 1.3)**2 + mueff)
                c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

                # State variables
                x_mean = best_x.copy()
                sigma = 1.0 
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                
                # Precompute
                B, D = np.linalg.eigh(C)
                D_inv = 1.0 / np.sqrt(D)
                
                # Main Loop
                while (budget - calls_made) >= lam and not check_deadline():
                    # Generate population
                    Z = rng.standard_normal((lam, n))
                    Y = x_mean + sigma * (B @ (D_inv * (B.T @ Z))).T
                    Y = np.clip(Y, -5, 5)
                    
                    F = np.array([func(y) for y in Y])
                    calls_made += lam
                    
                    # Update best
                    idx_best = np.argmin(F)
                    if F[idx_best] < best_f:
                        best_f = F[idx_best]
                        best_x = Y[idx_best].copy()

                    # Sort
                    order = np.argsort(F)
                    Y_sorted = Y[order]
                    F_sorted = F[order]
                    
                    # Update mean
                    x_mean_old = x_mean.copy()
                    x_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

                    # Update paths
                    y_vec = (x_mean - x_mean_old) / (sigma * chi_n)
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_vec
                    hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lam) / lam)) <
                            (1.4 + 2 / (n + 1)) * chi_n)
                    
                    pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_vec

                    # Update C
                    delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                    C = (1 - c_1 - c_mu) * C
                    C += c_1 * (np.outer(pc, pc) + delta_hsig * C - np.eye(n))
                    C += c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x_mean_old, Y_sorted[:mu] - x_mean_old), axis=0) / sigma**2
                    C = (C + C.T) / 2

                    # Update sigma
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                    
                    # Recompute eigen
                    B, D = np.linalg.eigh(C)
                    D = np.clip(D, 1e-10, None)
                    D_inv = 1.0 / np.sqrt(D)

        # 3. Local Refinement with (1+1)-ES if budget remains
        remaining = budget - calls_made
        if remaining > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(remaining):
                if check_deadline():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
