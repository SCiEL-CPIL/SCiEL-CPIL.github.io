import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin

        rng = np.random.default_rng(42)

        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # --- Phase 1: Global Exploration with LHS ---
        # Allocate a significant portion to global search to locate the basin.
        # BBOB-21 (Gallagher 101 peaks) is highly multimodal.
        n_global = int(budget * 0.4)
        n_global = max(n_global, 50)
        
        # Generate LHS points in [-5, 5]^dim
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        # Evaluate global points
        f_values = np.zeros(n_global)
        for i in range(n_global):
            if time.time() > deadline or calls >= budget:
                break
            f_values[i] = func(lhs[i])
            calls += 1
            
        if calls == 0:
            return np.zeros(dim), 0.0

        # Sort by fitness
        sorted_indices = np.argsort(f_values)
        best_x = lhs[sorted_indices[0]].copy()
        best_f = f_values[sorted_indices[0]]

        # --- Phase 2: Local Refinement with CMA-ES ---
        # We take the top K candidates from the global search and run CMA-ES on them.
        # CMA-ES adapts the covariance matrix to handle correlated variables and
        # non-separable landscapes, which is crucial for BBOB-21.
        
        n_local_candidates = 3
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget per local search
        budget_per_local = max(20, remaining_budget // n_local_candidates)
        
        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            # Start from the k-th best global sample
            x = lhs[sorted_indices[k]].copy()
            f = f_values[sorted_indices[k]]
            
            # CMA-ES Parameters
            # Population size
            lambda_ = 10
            # Initial sigma
            # Heuristic: 0.5 * (upper - lower) / dim? No, usually 0.5 or 1.0 for box constraints.
            # For [-5, 5], sigma=0.5 is reasonable.
            sigma = 0.5 
            # Mean
            mean = x.copy()
            # Covariance matrix
            C = np.eye(dim)
            # Inverse square root of C (for sampling)
            # Using eigen decomposition for stability
            evals, evecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            evals = np.maximum(evals, 1e-10)
            C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            
            # Strategy parameters
            mu = lambda_ // 2  # Number of parents
            weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            # Step size control parameters
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            hsig_threshold = 1.4 + 2 / (dim + 1)
            
            # Covariance adaptation parameters
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            # State variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            hsig = 0.0
            n_evals_cma = 0
            max_local_evals = budget_per_local
            
            # Track best within this CMA-ES run
            local_best_x = x.copy()
            local_best_f = f
            
            while n_evals_cma < max_local_evals and calls < budget and time.time() < deadline:
                # Generate population
                ys = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    # Sample from N(0, C)
                    z = rng.standard_normal(dim)
                    y = C_inv_sqrt @ z
                    ys[i] = y
                
                # Update p_sigma
                norm_y = np.sqrt(np.mean(np.sum(ys[:mu]**2, axis=1)))
                if norm_y > 0:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt.T @ np.mean(ys[:mu], axis=0)) / norm_y
                else:
                    p_sigma = (1 - c_sigma) * p_sigma
                
                # Update h_sigma
                hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (n_evals_cma + 1))) < hsig_threshold
                
                # Update p_c
                if hsig:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt.T @ np.mean(ys[:mu], axis=0))
                else:
                    p_c = (1 - c_c) * p_c
                
                # Update Covariance Matrix C
                v = p_c / np.sqrt(1 - (1 - c_c)**(2 * (n_evals_cma + 1)))
                rank_one = (1 - c_1) * C + c_1 * (p_c @ p_c + (1 - hsig) * c_c * (2 - c_c) * v @ v - C)
                
                # Rank-mu update
                diff_ys = ys[:mu] - np.mean(ys[:mu], axis=0, keepdims=True)
                # We need to transform back to x-space for the rank-mu update? 
                # Standard CMA-ES updates C in the y-space (standard normal space).
                # The formula is: C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + ...) + c_mu * sum w_i y_i y_i^T
                # But wait, the standard formulation updates C such that y ~ N(0, C).
                # So yes, the update is in y-space.
                
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(ys[i], ys[i])
                
                C_new = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - hsig) * c_c * (2 - c_c) * v @ v - C) + c_mu * rank_mu
                
                # Ensure symmetry
                C_new = (C_new + C_new.T) / 2
                
                # Numerical stability: Eigenvalue clipping
                evals, evecs = np.linalg.eigh(C_new)
                # Clip eigenvalues to a reasonable range
                evals = np.clip(evals, 1e-12, 1e12)
                C = evecs @ np.diag(evals) @ evecs.T
                C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim) - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Generate new x candidates
                xs = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    xs[i] = mean + sigma * ys[i]
                    xs[i] = np.clip(xs[i], -5, 5)
                
                # Evaluate
                fs = np.zeros(lambda_)
                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break
                    fs[i] = func(xs[i])
                    calls += 1
                    n_evals_cma += 1
                    
                    if fs[i] < local_best_f:
                        local_best_f = fs[i]
                        local_best_x = xs[i].copy()
                    
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = xs[i].copy()
                
                # If we broke out of the evaluation loop early, break the main loop
                if calls >= budget or time.time() > deadline:
                    break
                
                # Update mean
                mean = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
                
                # Convergence check
                if sigma < 1e-8:
                    break

        return best_x, best_f
