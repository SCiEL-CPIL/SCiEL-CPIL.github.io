"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using MO-CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Track all evaluated points
        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            # Clip to bounds before evaluating
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=np.float64)
            eval_count += 1
            all_X.append(x_clipped)
            all_F.append(f)
            return f

        # MO-CMA-ES Parameters
        # Population size lambda. For dim=4, standard lambda is 4 + floor(3 * log(4)) = 8.
        # However, with a budget of 200, we can afford a larger population for better sampling.
        # Let's use lambda = 20 to allow for ~10 generations (20 init + 10*20 offspring = 220? No, budget is 200).
        # Init: 20.
        # Gen 1: 20 offspring. Total 40.
        # ...
        # Gen 9: 20 offspring. Total 200.
        # So 9 generations with lambda=20 fits exactly.
        lambda_pop = 20
        
        # Initial mean: Center of the box
        mean = (lo + hi) / 2.0
        
        # Initial step size: 1/5 of the span is a reasonable start, or smaller if bounds are tight.
        # CMA-ES adapts this.
        sigma = np.mean(span) / 5.0
        
        # Covariance matrix: Identity initially
        C = np.eye(dim)
        
        # Weights for selection
        # Use standard CMA-ES weights: w_i = ln(lambda/2 + 0.5) - ln(i) for i=1..lambda/2, 0 otherwise
        # But for MO, we select a subset of non-dominated points.
        # A common heuristic for MO-CMA-ES is to select the top 'mu' points based on hypervolume contribution or just a random subset of the non-dominated set.
        # Let's select mu = lambda / 2 points from the non-dominated set to update the mean and covariance.
        mu = lambda_pop // 2
        
        # Calculate weights for the selected 'mu' individuals
        # Standard CMA-ES weights for the first mu individuals
        weights = np.zeros(lambda_pop)
        for i in range(mu):
            weights[i] = np.log(lambda_pop / 2 + 0.5) - np.log(i + 1)
        # Normalize weights to sum to 1
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights /= weights_sum
        else:
            weights[:mu] = 1.0 / mu
            
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        P = np.empty((lambda_pop, dim))
        F = np.empty((lambda_pop, n_obj))
        
        for i in range(lambda_pop):
            if eval_count >= self.budget or time.time() > deadline:
                break
            z = np.random.randn(dim)
            x = mean + sigma * (C @ z)
            P[i] = x
            F[i] = evaluate(x)
            
        # Evolutionary Loop
        while eval_count < self.budget and time.time() < deadline:
            # 1. Selection: Find non-dominated points
            # We need to select 'mu' points from the current population to update parameters.
            # Ideally, we pick points that contribute to the Pareto front.
            # Simple approach: Find non-dominated set, if size > mu, pick 'mu' with largest hypervolume contribution or just random.
            # Given small mu, let's just pick 'mu' random points from the non-dominated set.
            # If non-dominated set is smaller than mu, pad with dominated points closest to the front (or just random).
            
            # Efficient non-dominated sort for small populations (O(N^2))
            non_dominated_indices = []
            for i in range(lambda_pop):
                is_dominated = False
                for j in range(lambda_pop):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated_indices.append(i)
            
            # Select 'mu' indices
            if len(non_dominated_indices) >= mu:
                # Shuffle and take first mu
                np.random.shuffle(non_dominated_indices)
                selected_indices = non_dominated_indices[:mu]
            else:
                # Take all non-dominated, fill rest with random dominated
                selected_indices = non_dominated_indices.copy()
                dominated_indices = [i for i in range(lambda_pop) if i not in non_dominated_indices]
                np.random.shuffle(dominated_indices)
                while len(selected_indices) < mu and dominated_indices:
                    selected_indices.append(dominated_indices.pop())
                # If still not enough (shouldn't happen if mu <= lambda), pad with random
                while len(selected_indices) < mu:
                    selected_indices.append(np.random.randint(lambda_pop))

            # 2. Update Mean
            # Weighted average of selected points
            # We need to map selected_indices to their positions in the weight array?
            # Standard CMA-ES assumes the top 'mu' individuals are the ones with weights.
            # Here, our "ranking" is by non-domination. We assign the highest weights to the selected non-dominated points.
            # We need to construct a vector of selected X's ordered by weight.
            X_selected = P[selected_indices]
            
            # The weights array `weights` is defined for the first `mu` slots.
            # We assume the selected points are "the best".
            new_mean = np.sum(weights[:, np.newaxis] * X_selected, axis=0)
            
            # 3. Update Covariance and Sigma
            # Calculate rank-one and rank-mu updates
            # First, calculate the standardized deviations
            # C^{-1/2} is needed for p_sigma update. Since C starts as I, we can approximate or compute eigendecomp.
            # For simplicity and speed in this small dimension, we can use the inverse square root via Cholesky or just approximate with I if C is close to I.
            # Let's do a proper update using eigendecomposition for C^{-1/2}.
            
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update p_sigma
            delta_mean = new_mean - mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (delta_mean / sigma))
            
            # Hessian damping
            h_sigma = 1.0
            # Approximation for h_sigma: 1 if ||p_sigma|| < (1.4 + 1/dim) * sqrt((1 - (1-c_sigma)^{2*gen})) * sqrt(mu_eff)
            # We don't track gen count easily here without an extra var, let's use a simpler check or skip h_sigma for simplicity in this implementation.
            # A common simplification is to just use the norm check without the generation-dependent term.
            norm_p_sigma = np.linalg.norm(p_sigma)
            if norm_p_sigma < (1.4 + 1.0 / dim) * np.sqrt(mu_eff):
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / c_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1.0))
            # Clamp sigma to avoid explosion or zero
            sigma = np.clip(sigma, 1e-10, np.max(span))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (delta_mean / sigma)
            
            # Update C
            # Rank-one update
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1 - c_mu) * C + \
                c_1 * (p_c * p_c) + \
                (1 - h_sigma) * (C - C) # The (1-h_sigma) term is usually part of the formula, but let's stick to standard form:
            # Standard form: C = (1 + c1*(h_sigma*(mu_eff-1) - c1 - c_mu)) * C + c1 * p_c * p_c^T + c_mu * sum w_i * y_i * y_i^T
            # Let's re-calculate properly.
            
            # Recalculate C update
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1 - c_mu) * C
            C += c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # y_i = (x_i - mean_old) / sigma
            Y = (X_selected - mean) / sigma
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(Y[i], Y[i])
            
            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # Update mean
            mean = new_mean
            
            # 4. Generate new population
            for i in range(lambda_pop):
                if eval_count >= self.budget or time.time() > deadline:
                    break
                z = np.random.randn(dim)
                x = mean + sigma * (C @ z)
                P[i] = x
                F[i] = evaluate(x)
                
            # Break if we ran out of budget
            if eval_count >= self.budget:
                break

        return all_X
# EVOLVE-BLOCK-END
