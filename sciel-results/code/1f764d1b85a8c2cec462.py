"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MO-CMA-ES: Multi-objective CMA-ES with adaptive covariance matrix."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        lam = max(4, min(20, budget // 5))
        
        # Initialize population uniformly
        X = lo + (hi - lo) * np.random.rand(lam, dim)
        F = np.array([problem(x) for x in X])
        used = lam
        
        # CMA-ES parameters
        sigma = 0.1 * np.max(hi - lo)
        mu = np.mean(X, axis=0)
        C = np.eye(dim)
        
        # Weights for CMA-ES update (standard weights)
        # For multi-objective, we select a subset of parents
        mu_eff = lam // 2  # Number of parents to use for update
        weights = np.zeros(lam)
        # Standard CMA-ES weights: w_i = log(mu_eff/2 + 0.5) - log(i+1)
        for i in range(mu_eff):
            weights[i] = np.log(mu_eff / 2.0 + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # Initial covariance update parameters
        c_sigma = (4 + lam / dim) / ((dim + 4) ** 2 + 2)
        c_c = (4 + lam / dim) / (dim + 4 + 3)
        d_sigma = 1 + 2 * max(0, (lam / dim - 2) / (dim + lam - 2)) + c_sigma
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        pc *= (c_sigma * (2 - c_sigma) + c_c) / (c_c * (1 - c_c)) * np.trace(C)
        
        # Helper function to select parents based on Pareto dominance and crowding
        def select_parents(X, F, num_parents):
            # Compute non-dominated front
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                if dominated[i]: continue
                for j in range(len(F)):
                    if i == j or dominated[j]: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            
            front_indices = [i for i in range(len(F)) if not dominated[i]]
            
            if len(front_indices) <= num_parents:
                # If front is small, fill with best scalarized points
                if len(front_indices) == 0:
                    # Fallback to min sum
                    scores = np.sum(F, axis=1)
                    front_indices = np.argsort(scores)[:num_parents]
                else:
                    # Add best scalarized points not in front
                    scores = np.sum(F, axis=1)
                    all_indices = np.argsort(scores)
                    remaining = [i for i in all_indices if i not in front_indices]
                    front_indices.extend(remaining[:num_parents - len(front_indices)])
                return front_indices[:num_parents]
            
            else:
                # Use Crowding Distance to select most spread out points
                # Normalize objectives for crowding distance
                F_norm = F.copy()
                for j in range(n_obj):
                    min_val = np.min(F[:, j])
                    max_val = np.max(F[:, j])
                    if max_val > min_val:
                        F_norm[:, j] = (F[:, j] - min_val) / (max_val - min_val)
                    
                # Calculate crowding distance for points in the front
                cd = np.zeros(len(F))
                for j in range(n_obj):
                    sorted_idx = np.argsort(F_norm[front_indices, j])
                    cd[front_indices[sorted_idx[0]]] = np.inf
                    cd[front_indices[sorted_idx[-1]]] = np.inf
                    if len(front_indices) > 2:
                        for k in range(1, len(front_indices) - 1):
                            idx1 = front_indices[sorted_idx[k-1]]
                            idx2 = front_indices[sorted_idx[k+1]]
                            idx_mid = front_indices[sorted_idx[k]]
                            cd[idx_mid] += (F_norm[idx2, j] - F_norm[idx1, j]) / (F_norm[idx2, j] - F_norm[idx1, j] if False else (max_val - min_val if max_val > min_val else 1.0))
                
                # Select num_parents with highest crowding distance
                front_cd = cd[front_indices]
                top_k = np.argsort(front_cd)[::-1][:num_parents]
                return [front_indices[i] for i in top_k]

        while used < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Select parents
            parent_indices = select_parents(X, F, mu_eff)
            
            # Compute new mean
            w_sum = np.sum(weights[:mu_eff])
            new_mu = np.sum(weights[:mu_eff, np.newaxis] * X[parent_indices], axis=0) / w_sum
            
            # Update path evolution
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.inv(np.sqrtm(C)) @ (new_mu - mu) / sigma
            h_sigma = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (used + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 2) else 0.0
            
            pc = (1 - c_c) * pc + h_sigma * c_c * (2 - c_c) * np.outer(new_mu - mu, new_mu - mu) / sigma**2
            
            # Update covariance
            delta_c = c_sigma * h_sigma * (1 / sigma - 1) * (new_mu - mu)
            C = (1 + c_sigma * (2 - c_sigma) - c_c * (2 - c_c)) * C + c_sigma * (2 - c_sigma) * (np.outer(new_mu - mu, new_mu - mu) / sigma**2 + delta_c)
            
            # Ensure positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)
            
            mu = new_mu
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim + 1 - 2) - 1))
            
            # Generate offspring
            L = np.linalg.cholesky(C)
            # Sample standard normal
            z = np.random.randn(lam, dim)
            X_new = mu + sigma * (z @ L.T)
            
            # Clip to bounds
            X_new = np.clip(X_new, lo, hi)
            
            # Evaluate
            F_new = np.array([problem(x) for x in X_new])
            used += lam
            
            # Combine populations (Elitist)
            X_comb = np.vstack([X, X_new])
            F_comb = np.vstack([F, F_new])
            
            # Select next population
            # Use a similar selection strategy but for the whole combined population
            # We just need the best 'lam' points.
            # Simple approach: Non-dominated sort + Crowding on combined set
            dominated = np.zeros(len(F_comb), dtype=bool)
            for i in range(len(F_comb)):
                if dominated[i]: continue
                for j in range(len(F_comb)):
                    if i == j or dominated[j]: continue
                    if np.all(F_comb[i] <= F_comb[j]) and np.any(F_comb[i] < F_comb[j]):
                        dominated[j] = True
            
            front_indices = [i for i in range(len(F_comb)) if not dominated[i]]
            
            if len(front_indices) <= lam:
                # If front is small, fill with rest
                if len(front_indices) == 0:
                    scores = np.sum(F_comb, axis=1)
                    next_indices = np.argsort(scores)[:lam]
                else:
                    scores = np.sum(F_comb, axis=1)
                    all_indices = np.argsort(scores)
                    remaining = [i for i in all_indices if i not in front_indices]
                    next_indices = front_indices + remaining[:lam - len(front_indices)]
            else:
                # Crowding distance
                F_norm = F_comb.copy()
                for j in range(n_obj):
                    min_val = np.min(F_comb[:, j])
                    max_val = np.max(F_comb[:, j])
                    if max_val > min_val:
                        F_norm[:, j] = (F_comb[:, j] - min_val) / (max_val - min_val)
                
                cd = np.zeros(len(F_comb))
                for j in range(n_obj):
                    sorted_idx = np.argsort(F_norm[front_indices, j])
                    cd[front_indices[sorted_idx[0]]] = np.inf
                    cd[front_indices[sorted_idx[-1]]] = np.inf
                    if len(front_indices) > 2:
                        for k in range(1, len(front_indices) - 1):
                            idx1 = front_indices[sorted_idx[k-1]]
                            idx2 = front_indices[sorted_idx[k+1]]
                            idx_mid = front_indices[sorted_idx[k]]
                            cd[idx_mid] += (F_norm[idx2, j] - F_norm[idx1, j]) / (max_val - min_val if max_val > min_val else 1.0)
                
                front_cd = cd[front_indices]
                top_k = np.argsort(front_cd)[::-1][:lam]
                next_indices = [front_indices[i] for i in top_k]
            
            X = X_comb[next_indices]
            F = F_comb[next_indices]

        return X
# EVOLVE-BLOCK-END
