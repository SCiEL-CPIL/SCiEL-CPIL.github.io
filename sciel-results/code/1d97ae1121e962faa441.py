"""Multi-objective CMA-ES optimizer for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MO-CMA-ES style multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng(42)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size for CMA-ES
        pop_size = 2 * dim + 1  # ~25 for dim=12
        pop_size = min(pop_size, budget // 2)
        
        # Initial mean and sigma
        mean = mid.copy()
        sigma = 0.3  # Initial step size (fraction of range)
        
        # Covariance matrix
        C = np.eye(dim)
        # Precompute Cholesky factor
        C_cholesky = np.linalg.cholesky(C)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = 0.2 + c_sigma
        c_c = 0.8 + c_c * 0.5  # Scaled down for stability
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] + weights[1] - 1) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Archive of best points found (for hypervolume)
        archive = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=np.float64)
            used += 1
            return x_clipped, f
        
        # Initial population
        X = []
        F = []
        for i in range(pop_size):
            z = rng.standard_normal(dim)
            x = mean + sigma * (C_cholesky @ z)
            x, f = evaluate(x)
            X.append(x)
            F.append(f)
            archive.append((x, f))
        
        X = np.array(X)
        F = np.array(F)
        
        # Helper: fast non-dominated sort
        def fast_nondominated_sort(F_mat):
            n = len(F_mat)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            if not fronts[-1]:
                fronts.pop()
            return fronts
        
        # Crowding distance
        def crowding_distance(F_mat, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F_mat[indices]
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if F_sub[order[-1], k] - F_sub[order[0], k] > 1e-15:
                    for i in range(1, m - 1):
                        cd[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / (F_sub[order[-1], k] - F_sub[order[0], k])
            return cd
        
        # Main loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Sort population by rank and crowding distance
            fronts = fast_nondominated_sort(F)
            
            # Select top mu individuals
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= mu:
                    selected_indices.extend(front)
                else:
                    # Need to select from this front using crowding distance
                    remaining = mu - len(selected_indices)
                    cd = crowding_distance(F, front)
                    order = np.argsort(-cd)
                    selected_indices.extend([front[i] for i in order[:remaining]])
                    break
            
            # Sort selected by objective values (for weighting)
            F_selected = F[selected_indices]
            # Sort by sum of objectives (simple proxy)
            sort_idx = np.argsort(np.sum(F_selected, axis=1))
            selected_indices = [selected_indices[i] for i in sort_idx]
            
            # Update evolution paths
            # Weighted mean of selected
            y_w = np.zeros(dim)
            for i, idx in enumerate(selected_indices):
                y_w += weights[i] * (X[idx] - mean) / sigma
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_cholesky.T @ y_w)
            
            # Check if ps exceeds threshold
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (used + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                # Update pc
                pc = (1 - c_c) * pc + c_c * (np.linalg.norm(ps) / chi_n - 1) * (C_cholesky.T @ y_w)
                
                # Update C
                v = np.zeros((dim, dim))
                for i, idx in enumerate(selected_indices):
                    z_i = (X[idx] - mean) / sigma
                    v += weights[i] * np.outer(z_i, z_i)
                
                h_sigma = (1 + (1 + 2 * max(0, (np.linalg.norm(ps) / chi_n - 1) * (1 - c_c) / c_c)) * c_c / (1 - c_c))
                C = (1 + c_1 + c_mu - c_1) * C + c_1 * (np.outer(ps, ps) - C) * 0 + c_mu * (v - C)
                # Simplified: use standard update
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(ps, ps) + (1 - c_c) * (1 - c_c) * pc @ pc.T - C) + c_mu * v
                # Ensure symmetry
                C = 0.5 * (C + C.T)
                
                # Regularize C
                C = C + 1e-10 * np.eye(dim)
            
            # Update mean
            new_mean = mean + sigma * (C_cholesky @ y_w)
            new_mean = np.clip(new_mean, lo, hi)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-4, 0.5)
            
            # Update Cholesky
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_cholesky = np.eye(dim)
            
            mean = new_mean
            
            # Generate new population
            new_X = []
            new_F = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_cholesky @ z)
                x, f = evaluate(x)
                new_X.append(x)
                new_F.append(f)
                archive.append((x, f))
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
