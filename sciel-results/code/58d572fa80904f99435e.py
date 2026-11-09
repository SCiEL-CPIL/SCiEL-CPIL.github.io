<<<<<<< SEARCH
        # Phase 3: Local Refinement with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.3
        
        # Track consecutive failures for adaptive sigma
        fail_count = 0
        max_fails = 15
        
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.05, 2.0)
                fail_count = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.95
                fail_count += 1
                
            # Reset if too many failures
            if fail_count > max_fails:
                sigma = max(sigma, 0.05)
                fail_count = 0
                
            if sigma < 1e-10:
                break
=======
        # Phase 3: Local Refinement with CMA-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        C = np.eye(dim)  # Covariance matrix
        B = np.eye(dim)  # Eigenvectors
        D = np.ones(dim)  # Eigenvalues
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # CMA-ES parameters
        mu = max(1, dim // 2)  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        def update_cma(x, sigma, C, B, D, p_sigma, p_c, fx, remaining_budget):
            nonlocal budget_used, best_f, best_x
            
            while budget_used < budget and time.time() < deadline:
                if sigma < 1e-15:
                    break
                
                # Generate lambda = 2*mu offspring
                lam = 2 * mu
                ys = []
                xs = []
                fs = []
                
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    # Sample from N(x, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * (B.T @ z)))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    budget_used += 1
                    ys.append(y)
                    xs.append(y)
                    fs.append(fy)
                
                if len(ys) == 0:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in order])
                fs_sorted = [fs[i] for i in order]
                
                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = ys_sorted[0].copy()
                
                # Selection
                y_best = ys_sorted[0]
                y_mean = np.zeros(dim)
                for i in range(mu):
                    y_mean += weights[i] * ys_sorted[i]
                
                # Update p_sigma
                c_sqrt_sigma = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
                C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
                p_sigma = (1 - c_sigma) * p_sigma + c_sqrt_sigma * (C_inv_sqrt @ (y_mean - x) / sigma)
                
                # Update p_c
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget_used - dim)))) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (budget_used - dim)))
                p_c = (1 - c_c) * p_c + h_sigma * c_c * (y_mean - x) / sigma
                
                # Update covariance matrix
                delta_h = (2 - c_c) * h_sigma - 1
                rank_one = np.outer(p_c + delta_h * x, p_c + delta_h * x)
                
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    diff = ys_sorted[i] - x
                    rank_mu += weights[i] * np.outer(diff, diff)
                
                C = (1 + c_1 * (c_mu - c_1) - c_1) * C + c_1 * rank_one + c_mu * rank_mu / sigma ** 2
                
                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update x
                x = y_mean
                fx = fs_sorted[0]
                
                # Update eigendecomposition periodically
                if budget_used % dim == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        update_cma(x, sigma, C, B, D, p_sigma, p_c, fx, budget - budget_used)
>>>>>>> REPLACE
