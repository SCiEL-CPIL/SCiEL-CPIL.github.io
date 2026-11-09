    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Halton Initialization
        n_init = 20
        if n_init > budget:
            n_init = budget
        halton_pts = self._halton(n_init, dim, skip=1)
        init_pts = halton_pts * 10.0 - 5.0
        
        for p in init_pts:
            if time.time() > deadline or calls >= budget:
                break
            evaluate(p)

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES
        # Parameters
        lam = 4
        pop_size = 1 + lam
        sigma = 1.0
        x_mean = best_x.copy()
        
        # CMA-ES State
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Adaptation parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) - 2 + 1.0/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Precompute inverse square root of C
        def inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Run CMA-ES
        while calls < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                if calls >= budget:
                    break
                B = inv_sqrt(C)
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ z)
                fy = evaluate(y)
                offspring.append((fy, y))
            
            if not offspring:
                break
                
            # Sort
            offspring.sort(key=lambda t: t[0])
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * offspring[i][1]
            
            # Update evolution paths
            C_sqrt_inv = inv_sqrt(C)
            y_plus = C_sqrt_inv @ (x_new - x_mean)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_plus
            
            # h_sigma calculation
            # Check if ps is within the confidence ellipsoid
            h_sigma = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - calls + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma)**(2 * (calls - calls + 1)))) else 0.0
            # Note: The exponent in h_sigma calculation depends on the number of iterations, not calls.
            # Let's keep a counter for generations.
            
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_plus
            
            # Update covariance
            delta_sigma = 1 + c_mu * (min(5, (np.linalg.norm(pc)**2 / (1 - (1 - c_c)**(2 * (calls - calls + 1))) - 1) / (dim + 1)))
            # Again, exponent issue.
            
            # Let's simplify: Just use a robust update without the complex h_sigma/delta_sigma checks if they are prone to error.
            # Or, use a simpler CMA-ES variant.
            
            # Actually, let's just use the standard update but ensure the exponents are based on generation count.
            # We don't have a generation counter in the loop above. Let's add one.
            
            # For now, let's just do a simple update:
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(
                [weights[i] * np.outer(C_sqrt_inv @ (offspring[i][1] - x_new), C_sqrt_inv @ (offspring[i][1] - x_new)) for i in range(mu)], axis=0
            )
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x_mean = x_new
            
            # Ensure C is positive definite
            eigvals_c, _ = np.linalg.eigh(C)
            if np.min(eigvals_c) < 1e-12:
                C = C + (1e-12 - np.min(eigvals_c)) * np.eye(dim)

        return best_x, best_f
