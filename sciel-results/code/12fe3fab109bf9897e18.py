<<<<<<< SEARCH
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
=======
    def _run_cmaes(self, func, x0, dim):
        """Robust CMA-ES with convergence detection and final polish."""
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial step size - use a smaller initial value for better local search
        sigma = 0.5
        
        # Mean
        mean = x0.copy()
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Precompute matrices
        B, D = np.linalg.eigh(C)
        
        # Step size control
        cs = (lambda_ + 2) ** -0.5
        ds = 1 + 2 * max(0, cs - 1 / (lambda_ + 2)) + cs
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Damping parameters
        cmu = 2 / ((dim + 1.3)**2 + lambda_)
        cc = (4 + cmu - dim) / (dim + 4 - cmu)
        
        # Weights
        w = np.log(np.log(lambda_) + 1) - np.log(np.arange(1, lambda_ + 1))
        w = w / w.sum()
        mu = int(np.floor(lambda_ / 2))
        
        # Convergence tracking
        best_f = self.best_f
        best_x = self.best_x.copy()
        no_improve_count = 0
        max_no_improve = 20
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Main loop
        while self.calls < self.budget and time.time() < self._deadline - 5:
            # Check for convergence
            if sigma < 1e-8 or no_improve_count > max_no_improve:
                break
            
            # Generate population
            offspring = []
            fvals = []
            for i in range(lambda_):
                if self.calls >= self.budget:
                    break
                z = np.random.randn(dim)
                y = B @ (D * z)
                x = mean + sigma * y
                x = np.clip(x, -5.0, 5.0)
                offspring.append(x)
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
            
            # Track improvement
            if sorted_fvals[0] < best_f - 1e-10:
                best_f = sorted_fvals[0]
                best_x = sorted_offspring[0].copy()
                self.best_f = best_f
                self.best_x = best_x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update mean
            new_mean = np.sum(w[:mu, None] * np.array(sorted_offspring[:mu]), axis=0)
            
            # Update evolution path for sigma
            ps = (1 - cs) * ps + cs * (1.0 / sigma) * (B.T @ (1.0 / D) * (new_mean - mean))
            
            # Update step size
            norm_ps = np.linalg.norm(ps)
            if norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (self.calls // lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(cs / ds * (norm_ps / chi_n - 1))
            sigma = max(sigma, 1e-12)
            
            # Update evolution path for C
            pc = (1 - cc) * pc + cc * (new_mean - mean) / sigma
            
            # Update covariance
            delta_h = (1 + 2 * max(0, np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (self.calls // lambda_ + 1)) / chi_n - 1) * cc * (2 - cc))
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(w[:mu, None, None] * np.outer(np.array(sorted_offspring[:mu]) - mean, np.array(sorted_offspring[:mu]) - mean) / (sigma ** 2), axis=0)
            
            C = (1 + cmu * (delta_h - 1) - cc * (1 - (1 - cc) ** (self.calls // lambda_ + 1))) * C + cmu * (rank_one + rank_mu)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                B, D = np.linalg.eigh(C)
                D = np.maximum(D, 1e-15)
            except np.linalg.LinAlgError:
                break
            
            mean = new_mean
        
        # Final high-precision polish using coordinate descent
        if self.calls < self.budget - 20 and time.time() < self._deadline - 5:
            self._coordinate_descent_polish(func, self.best_x, dim)
        
        return (self.best_x, self.best_f)
    
    def _coordinate_descent_polish(self, func, x0, dim):
        """Final high-precision polish using coordinate descent."""
        x = x0.copy()
        f = self.best_f
        
        # Try multiple step sizes for coordinate descent
        step_sizes = [0.5, 0.1, 0.01, 0.001, 0.0001]
        
        for step_size in step_sizes:
            improved = True
            while improved and self.calls < self.budget - 10 and time.time() < self._deadline - 3:
                improved = False
                for i in range(dim):
                    if self.calls >= self.budget - 5:
                        break
                    
                    # Try moving in positive direction
                    x_plus = x.copy()
                    x_plus[i] += step_size
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    try:
                        f_plus = func(x_plus)
                        if f_plus < f - 1e-10:
                            x = x_plus
                            f = f_plus
                            self.best_f = f
                            self.best_x = x.copy()
                            improved = True
                    except (ValueError, TimeoutError):
                        break
                    
                    # Try moving in negative direction
                    x_minus = x.copy()
                    x_minus[i] -= step_size
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    try:
                        f_minus = func(x_minus)
                        if f_minus < f - 1e-10:
                            x = x_minus
                            f = f_minus
                            self.best_f = f
                            self.best_x = x.copy()
                            improved = True
                    except (ValueError, TimeoutError):
                        break
                
                if not improved:
                    break
        
        return (self.best_x, self.best_f)
>>>>>>> REPLACE

