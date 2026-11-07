import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # Use a Sobol-like quasi-random sequence for initial exploration
        # to cover the space more efficiently than pure random sampling.
        # We'll use a simple Halton sequence generator.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(count, dim, offset=1.0, scale=1.0):
            points = np.zeros((count, dim))
            for i in range(count):
                for j in range(dim):
                    points[i, j] = halton(i + offset, j + 2) * scale + offset
            return points

        # 1. Initial Global Exploration using Halton sequence
        # Spend ~30% of budget on structured sampling
        n_init = max(10, int(budget * 0.3))
        
        # Generate Halton points in [-4.5, 4.5]^dim
        halton_points = halton_sequence(n_init, dim, offset=-4.5, scale=9.0)
        
        for i in range(n_init):
            if n_calls >= budget or time.time() > deadline:
                break
            x = halton_points[i]
            f = timed_call(x)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = timed_call(best_x)

        # 2. Surrogate-based optimization using Gaussian Process
        # We'll use a simple GP with Matern kernel
        
        class SimpleGP:
            def __init__(self, length_scale=1.0, sigma_f=1.0, noise=1e-8):
                self.length_scale = length_scale
                self.sigma_f = sigma_f
                self.noise = noise
                self.X = None
                self.Y = None
                self.L = None
                self.alpha = None
                
            def matern_kernel(self, X1, X2, nu=2.5):
                # Matern 5/2 kernel
                d = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=-1))
                sqrt_5_d = np.sqrt(5) * d / self.length_scale
                K = self.sigma_f**2 * (1 + sqrt_5_d + 5/3 * sqrt_5_d**2) * np.exp(-sqrt_5_d)
                return K
            
            def fit(self, X, Y):
                self.X = X
                self.Y = Y
                n = len(X)
                K = self.matern_kernel(X, X) + self.noise * np.eye(n)
                try:
                    self.L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
                except:
                    self.L = np.linalg.cholesky(K + 1e-4 * np.eye(n))
                self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))
            
            def predict(self, X_new):
                K_s = self.matern_kernel(self.X, X_new)
                mu = K_s.T @ self.alpha
                K_ss = self.matern_kernel(X_new, X_new)
                v = np.linalg.solve(self.L, K_s)
                var = np.diag(K_ss) - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-12)
                return mu, np.sqrt(var)
            
            def expected_improvement(self, X_new, best_f, xi=0.01):
                mu, sigma = self.predict(X_new)
                sigma = np.maximum(sigma, 1e-12)
                imp = best_f - mu - xi
                z = imp / sigma
                # Standard normal CDF and PDF
                from scipy.stats import norm
                EI = imp * norm.cdf(z) + sigma * norm.pdf(z)
                return EI

        # Initialize GP
        gp = SimpleGP(length_scale=1.0, sigma_f=1.0)
        
        # Collect initial points for GP
        X_data = []
        Y_data = []
        
        # Re-evaluate best point and some nearby points to build initial GP
        # Actually, we already have n_init points. Let's collect them.
        # We need to store them during the initial phase.
        # Let's refactor to store points.
        
        # Reset and redo initial phase with storage
        n_calls = 0
        best_x = None
        best_f = float('inf')
        X_data = []
        Y_data = []
        
        for i in range(n_init):
            if n_calls >= budget or time.time() > deadline:
                break
            x = halton_points[i]
            f = timed_call(x)
            X_data.append(x)
            Y_data.append(f)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = timed_call(best_x)
            X_data.append(best_x)
            Y_data.append(best_f)

        # 3. Iterative optimization with GP
        # Alternate between GP-guided exploration and local refinement
        
        # Local refinement function using coordinate-wise line search
        def local_refine(x0, max_calls=20):
            nonlocal n_calls, best_x, best_f
            x = x0.copy()
            f = best_f
            improved = True
            local_calls = 0
            
            while improved and local_calls < max_calls and n_calls < budget and time.time() < deadline:
                improved = False
                for d in range(dim):
                    # Try steps in both directions
                    for direction in [1.0, -1.0]:
                        step = 0.5
                        while step > 1e-4:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5.0, 5.0)
                            
                            f_new = timed_call(x_new)
                            local_calls += 1
                            
                            if f_new < f - 1e-10:
                                x = x_new
                                f = f_new
                                improved = True
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                break
                            else:
                                step *= 0.5
                        else:
                            # If we didn't break, try smaller step
                            continue
                    
                    if local_calls >= max_calls or n_calls >= budget or time.time() > deadline:
                        break
            
            return x, f

        # Main loop: alternate between GP suggestion and local refinement
        n_gp_iters = 0
        stagnation_count = 0
        best_f_prev = best_f
        
        while n_calls < budget and time.time() < deadline:
            # Fit GP
            if len(X_data) >= 2:
                try:
                    gp.fit(np.array(X_data), np.array(Y_data))
                except:
                    break
            
            # Suggest next point using EI
            if len(X_data) < 50:
                # Early on, explore more
                n_candidates = 200
            else:
                n_candidates = 100
            
            # Generate candidate points
            # Mix of random and around current best
            candidates = []
            for _ in range(n_candidates // 2):
                candidates.append(np.random.uniform(-5.0, 5.0, dim))
            for _ in range(n_candidates // 2):
                candidates.append(best_x + 0.5 * np.random.randn(dim))
            
            candidates = np.array(candidates)
            candidates = np.clip(candidates, -5.0, 5.0)
            
            # Compute EI
            if len(X_data) >= 2:
                try:
                    EI = gp.expected_improvement(candidates, best_f, xi=0.01)
                    idx = np.argmax(EI)
                    x_new = candidates[idx]
                except:
                    x_new = np.random.uniform(-5.0, 5.0, dim)
            else:
                x_new = np.random.uniform(-5.0, 5.0, dim)
            
            # Evaluate
            f_new = timed_call(x_new)
            X_data.append(x_new)
            Y_data.append(f_new)
            
            # Check for improvement
            if f_new < best_f_prev - 1e-8:
                stagnation_count = 0
                best_f_prev = best_f
                n_gp_iters += 1
            else:
                stagnation_count += 1
            
            # If stagnating, do local refinement
            if stagnation_count >= 3 and n_calls + 20 < budget:
                x_local, f_local = local_refine(best_x, max_calls=20)
                X_data.append(x_local)
                Y_data.append(f_local)
                stagnation_count = 0
            
            # Adapt GP hyperparameters roughly
            if n_gp_iters % 10 == 0 and n_gp_iters > 0:
                # Adjust length scale based on spread of data
                if len(X_data) > 5:
                    X_arr = np.array(X_data)
                    mean_dist = np.mean(np.linalg.norm(X_arr - X_arr.mean(axis=0), axis=1))
                    gp.length_scale = max(0.1, min(5.0, mean_dist / 2))
        
        return best_x, best_f
