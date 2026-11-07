import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Global exploration with diverse starting points.
    2. Proper CMA-ES on top promising starts for exploitation.
    3. Nelder-Mead local refinement around best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)

        # Phase 1: Global exploration - diverse starting points
        n_starts = max(15, min(60, budget // 3))
        
        starts = []
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            x = np.empty(dim)
            for d in range(dim):
                perm = rng.permutation(n_starts)
                x[d] = -5.0 + 10.0 * (perm[i] + rng.random()) / n_starts
            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is None:
                break
            starts.append((fx, x.copy()))

        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
                starts = [(fx, x.copy())]

        starts.sort(key=lambda t: t[0])

        # Phase 2: Proper CMA-ES for top promising starts
        def cma_es(x0, max_evals, sigma0=1.0):
            """
            Proper CMA-ES implementation with covariance matrix adaptation.
            """
            n = len(x0)
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2  # number of recombination weights
            
            # Weights
            w = np.log(np.log(lam + 1) - np.log(np.arange(1, mu + 1)))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            
            # Parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            # Initial state
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Cholesky of C
            try:
                C_chol = np.linalg.cholesky(C)
            except:
                C_chol = np.eye(n)
            
            evals_used = 0
            
            while evals_used < max_evals and check_time() and calls_made < budget:
                # Sample population
                pop = []
                for j in range(lam):
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_chol @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    pop.append((fy, y.copy()))
                    evals_used += 1
                
                if len(pop) < mu:
                    break
                
                # Sort by fitness
                pop.sort(key=lambda t: t[0])
                
                # Compute new mean
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += w[i] * pop[i][1]
                
                # Update evolution paths
                y_mean = (x_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) / chi_n < (1.4 + 2 / (n + 1))
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_mean
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 - delta_h) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    diff = pop[i][1] - x_mean
                    C += w[i] * np.outer(diff, diff)
                
                # Ensure positive definiteness
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = (eigvecs * eigvals) @ eigvecs.T
                    C_chol = np.linalg.cholesky(C)
                except:
                    C = np.eye(n)
                    C_chol = np.eye(n)
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update mean
                x_mean = x_new
            
            return x_mean, sigma

        # Run CMA-ES for top starts
        remaining = budget - calls_made
        n_local = min(5, len(starts))
        budget_per_local = max(30, remaining // max(1, n_local))
        
        for i, (f0, x0) in enumerate(starts[:n_local]):
            if not check_time() or calls_made >= budget:
                break
            
            if f0 > best_f + 100:
                continue
                
            x_cma, sigma_cma = cma_es(x0.copy(), budget_per_local, sigma0=1.0)

        # Phase 3: Nelder-Mead local refinement
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            
            # Initialize simplex
            alpha = 0.5
            simplex = [x.copy()]
            for d in range(dim):
                x_temp = x.copy()
                x_temp[d] += alpha
                x_temp = np.clip(x_temp, -5, 5)
                simplex.append(x_temp)
            
            # Evaluate simplex
            simplex_f = []
            for sx in simplex:
                fs = safe_call(sx)
                if fs is None:
                    break
                simplex_f.append(fs)
            
            if len(simplex_f) == dim + 1:
                # Nelder-Mead iterations
                max_nm_iters = 200
                for _ in range(max_nm_iters):
                    if not check_time() or calls_made >= budget:
                        break
                    
                    # Sort simplex
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid (excluding worst)
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = np.clip(centroid + (centroid - simplex[-1]), -5, 5)
                    f_r = safe_call(x_r)
                    if f_r is None:
                        break
                    
                    if f_r < simplex_f[0]:
                        # Expansion
                        x_e = np.clip(centroid + 2 * (centroid - simplex[-1]), -5, 5)
                        f_e = safe_call(x_e)
                        if f_e is None:
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contraction
                        x_c = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                        f_c = safe_call(x_c)
                        if f_c is None:
                            break
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                                f_s = safe_call(simplex[i])
                                if f_s is None:
                                    break
                                simplex_f[i] = f_s
                            if calls_made >= budget:
                                break
        
        # Final (1+1)-ES refinement
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            no_improve = 0
            max_no_improve = 50
            
            while check_time() and calls_made < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    no_improve += 1
                
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 3.0)
                
                if no_improve > max_no_improve:
                    break

        return best_x, best_f
