"""Direct optimizer exploiting Rastrigin periodicity via multi-start CMA-ES with restarts."""
import numpy as np
import time
import math
from scipy.stats import norm


class Optimizer:
    """Direct optimizer using multi-start CMA-ES with adaptive restarts for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # Phase 1: Initial random sampling to find a promising basin
        n_random = 20
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        if not time_ok():
            return (best_x, best_f)

        # CMA-ES parameters
        lambda_pop = 10 + int(3 * math.log(dim))
        mu = lambda_pop // 2
        
        c_sigma = (mu + 0.5) / (dim + 2) / 5
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        chi_N = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        w /= w.sum()
        nu_eff = 1 / sum(w ** 2)
        
        def run_cmaes(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma0 for at most max_evals evaluations."""
            nonlocal calls_used, best_x, best_f
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            evolution_path_sigma = np.zeros(dim)
            evolution_path_cov = np.zeros((dim, dim))
            
            last_best = best_f
            no_improve = 0
            evals_this_run = 0
            
            while evals_this_run < max_evals and time_ok():
                # Generate population
                offspring = []
                for _ in range(lambda_pop):
                    if not time_ok() or evals_this_run >= max_evals:
                        break
                    z = rng.standard_normal(dim)
                    x_off = x_mean + sigma * (C @ z)
                    x_off = np.clip(x_off, -5, 5)
                    fx_off = evaluate(x_off)
                    evals_this_run += 1
                    offspring.append((fx_off, x_off))
                
                if len(offspring) < mu:
                    break
                
                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]
                
                if selected[0][0] < last_best - 1e-10:
                    last_best = selected[0][0]
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > 15:
                    break
                
                # Update mean
                x_new_mean = sum(w[i] * selected[i][1] for i in range(mu))
                
                # Update sigma evolution path
                evolution_path_sigma = (1 - c_sigma) * evolution_path_sigma + \
                                       math.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * \
                                       (C @ ((x_new_mean - x_mean) / sigma))
                
                h_sigma = float(np.linalg.norm(evolution_path_sigma) / chi_N < (1.4 + 2 / (dim + 1)))
                
                # Update covariance evolution path
                evolution_path_cov = (1 - c_c) * evolution_path_cov + \
                                     h_sigma * c_c * (1 - c_c) * np.outer(x_new_mean - x_mean,
                                                                          x_new_mean - x_mean)
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
                rank_mu = sum(w[i] * np.outer(selected[i][1] - x_mean,
                                              selected[i][1] - x_mean) for i in range(mu))
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * evolution_path_cov + c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= math.exp(c_sigma / chi_N * (np.linalg.norm(evolution_path_sigma) / chi_N - 1))
                sigma = float(np.clip(sigma, 0.001, 10.0))
                
                x_mean = x_new_mean
                
                if sigma < 0.001:
                    break
            
            return best_f

        def run_gp_bo(x0, sigma0, max_evals):
            """Run GP-based Bayesian Optimization for at most max_evals evaluations."""
            nonlocal calls_used, best_x, best_f
            
            # Store evaluated points
            X_eval = [x0.copy()]
            y_eval = [best_f]
            
            # Initial GP parameters
            length_scale = sigma0
            noise = 1e-6
            y_std = 1.0
            
            evals_this_run = 0
            no_improve = 0
            last_best = best_f
            
            while evals_this_run < max_evals and time_ok():
                # Fit GP to current data
                X_arr = np.array(X_eval)
                y_arr = np.array(y_eval)
                
                # Standardize y for GP
                y_mean = y_arr.mean()
                y_scale = y_arr.std() + 1e-8
                y_std_arr = (y_arr - y_mean) / y_scale
                
                # Compute GP kernel matrix
                def rbf_kernel(X1, X2, ls):
                    # X1: (n, d), X2: (m, d)
                    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                    sq_dists = np.maximum(sq_dists, 0)
                    return np.exp(-sq_dists / (2 * ls**2))
                
                K = rbf_kernel(X_arr, X_arr, length_scale) + noise * np.eye(len(X_arr))
                
                # Predict at candidate points
                # Generate candidate points: mixture of random and around best
                n_candidates = 50
                candidates = []
                # 30 random points
                for _ in range(30):
                    candidates.append(rng.uniform(-5, 5, dim))
                # 20 points near best
                for _ in range(20):
                    candidates.append(best_x + rng.standard_normal(dim) * 0.5)
                candidates = np.array(candidates)
                
                # Compute predictions
                K_star = rbf_kernel(candidates, X_arr, length_scale)
                K_inv = np.linalg.inv(K)
                
                mu_pred = K_star @ K_inv @ y_std_arr
                v = K_inv @ K_star.T
                var_pred = 1 - np.sum(v * K_star, axis=1)
                var_pred = np.maximum(var_pred, 1e-8)
                
                # Expected Improvement acquisition
                f_best_std = y_std_arr.min()
                Z = (f_best_std - mu_pred) / np.sqrt(var_pred)
                EI = (f_best_std - mu_pred) * norm.cdf(Z) + np.sqrt(var_pred) * norm.pdf(Z)
                
                # Select candidate with max EI
                idx = np.argmax(EI)
                x_next = candidates[idx]
                x_next = np.clip(x_next, -5, 5)
                
                # Evaluate
                fx_next = evaluate(x_next)
                evals_this_run += 1
                
                # Update data
                X_eval.append(x_next.copy())
                y_eval.append(fx_next)
                
                # Update best
                if fx_next < last_best - 1e-10:
                    last_best = fx_next
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > 20:
                    break
                
                # Adapt length scale based on improvement
                if no_improve % 5 == 0:
                    length_scale *= 0.8  # Shrink to focus more
                    length_scale = max(length_scale, 0.1)
            
            return best_f

        # Phase 2: Multi-start with CMA-ES and GP-BO
        # Strategy: Use CMA-ES for exploitation, GP-BO for exploration
        remaining_budget = budget - calls_used
        
        # Split budget: 60% for CMA-ES runs, 40% for GP-BO
        cmaes_budget = int(remaining_budget * 0.6)
        gpbo_budget = int(remaining_budget * 0.4)
        
        # Phase 2a: Multi-start CMA-ES
        num_cmaes_runs = 4
        evals_per_cmaes = max(10, int(cmaes_budget / num_cmaes_runs))
        
        for i in range(num_cmaes_runs):
            if not time_ok():
                break
            
            if i == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            elif i % 2 == 0:
                x0 = best_x.copy()
                sigma0 = 0.5
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.5
            
            max_evals = min(evals_per_cmaes, budget - calls_used)
            if max_evals > 0:
                run_cmaes(x0, sigma0, max_evals)
        
        # Phase 2b: GP-based Bayesian Optimization for exploration
        if time_ok() and budget - calls_used > 10:
            # Use GP-BO with larger initial length scale for exploration
            max_gpbo_evals = min(gpbo_budget, budget - calls_used)
            if max_gpbo_evals > 0:
                run_gp_bo(best_x.copy(), 2.0, max_gpbo_evals)
        
        # Phase 3: Final refinement with small sigma CMA-ES
        if time_ok() and budget - calls_used > 10:
            max_evals = min(30, budget - calls_used)
            run_cmaes(best_x.copy(), 0.05, max_evals)

        return (best_x, best_f)
