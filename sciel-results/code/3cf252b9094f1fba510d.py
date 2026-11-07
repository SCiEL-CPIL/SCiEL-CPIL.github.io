"""Optimizer for BBOB f19 (Griewank-Rosenbrock): Gaussian Process BO with Nelder-Mead polish."""
import numpy as np
import time
import math
from scipy.stats import norm
from scipy.spatial.distance import cdist

class Optimizer:
    """GP-based optimizer with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Storage for GP
        X_train = []
        y_train = []

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                fx = func(np.clip(x, -5, 5))
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        rng = np.random.default_rng(12345)

        # Nelder-Mead implementation for local polish
        def nelder_mead(x0, max_calls, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                x_temp = x0.copy()
                if x_temp[i] == 0:
                    x_temp[i] = 0.05
                else:
                    x_temp[i] += 0.05
                simplex.append(x_temp)
            
            f_simplex = []
            for i in range(dim + 1):
                if calls_made >= budget or time.time() >= deadline:
                    return
                fx = safe_eval(simplex[i])
                if fx is None:
                    return
                f_simplex.append(fx)
            
            # Main loop
            max_iter = max_calls // (dim + 1) + 5
            for k in range(max_iter):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_best = simplex[0]
                f_best = f_simplex[0]
                
                # Check for convergence
                if f_worst - f_best < 1e-12:
                    break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break

            return

        # GP Surrogate Implementation
        def gp_predict(X_new, X_train, y_train, length_scale=1.0, signal_variance=1.0, noise_variance=1e-6):
            """
            Predict mean and variance for X_new using GP.
            X_new: (N_new, D)
            X_train: (N_train, D)
            y_train: (N_train,)
            """
            if len(X_train) == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new))
            
            # Normalize inputs to [0, 1] to stabilize kernel
            X_train_norm = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8)
            X_new_norm = (X_new - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8)
            
            # Squared Exponential Kernel
            def kernel(A, B):
                dists = cdist(A, B, metric='sqeuclidean')
                return signal_variance * np.exp(-0.5 * dists / (length_scale ** 2))
            
            K_train = kernel(X_train_norm, X_train_norm) + noise_variance * np.eye(len(X_train))
            K_train_new = kernel(X_train_norm, X_new_norm)
            
            try:
                L = np.linalg.cholesky(K_train + 1e-8 * np.eye(len(K_train)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                
                mean = K_train_new.T @ alpha
                v = np.linalg.solve(L, K_train_new)
                var = signal_variance - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
            except Exception:
                # Fallback if Cholesky fails
                try:
                    K_inv = np.linalg.pinv(K_train)
                    alpha = K_inv @ y_train
                    mean = K_train_new.T @ alpha
                    v = K_train_new @ K_inv
                    var = signal_variance - np.sum(v * K_train_new.T, axis=0)
                    var = np.maximum(var, 1e-10)
                except Exception:
                    mean = np.full(len(X_new), np.mean(y_train))
                    var = np.full(len(X_new), np.var(y_train) + 1e-10)
            
            return mean, var

        # Expected Improvement Acquisition
        def expected_improvement(X_cand, X_train, y_train, best_f, length_scale=1.0, signal_variance=1.0, noise_variance=1e-6, xi=0.01):
            """
            Calculate EI for a set of candidates.
            """
            mean, var = gp_predict(X_cand, X_train, y_train, length_scale, signal_variance, noise_variance)
            std = np.sqrt(var)
            
            # Avoid division by zero
            std[std < 1e-10] = 1e-10
            
            # EI = (mu - best_f - xi) * Phi(Z) + std * phi(Z)
            # Z = (mu - best_f - xi) / std
            improvement = mean - best_f - xi
            Z = improvement / std
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            
            return ei

        # Phase 1: Initial Exploration with Diverse Points
        # Start with a few random points to initialize GP
        n_init = 10
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))
        
        for x0 in init_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            fx = safe_eval(x0)
            if fx is None:
                break
            if fx is not None:
                X_train.append(x0.copy())
                y_train.append(fx)

        # Phase 2: Bayesian Optimization Loop
        # We use GP to guide the next point until we have enough confidence or budget is low
        # We switch to local polish when we have a good candidate
        gp_iterations = 0
        max_gp_iters = 300 # Cap GP iterations to leave room for polish
        
        while calls_made < budget and time.time() < deadline and gp_iterations < max_gp_iters:
            gp_iterations += 1
            
            # Generate candidates
            # Mix of random and local perturbations around best
            n_cand = 200
            candidates = []
            
            # 50% random
            n_rand = n_cand // 2
            candidates.extend(rng.uniform(-4, 4, (n_rand, dim)).tolist())
            
            # 50% local around best
            n_local = n_cand - n_rand
            if best_f < float('inf'):
                local_cands = best_x + rng.normal(0, 0.5, (n_local, dim))
                local_cands = np.clip(local_cands, -4, 4)
                candidates.extend(local_cands.tolist())
            
            candidates = np.array(candidates)
            
            # Calculate EI
            try:
                ei = expected_improvement(candidates, np.array(X_train), np.array(y_train), best_f, length_scale=1.5, signal_variance=1.0)
                idx_max = np.argmax(ei)
                next_x = candidates[idx_max]
            except Exception:
                # Fallback to random
                next_x = rng.uniform(-4, 4, dim)

            # Evaluate next point
            fx = safe_eval(next_x)
            if fx is None:
                break
            
            X_train.append(next_x.copy())
            y_train.append(fx)
            
            # Check for convergence/stagnation to trigger early polish
            if gp_iterations > 20 and len(X_train) > 5:
                # Simple check: if best hasn't improved in last 10 iterations, polish
                # We don't track history explicitly, so we just check if we've done enough GP steps
                # and if the variance is low, we can polish.
                # For simplicity, we will run GP for a fixed number of iterations or until budget is tight.
                pass

        # Phase 3: Nelder-Mead Polish
        # Use remaining budget for local polish
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 10:
                nm_budget = min(remaining, max(20, int(remaining * 0.5)))
                nelder_mead(best_x, nm_budget)
            
            # If budget still remains, do another polish from a perturbed best
            if calls_made < budget and time.time() < deadline:
                x0 = best_x + rng.standard_normal(dim) * 0.1
                x0 = np.clip(x0, -5, 5)
                remaining = budget - calls_made
                if remaining > 10:
                    nm_budget = min(remaining, 20)
                    nelder_mead(x0, nm_budget)

        return best_x, best_f
