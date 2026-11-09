import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([3.0, 5.0, 7.0])
        
        X_data = []
        F_data = []
        
        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X_data.append(x)
            F_data.append(f)
            return f

        # 1. Initial Sampling (Latin Hypercube)
        n_init = 20
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            X_init[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + 0.5) / n_init
            
        for x in X_init:
            evaluate(x)
            
        used = n_init

        # GP Parameters
        length_scale = 0.5
        sigma_f = 1.0
        noise = 1e-6

        # Precompute weight vectors for decomposition
        # Simple grid of weights in the simplex
        n_weights = 10
        # For 3 objectives, we can use a fixed set of weights
        # e.g., (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), etc.
        # Or generate random weights
        weights = []
        # Add extreme weights
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add uniform
        weights.append(np.ones(self.n_obj) / self.n_obj)
        # Add some random weights to cover the middle
        for _ in range(n_weights - self.n_obj - 1):
            w = np.random.dirichlet(np.ones(self.n_obj))
            weights.append(w)
        weights = np.array(weights)

        def gp_predict(X_new, X_train, Y_train, K_inv, K_new, mu, var):
            """
            Efficient GP prediction using pre-computed K_inv and K_new.
            X_new: (K, D)
            Returns: mean (K, M), var (K, M)
            """
            N = len(X_train)
            M = Y_train.shape[1]
            
            # Y_train: (N, M)
            # K_inv: (N, N)
            # K_new: (N, K)
            
            means = np.zeros((len(X_new), M))
            vars_ = np.zeros((len(X_new), M))
            
            for m in range(M):
                y = Y_train[:, m]
                alpha = K_inv.dot(y)
                mu_m = K_new.T.dot(alpha)
                var_m = sigma_f**2 - np.sum(K_new**2 * K_inv, axis=1)
                var_m = np.maximum(var_m, 1e-8)
                means[:, m] = mu_m
                vars_[:, m] = var_m
                
            return means, vars_

        def rbf_kernel(A, B, ls):
            A_sq = np.sum(A**2, axis=1)[:, np.newaxis]
            B_sq = np.sum(B**2, axis=1)[np.newaxis, :]
            dist_sq = np.maximum(A_sq + B_sq - 2 * A.dot(B.T), 0)
            return np.exp(-0.5 * dist_sq / (ls**2))

        def ucb_acquisition(x, X_train, F_train, K_inv, K_new, w, beta):
            """
            Calculate UCB for a weighted sum objective.
            """
            x = np.array([x])
            # Standardize x using training stats (pre-computed outside or passed)
            # For simplicity, assume X_train is standardized or we standardize here
            # To be efficient, we should pre-compute standardization stats
            
            # Let's standardize X_train and X_new
            X_train_std = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
            x_std = (x - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
            
            # Recalculate K_new for this specific x? 
            # This is inefficient. We should vectorize or pre-compute.
            # Given the small number of starts, maybe acceptable if we cache?
            # Better: Pass in the standardized training data and kernel matrix components.
            
            # For this implementation, we will re-compute K_new for the candidate x
            # This is O(N) per candidate.
            K_cand = rbf_kernel(X_train_std, x_std, length_scale) # (N, 1)
            
            # Predict mean and var for all objectives
            # We need K_inv which is (N, N)
            # alpha = K_inv @ Y
            # mu = K_cand.T @ alpha
            # var = sigma_f^2 - K_cand.T @ K_inv @ K_cand
            
            # Since we have multiple objectives, we can do this for all at once
            # But UCB is for a weighted sum.
            # f_w = sum w_i f_i
            # mu_w = sum w_i mu_i
            # var_w = sum w_i^2 var_i + sum_{i!=j} w_i w_j cov_ij
            # GP assumes independent objectives? No, multi-output GP usually shares kernel.
            # If we assume independent objectives (diagonal covariance in output space), 
            # then var_w = sum w_i^2 var_i.
            # Let's assume independent objectives for simplicity and speed.
            
            means, vars_ = gp_predict(x, X_train, F_train, K_inv, K_cand, None, None)
            # Wait, gp_predict above expects K_new as (N, K). Here K_cand is (N, 1).
            # Let's rewrite gp_predict to be flexible or just inline the math.
            
            # Inline math for single point
            # K_cand: (N, 1)
            # K_inv: (N, N)
            # Y: (N, M)
            
            # mu = K_cand.T @ K_inv @ Y  -> (1, N) @ (N, N) @ (N, M) -> (1, M)
            # var = sigma_f^2 - K_cand.T @ K_inv @ K_cand -> (1, N) @ (N, N) @ (N, 1) -> (1, 1)
            
            mu_vec = (K_cand.T @ K_inv) @ F_train # (1, M)
            var_vec = sigma_f**2 - (K_cand.T @ K_inv) @ K_cand # (1, 1)
            var_val = var_vec[0, 0]
            if var_val < 0: var_val = 1e-8
            
            # Weighted sum
            mu_w = np.dot(w, mu_vec[0])
            # Assuming independence for variance of weighted sum
            var_w = np.sum(w**2 * var_val) # This is wrong if correlated.
            # If we use a single kernel for all, the outputs are correlated.
            # Cov(f_i, f_j) = K_cand.T @ K_inv @ K_cand (same for all if shared kernel)
            # So Cov matrix is var_val * I? No.
            # If we use independent GPs for each objective, then Cov is diagonal.
            # Let's assume independent GPs for each objective to simplify variance calc.
            # Then var_w = sum w_i^2 * var_i.
            # But we used a single K_inv. This implies shared kernel structure.
            # To be safe and simple, let's use the max variance or just the variance of the weighted sum if we can compute it.
            # Actually, if we assume independent objectives, we should have separate K_inv for each.
            # That's 3x memory and time.
            # Alternative: Use the variance of the weighted sum as the variance of the GP prediction for that specific linear combination.
            # If the GP is multi-output with shared kernel, the prediction covariance is a matrix.
            # For UCB, we need the std of the weighted sum.
            # Let's approximate var_w = var_val (assuming the kernel captures the structure).
            # Or better: var_w = np.sum(w**2) * var_val.
            
            std_w = np.sqrt(np.sum(w**2) * var_val)
            
            ucb = mu_w + beta * std_w
            return -ucb # Minimize negative UCB

        while used < self.budget:
            if time.time() > deadline - 5:
                break
                
            # Prepare GP data
            X_train = np.array(X_data)
            F_train = np.array(F_data)
            N = len(X_train)
            
            # Standardize X_train
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0)
            X_std[X_std < 1e-8] = 1.0
            X_train_std = (X_train - X_mean) / X_std
            
            # Compute Kernel Matrix
            K = rbf_kernel(X_train_std, X_train_std, length_scale)
            K += noise * np.eye(N)
            
            try:
                K_inv = np.linalg.inv(K)
            except:
                K_inv = np.linalg.pinv(K)
                
            # Find best point
            best_x = None
            best_score = np.inf
            
            # Cycle through weights or select best weight
            # For simplicity, let's pick the weight that currently has the worst (highest) weighted sum among non-dominated points
            # Or just cycle through weights in a round-robin fashion to ensure coverage.
            # Let's use a fixed set of weights and pick the one with the highest UCB potential.
            
            # To save time, we will only optimize for 1 weight per iteration.
            # We will cycle through the weights.
            weight_idx = used % len(weights)
            w = weights[weight_idx]
            
            # Multi-start optimization for the selected weight
            n_starts = 5
            # Start points: current best for this weight, random, and center
            starts = []
            # Random starts
            for _ in range(n_starts - 2):
                starts.append(np.random.uniform(lo, hi))
            # Center
            starts.append((lo + hi) / 2.0)
            # Best known point for this weight
            F_weighted = F_train.dot(w)
            best_idx = np.argmin(F_weighted)
            starts.append(X_train[best_idx])
            
            for start in starts:
                def neg_ucb(x):
                    # Standardize x
                    x_std = (x - X_mean) / X_std
                    # Compute K_cand
                    K_cand = rbf_kernel(X_train_std, x_std.reshape(1, -1), length_scale) # (N, 1)
                    
                    # Compute mu and var
                    # mu = K_cand.T @ K_inv @ F_train
                    mu_vec = (K_cand.T @ K_inv) @ F_train # (1, M)
                    # var = sigma_f^2 - K_cand.T @ K_inv @ K_cand
                    var_val = sigma_f**2 - (K_cand.T @ K_inv) @ K_cand # (1, 1)
                    var_val = var_val[0, 0]
                    if var_val < 0: var_val = 1e-8
                    
                    mu_w = np.dot(w, mu_vec[0])
                    std_w = np.sqrt(np.sum(w**2) * var_val)
                    
                    # Beta for UCB (exploration-exploitation trade-off)
                    # Increase beta over time? Or keep fixed.
                    beta = 2.0
                    
                    ucb = mu_w + beta * std_w
                    return -ucb

                try:
                    res = minimize(neg_ucb, start, bounds=list(zip(lo, hi)), method='L-BFGS-B', options={'maxiter': 20})
                    if res.success:
                        score = -res.fun
                        if score < best_score: # Minimizing neg_ucb, so lower is better
                            best_score = score
                            best_x = res.x
                except:
                    pass
            
            if best_x is None:
                best_x = np.random.uniform(lo, hi)
                
            f_new = evaluate(best_x)
            used += 1
            
        return X_data[-1] if X_data else None
