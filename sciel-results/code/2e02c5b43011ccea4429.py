import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        N = 4 + int(3 * np.log(d))
        if N < 4: N = 4
        # For d=5, N=4+3*1.609=8.8 -> 8. Let's use N=10 for robustness.
        N = 10 
        
        # CMA-ES parameters
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        chi_lambda = np.sqrt(N) * (1 - 1/(2*N) + 1/(8*N**2))
        
        c_sigma = (mu_eff + 0.5) / (N + 1.5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (N + 1)) + c_sigma
        c_c = (4 - mu_eff + c_sigma) / (N + 4 + c_sigma)
        c_cov = (2 * (mu_eff - 2 + 1/mu_eff)) / ((N + 1.3)**2 + mu_eff) + c_c * (2 - c_c - 2/(N+1.3))
        c_1 = 2 / ((d + 1.3)**2 + N)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((d + 2)**2 + N))
        
        sigma = 2.0
        x_mean = np.random.uniform(-5, 5, d)
        C = np.eye(d)
        C_inv_sqrt = np.eye(d) # Not strictly needed if we use Cholesky
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Initial evaluation
        fx_mean = func(x_mean)
        best_x = x_mean.copy()
        best_f = fx_mean
        calls = 1
        
        # Precompute eigen decomposition for C update? No, update C directly.
        # We need to sample from N(0, C). We can use Cholesky of C.
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Sample
            # Get Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, project to PD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-6)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C)
            
            Y = np.random.normal(0, 1, (N, d))
            X = x_mean + sigma * (Y @ L.T) # Or Y @ L.T depending on convention. Usually x = mean + sigma * L @ z.
            # Let's use x_i = x_mean + sigma * (L @ z_i).
            # Y is (N, d). L is (d, d).
            # We want X[i] = x_mean + sigma * (L @ Y[i])? No, Y[i] is standard normal.
            # Covariance of L @ z is L L^T = C.
            # So X = x_mean[None, :] + sigma * (Y @ L.T) is correct if Y rows are z^T.
            # Wait, (Y @ L.T)[i, j] = sum_k Y[i, k] L.T[k, j] = sum_k Y[i, k] L[j, k].
            # This is equivalent to (L @ Y[i])^T?
            # Let's stick to: Z = np.random.normal(0, 1, (N, d))
            # X = x_mean[None, :] + sigma * (Z @ L.T)
            
            # Evaluate
            F = np.zeros(N)
            for i in range(N):
                if calls >= self.budget:
                    break
                F[i] = func(X[i])
                calls += 1
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            Y = Y[idx]
            
            # Update mean
            w = weights
            x_new = x_mean + sigma * np.dot(w, Y[:mu]) # Wait, Y here is the sorted Y?
            # The formula is x_{t+1} = x_t + sigma * sum w_i * (z_{i,t} - z_{mean,t})?
            # No, standard CMA-ES:
            # x_{t+1} = x_t + sigma * sum_{i=1}^mu w_i (y_{i,t})
            # where y_{i,t} are the sorted samples from N(0, C).
            
            # But we sampled X = x_mean + sigma * (Z @ L.T).
            # So Z @ L.T are the y's.
            # We need the sorted y's.
            # We have X sorted.
            # y_sorted = (X_sorted - x_mean) / sigma.
            
            Y_sorted = (X - x_mean[None, :]) / sigma
            
            x_new = x_mean + sigma * np.dot(w, Y_sorted[:mu])
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * h_sigma * (L^-1 / sigma) * (x_new - x_mean)
            # (x_new - x_mean) / sigma = sum w_i y_i
            # L^-1 @ (sum w_i y_i)
            
            # Calculate h_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(t+1)}) < (1.4 + 2/(d+1)) * chi_lambda
            
            # We need to track t or just use the approximation.
            # Let's approximate the denominator as 1 for simplicity or track it.
            # For short runs, 1 is okay, but let's be precise.
            # Actually, we can just use the condition ||p_sigma|| < (1.4 + 2/(d+1)) * chi_lambda * sqrt(1 - (1-c_sigma)^{2})?
            # No, the standard check is:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2}) < (1.4 + 2/(d+1)) * chi_lambda
            
            # We need L_inv.
            # L is lower triangular. L_inv = inv(L).
            L_inv = np.linalg.inv(L)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (L_inv @ (x_new - x_mean) / sigma)
            
            # h_sigma calculation
            # norm_p_sigma = np.linalg.norm(p_sigma)
            # denom = np.sqrt(1 - (1 - c_sigma)**2) # This is an approximation for t=1.
            # Better to track the power.
            # Let's just use the threshold without the time-dependent correction for simplicity, 
            # or use a fixed threshold.
            # Threshold = (1.4 + 2/(d+1)) * chi_lambda
            threshold = (1.4 + 2/(d+1)) * chi_lambda
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < threshold else 0.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (x_new - x_mean) / sigma
            
            # Update C
            # C = (1 - c_cov - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma)c_c(1-c_c)C) + c_mu sum w_i (y_i y_i^T)
            # Note: y_i here are the standardized samples (Y_sorted).
            
            # Rank-1 update
            # p_c is in x-space. We need it in y-space?
            # Formula: C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma)c_c(1-c_c)C) + c_mu sum w_i y_i y_i^T
            # Wait, p_c is usually defined such that its covariance matches the y-space?
            # In Hansen's thesis:
            # p_c is updated with (x_{new} - x_{old})/sigma.
            # The update for C uses p_c p_c^T.
            
            # Let's verify the units.
            # C is covariance of Y (standard normal transformed).
            # x = x_mean + sigma * Y.
            # So Cov(X) = sigma^2 C.
            # p_c is a vector.
            
            # Update C
            delta_c = (1 - h_sigma) * c_c * (1 - c_c)
            C = (1 - c_cov - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_c * C)
            
            # Rank-mu update
            # sum w_i y_i y_i^T
            # Y_sorted[:mu] are the y's.
            Y_mu = Y_sorted[:mu]
            # sum w_i outer(y_i, y_i)
            # This is (Y_mu.T @ (w[:, None] * Y_mu)) ?
            # Let W = np.diag(w)
            # Sum = Y_mu.T @ W @ Y_mu
            # Efficiently: (Y_mu * w[:, None]).T @ Y_mu
            
            C += c_mu * ((Y_mu * w[:, None]).T @ Y_mu)
            
            # Ensure PD
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-6:
                # Add small identity
                C += (1e-6 - np.min(eigvals)) * np.eye(d)
            
            x_mean = x_new
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
        
        return best_x, best_f
