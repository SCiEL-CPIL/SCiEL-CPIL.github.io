import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 300s limit
        
        def time_check():
            return (time.time() - start_time) > timeout

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        eval_count = 0

        # CMA-ES Parameters
        # For dim=5, lambda=4+floor(3*ln(5)) = 4+5 = 9 is standard
        # Let's use lambda=10 for even numbers
        lam = 10
        mu = lam // 2  # Number of parents
        
        # Initial parameters
        # The problem states minimizer is in [-4, 4]^5. 
        # Start at 0,0,0,0,0
        x_mean = np.zeros(dim)
        
        # Initial sigma. The box is [-5,5], width 10. 
        # A reasonable initial step size is 1.0 or 2.0.
        # Since the step ellipsoidal function has a "step", a larger initial sigma helps exploration.
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Adaptation parameters
        # Standard CMA-ES parameters for dim=5
        # c_sigma = (mu_eff + 2) / ((N + 2) + mu_eff)
        # c_c = (4 + mu_eff/N) / (N + 4 + 2*mu_eff/N)
        # c_1 = 2 / ((N+1.3)^2 + mu)
        # c_mu = min(1-c_1, 2*(mu_eff-2+1/mu)/( (N+2)^2 + mu_eff))
        
        N = dim
        # Effective population size
        mu_eff = mu / np.sum((np.arange(1, mu+1)**-1.0))  # Standard weights w_i = ln(mu/2+1) - ln(i)
        
        # Weights
        w = np.log(mu / 2 + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / ((N + 2) + mu_eff)
        c_c = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
        c_1 = 2 / ((N + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2)**2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        
        # Eigen decomposition cache for speed
        eigvals, eigvecs = np.linalg.eigh(C)
        
        def update_covariance(psi, p_sigma, p_c):
            nonlocal C, eigvals, eigvecs
            # Standard CMA-ES update
            # h_sigma = 1 if ||p_sigma|| < (1 + 0.3/N) * sqrt(1 - (1-c_sigma)^{2/gen}) else 0
            # For simplicity and robustness in small budget, we can use a simplified update or standard.
            # Given dim=5, full rank update is cheap.
            
            # Rank-one update
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - (1 - c_sigma)**2) * C)
            
            # Rank-mu update
            # We need the selected individuals
            # This function is called with the data needed, but in the main loop we pass them
            # For this implementation, I will do the update inside the loop for clarity and speed
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip negative eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T
            # Symmetrize
            C = (C + C.T) / 2

        # Main Loop
        # We will use a simple CMA-ES loop.
        # To answer the "error_repair" question: 
        # When generating x_new, if it falls outside [-5, 5], we "repair" it by clipping.
        # The current program relies on func() clipping. 
        # The hypothesis is that explicit repair (clipping) before evaluation might allow 
        # the optimizer to recognize that the boundary is a constraint and adjust sigma,
        # OR it's just the same as func clipping. 
        # However, a "different" strategy is to use CMA-ES which is generally superior 
        # to (1+1)-ES for ellipsoidal functions due to covariance adaptation.
        
        # Let's implement a robust CMA-ES.
        
        # Initial population generation
        # We need to manage the budget carefully.
        # Each generation costs 'lam' evaluations.
        
        while eval_count < budget and not time_check():
            # Generate population
            Y = np.array([rng.multivariate_normal(np.zeros(dim), C) for _ in range(lam)])
            X = x_mean + sigma * Y
            
            # Error Repair: Clip to box [-5, 5]
            # This ensures we only evaluate valid points. 
            # Note: func() also clips, but doing it here allows us to track if points are on boundary.
            # If a point is clipped, it's on the boundary. 
            # For now, we just clip.
            X = np.clip(X, -5.0, 5.0)
            
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                if eval_count >= budget or time_check():
                    break
                F[i] = func(X[i])
                eval_count += 1
                
            # If we ran out of budget mid-generation, handle partial population
            # For simplicity, if we don't have full lam, we break and return best so far.
            # But we should update best_f with whatever we evaluated.
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            # If we didn't complete the generation, we can't do a proper CMA update.
            # We could do a simplified (1+1) update on the best, but let's just break 
            # if we are close to budget, or handle partial.
            # Given budget=500 and lam=10, we have ~50 generations. 
            # If eval_count < budget but we broke the inner loop, it means time_check failed or budget hit.
            # If budget hit exactly, we are done.
            # If time_check failed, we are done.
            # If we broke early for another reason, we should probably just return.
            # Let's assume we complete the generation if we enter the while loop.
            # But to be safe, if we didn't evaluate all 'lam', we break the outer loop.
            
            # Check if we evaluated all lam
            # The inner loop breaks if eval_count >= budget.
            # If eval_count < budget, it means we evaluated all lam.
            # Wait, the inner loop condition is `if eval_count >= budget ... break`.
            # So if we finish the for loop, we evaluated min(lam, budget - old_count) points.
            # If budget was sufficient, we have lam points.
            
            if eval_count >= budget:
                break
            
            # Sort indices by fitness
            idx = np.argsort(F)
            X_sel = X[idx[:mu]]
            F_sel = F[idx[:mu]]
            
            # Update mean
            x_mean = np.sum(w[:, np.newaxis] * X_sel, axis=0)
            
            # Update paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} * (x_mean_new - x_mean_old) / sigma
            # Need C^{-1/2}
            # We have eigvals, eigvecs. C^{-1/2} = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_mean - x_mean_old)) / sigma
            
            # h_sigma check
            # Approximate
            h_sigma = 1.0 # Simplified, usually 1
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (x_mean - x_mean_old)) / sigma
            
            # Update Covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - (1 - c_sigma)**2) * C)
            # + c_mu * sum(w_i * (y_i @ y_i))
            
            # Calculate rank-mu term
            # y_i = (X_sel_i - x_mean_old) / sigma
            Y_sel = (X_sel - x_mean_old) / sigma
            rank_mu_term = np.sum(w[:, np.newaxis, np.newaxis] * (Y_sel[:, :, np.newaxis] * Y_sel[:, np.newaxis, :]), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - (1 - c_sigma)**2) * C) + c_mu * rank_mu_term
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / chi_N * (||p_sigma|| / chi_N - 1))
            # chi_N = sqrt(N) * (1 - 1/(4N) + 1/(21N^2))
            chi_N = np.sqrt(N) * (1 - 1/(4*N) + 1/(21*N**2))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp((c_sigma / chi_N) * (norm_p_sigma / chi_N - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Store old mean for next iteration
            x_mean_old = x_mean.copy()

        return best_x, best_f
