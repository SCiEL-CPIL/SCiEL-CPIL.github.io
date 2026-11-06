import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 24
    # We fix c[0] = 1 to break symmetry, so we optimize c[1..23]
    # But for the continuous relaxation, we can just optimize all 24 and then round.
    # Actually, fixing one helps with symmetry but doesn't change the landscape much.
    # Let's optimize all 24 coefficients in [-1, 1].
    
    # Evaluation grid
    # Use a reasonably dense grid for the surrogate
    M = 2048
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * angles)
    
    # Precompute powers of z
    # Z[k, m] = z[m]^k
    Z = np.empty((N, M), dtype=np.complex128)
    Z[0, :] = 1.0
    for k in range(1, N):
        Z[k, :] = Z[k-1, :] * z
        
    # ZT[m, k] = Z[k, m] for easier matrix multiplication
    # P(z_m) = sum_k c_k Z[k, m]
    # So P = ZT @ c where ZT is (M, N) and c is (N,)
    ZT = Z.T  # (M, N)
    
    best_score = np.inf
    best_coeffs = None
    
    # Parameters for optimization
    num_restarts = 20
    steps_per_restart = 200
    lr = 0.05
    p_power = 100  # High power to approximate max
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Initialize coefficients randomly in [-1, 1]
        # Start with random +/-1 or random continuous?
        # Continuous random is better for gradient descent.
        x = np.random.uniform(-1, 1, N)
        
        # Adaptive learning rate or fixed? Fixed is simpler.
        # We will use a simple gradient descent with numerical gradients.
        # To speed up, we can use analytic gradients if possible, but numerical is safer for implementation.
        # However, numerical gradient on M=2048 points is expensive.
        # Let's try a few steps of gradient descent.
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 10.0:
                break
                
            # Evaluate P(z)
            P = ZT @ x  # (M,)
            abs_P = np.abs(P)
            
            # Objective: sum(abs_P^p)
            # Gradient: d/dx_j sum_m |P_m|^p = sum_m p |P_m|^{p-2} P_m conj(dP_m/dx_j)
            # dP_m/dx_j = Z[j, m]
            # So grad_j = sum_m p |P_m|^{p-2} P_m conj(Z[j, m])
            # Note: Z[j, m] = z_m^j. conj(Z[j, m]) = z_m^{-j}.
            
            # Weight for each point m
            # w_m = p * |P_m|^{p-2} * P_m
            # To avoid overflow with p=100, we can scale or use log-sum-exp style, 
            # but for p=100, |P| is around sqrt(24)~5, 5^100 is huge.
            # Let's use a smaller p or normalize.
            # Alternatively, use p=10 which is smoother and still pushes peaks.
            # Or use soft-max: max is approximated by (1/p) log sum exp(p |P|).
            # Let's stick to sum of squares of high power but normalize by max to keep numbers stable.
            max_abs = np.max(abs_P)
            if max_abs < 1e-9:
                break
            # Normalize P for gradient calculation to avoid overflow
            P_norm = P / max_abs
            abs_P_norm = abs_P / max_abs
            
            # Use p=20 for stability
            p = 20
            weights = (p * (abs_P_norm ** (p - 2)) * P_norm) * (max_abs ** (p - 1))
            # weights[m] is the coefficient for conj(Z[j, m])
            
            # grad[j] = sum_m weights[m] * conj(Z[j, m])
            # grad = Z @ weights  (since Z[j, m] = z^j, conj is z^-j)
            # Wait, Z[j, m] = z_m^j.
            # grad_j = sum_m weights_m * conj(z_m^j) = sum_m weights_m * z_m^{-j}
            # This is the inverse DFT of weights, essentially.
            # We can compute this efficiently or just use matrix multiplication.
            # Z is (N, M). weights is (M,).
            # grad = Z @ weights
            
            grad = Z @ weights
            
            # Update x
            x -= lr * grad
            
            # Project to [-1, 1]?
            # Gradient descent on the box constraint.
            # Simple projection:
            x = np.clip(x, -1.0, 1.0)
            
            # Every 10 steps, check the rounded solution
            if step % 10 == 0:
                # Round to nearest +/-1
                x_rounded = np.where(x > 0, 1, -1).astype(np.float64)
                
                # Evaluate true sup norm on a finer grid
                M_eval = 4096
                angles_eval = np.linspace(0, 2 * np.pi, M_eval, endpoint=False)
                z_eval = np.exp(1j * angles_eval)
                
                # Efficient evaluation:
                # P_eval = sum_k x_rounded[k] z_eval^k
                # We can reuse the precomputed Z if M_eval matches, but it doesn't.
                # For M=4096, it's fast enough.
                Z_eval = np.empty((N, M_eval), dtype=np.complex128)
                Z_eval[0, :] = 1.0
                for k in range(1, N):
                    Z_eval[k, :] = Z_eval[k-1, :] * z_eval
                
                P_eval = Z_eval @ x_rounded
                score = np.max(np.abs(P_eval)) / np.sqrt(N)
                
                if score < best_score:
                    best_score = score
                    best_coeffs = x_rounded.copy()
                    
        # After gradient descent, do local search on the best rounded candidate found in this restart
        # Or just on the current rounded x
        x_rounded = np.where(x > 0, 1, -1).astype(np.float64)
        
        # Local search: flip one coefficient at a time
        improved = True
        while improved:
            improved = False
            for k in range(N):
                if time.time() > deadline - 5.0:
                    break
                    
                # Flip coefficient k
                x_test = x_rounded.copy()
                x_test[k] *= -1
                
                # Evaluate
                M_eval = 4096
                angles_eval = np.linspace(0, 2 * np.pi, M_eval, endpoint=False)
                z_eval = np.exp(1j * angles_eval)
                
                Z_eval = np.empty((N, M_eval), dtype=np.complex128)
                Z_eval[0, :] = 1.0
                for k_idx in range(1, N):
                    Z_eval[k_idx, :] = Z_eval[k_idx-1, :] * z_eval
                
                P_eval = Z_eval @ x_test
                score = np.max(np.abs(P_eval)) / np.sqrt(N)
                
                if score < best_score - 1e-12:
                    best_score = score
                    best_coeffs = x_test.copy()
                    x_rounded = x_test.copy()
                    improved = True
                    break # Restart the while loop with the new x_rounded

    if best_coeffs is None:
        # Fallback
        return [1] * 24
        
    return [int(c) for c in best_coeffs]
