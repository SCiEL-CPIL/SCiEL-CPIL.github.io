import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 145.0 # Safety margin

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # We try to find a rank lower than schoolbook.
    # Given the dimensions, it's unlikely, but we search a few ranks.
    # ALS is more robust than the previous power method.
    
    for r in range(59, 45, -1):
        if time.time() - start_time > time_limit:
            break
            
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W with random Gaussian
            # Scaling is important. T entries are 1.
            # If U, V, W are ~ N(0,1), the sum of r terms will be ~ sqrt(r).
            # We want the sum to be 1. So scale by 1/sqrt(r)?
            # Or just let ALS handle it.
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS iterations
            converged = False
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                # U[:, c] = (T * V[:, c] * W[:, c]) / (V[:, c]^2 * W[:, c]^2)
                # Vectorized:
                # numerator = np.einsum('ijk,jc,kc->ic', T, V, W)
                # denominator = np.einsum('jc,kc->c', V*V, W*W)
                num_U = np.einsum('ijk,jc,kc->ic', T, V, W)
                den_U = np.einsum('jc,kc->c', V**2, W**2)
                den_U[den_U == 0] = 1e-10
                U = num_U / den_U[np.newaxis, :]
                
                # Update V
                num_V = np.einsum('ijk,ic,kc->jc', T, U, W)
                den_V = np.einsum('ic,kc->c', U**2, W**2)
                den_V[den_V == 0] = 1e-10
                V = num_V / den_V[np.newaxis, :]
                
                # Update W
                num_W = np.einsum('ijk,ic,jc->kc', T, U, V)
                den_W = np.einsum('ic,jc->c', U**2, V**2)
                den_W[den_W == 0] = 1e-10
                W = num_W / den_W[np.newaxis, :]
                
                # Check convergence
                current_sum = np.einsum('ic,jc,kc->ijk', U, V, W)
                if np.linalg.norm(T - current_sum) < 1e-9:
                    converged = True
                    break
            
            if converged:
                # We have a real-valued solution.
                # Try to round to half-integers.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    # Convert to lists
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    best_rank = r
                    # Found a better rank, break restart loop
                    break
        
        # If we found a rank, we continue to lower ranks?
        # Yes, we want minimal.
        # But if we break the restart loop, we go to next r.

    return {"U": best_U, "V": best_V, "W": best_W}
