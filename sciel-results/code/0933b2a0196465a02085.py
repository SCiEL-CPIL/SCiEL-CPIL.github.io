import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Target Tensor T
    # T[alpha, beta, gamma] = 1 if (i, j, k) is valid, else 0
    # alpha = i*m + j
    # beta  = j*p + k
    # gamma = k*n + i
    
    T = np.zeros((size_U, size_V, size_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(size_U, size_V, size_W) = 15
    # Schoolbook is size_U * size_V * size_W / (size_U * size_V) ? No, rank is n*m*p = 30.
    # Actually, the rank of the matmul tensor is at most n*m*p.
    # We try R from 15 to 30.
    
    # We will use a random search strategy:
    # 1. Randomly generate U and V with small integer/half-integer entries.
    # 2. Solve for W using linear least squares for each gamma.
    # 3. Check if W entries are half-integers and if the decomposition is exact.
    
    possible_values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for R in range(15, 31):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        # We can afford many trials if they are fast.
        # Solving 10 linear systems of size 90xR is fast.
        num_trials = 5000 if R > 25 else 10000
        
        for _ in range(num_trials):
            if time.time() > deadline:
                break
                
            # Generate random U and V
            # Using small values to increase chance of exact half-integer W
            U = np.random.choice(possible_values, size=(size_U, R))
            V = np.random.choice(possible_values, size=(size_V, R))
            
            W = np.zeros((size_W, R))
            valid_W = True
            
            for gamma in range(size_W):
                # Construct A_gamma: (size_U * size_V) x R
                # A_gamma[(alpha, beta), r] = U[alpha, r] * V[beta, r]
                # We can vectorize this.
                # U is (size_U, R), V is (size_V, R)
                # We need the outer product of each column? No.
                # A_gamma is a matrix where row index is (alpha, beta) and col is r.
                # A_gamma[alpha, beta, r] = U[alpha, r] * V[beta, r]
                
                # Flatten T[:, :, gamma] to get target vector
                target = T[:, :, gamma].flatten()
                
                # Construct A
                # Using einsum: 'ar,br->abr' then reshape
                A = np.einsum('ar,br->abr', U, V).reshape(-1, R)
                
                # Solve A W_gamma = target
                # A is (90, R), target is (90,)
                try:
                    W_gamma, residuals, rank_A, s = np.linalg.lstsq(A, target, rcond=None)
                except np.linalg.LinAlgError:
                    valid_W = False
                    break
                    
                # Check if W_gamma entries are close to half-integers
                # Round to nearest half-integer
                W_gamma_snapped = np.round(W_gamma * 2) / 2.0
                
                # Check if the snapped values satisfy the equation exactly (or very closely)
                # Residual check
                residual = A @ W_gamma_snapped - target
                if np.max(np.abs(residual)) > 1e-9:
                    valid_W = False
                    break
                    
                W[gamma, :] = W_gamma_snapped
            
            if valid_W:
                # Final verification
                # Compute T_recon = sum_r U[:, r] V[:, r] W[:, r]
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_recon, T, atol=1e-9):
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
