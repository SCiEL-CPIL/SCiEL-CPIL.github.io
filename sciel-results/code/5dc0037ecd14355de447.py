import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Target tensor T
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # All other entries are 0.
    
    # Try to find a lower rank decomposition
    # We will try ranks from rank_school - 1 down to rank_school - 5
    # If found, return it.
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Helper to check if a decomposition is valid
    def check_decomposition(U, V, W, rank):
        # Compute tensor product
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T[alpha, beta, gamma] = sum_r U[alpha, r] * V[beta, r] * W[gamma, r]
        # We only need to check non-zero entries of T and some zero entries.
        # Non-zero entries: (i*m+j, j*p+k, k*n+i) for all i,j,k.
        # Zero entries: pick a few random ones to ensure no spurious non-zeros.
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[alpha][r] * V[beta][r] * W[gamma][r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        
        # Check some zero entries
        # Total entries: (n*m) * (m*p) * (p*n)
        # We can't check all. Check random ones.
        for _ in range(100):
            alpha = random.randint(0, n*m - 1)
            beta = random.randint(0, m*p - 1)
            gamma = random.randint(0, p*n - 1)
            
            # If this is a non-zero entry, skip
            # alpha = i*m+j, beta = j'*p+k, gamma = k'*n+i'
            # It is non-zero if j==j' and k==k' and i==i'
            i = alpha // m
            j = alpha % m
            jp = beta // p
            k = beta % p
            kn = gamma // n
            ip = gamma % n
            
            if j == jp and k == kn and i == ip:
                continue
            
            val = 0.0
            for r in range(rank):
                val += U[alpha][r] * V[beta][r] * W[gamma][r]
            if abs(val) > 1e-6:
                return False
                
        return True

    # Helper to snap to half-integers
    def snap_to_half_integer(matrix):
        snapped = []
        for row in matrix:
            snapped_row = []
            for val in row:
                # Round to nearest 0.5
                snapped_val = round(val * 2.0) / 2.0
                snapped_row.append(snapped_val)
            snapped.append(snapped_row)
        return snapped

    # Try to find a lower rank
    for R in range(rank_school - 1, max(rank_school - 10, 0), -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS steps
            for step in range(20):
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each alpha, U[alpha, r] = sum_beta,gamma T[alpha,beta,gamma] V[beta,r] W[gamma,r] / sum_beta,gamma V[beta,r]^2 W[gamma,r]^2
                # This is complex to implement efficiently for sparse T.
                # Given the small size, we can iterate over non-zero T entries.
                
                # Actually, a simpler approach:
                # Fix V and W, solve for U.
                # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                # For a fixed alpha, we have a system of linear equations for U[alpha, r].
                # The equations are for all beta, gamma such that T[alpha, beta, gamma] != 0.
                # For alpha = i*m+j, the non-zero T entries are for beta = j*p+k, gamma = k*n+i.
                # There are p such entries.
                # So for each alpha, we have p equations and R unknowns.
                # If R > p, the system is underdetermined.
                # We can use least squares.
                
                # Let's implement a simple ALS step for U.
                new_U = np.zeros_like(U)
                for alpha in range(n * m):
                    # Collect equations
                    # T[alpha, beta, gamma] = 1 for specific beta, gamma
                    # We need to find beta, gamma such that T[alpha, beta, gamma] != 0
                    # alpha = i*m+j
                    i = alpha // m
                    j = alpha % m
                    for k in range(p):
                        beta = j * p + k
                        gamma = k * n + i
                        # Equation: sum_r U[alpha, r] V[beta, r] W[gamma, r] = 1
                        # This is a linear equation in U[alpha, :]
                        pass
                    
                    # Construct matrix A_alpha and vector b_alpha
                    # A_alpha is p x R
                    # b_alpha is p x 1
                    A_alpha = np.zeros((p, R))
                    b_alpha = np.zeros(p)
                    for idx_k in range(p):
                        k = idx_k
                        beta = j * p + k
                        gamma = k * n + i
                        A_alpha[idx_k, :] = V[beta, :] * W[gamma, :]
                        b_alpha[idx_k] = 1.0
                    
                    # Solve A_alpha * U[alpha, :]^T = b_alpha
                    # Using least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_alpha, b_alpha, rcond=None)
                        new_U[alpha, :] = sol
                    except:
                        new_U[alpha, :] = U[alpha, :]
                        
                U = new_U
                
                # Similarly for V and W
                # For V[beta, r], fix U and W
                # beta = j*p+k
                # Non-zero T entries for this beta:
                # alpha = i*m+j, gamma = k*n+i
                # There are n such entries.
                new_V = np.zeros_like(V)
                for beta in range(m * p):
                    j = beta // p
                    k = beta % p
                    A_beta = np.zeros((n, R))
                    b_beta = np.zeros(n)
                    for idx_i in range(n):
                        i = idx_i
                        alpha = i * m + j
                        gamma = k * n + i
                        A_beta[idx_i, :] = U[alpha, :] * W[gamma, :]
                        b_beta[idx_i] = 1.0
                        
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_beta, b_beta, rcond=None)
                        new_V[beta, :] = sol
                    except:
                        new_V[beta, :] = V[beta, :]
                V = new_V
                
                # For W[gamma, r], fix U and V
                # gamma = k*n+i
                # Non-zero T entries:
                # alpha = i*m+j, beta = j*p+k
                # There are m such entries.
                new_W = np.zeros_like(W)
                for gamma in range(p * n):
                    k = gamma // n
                    i = gamma % n
                    A_gamma = np.zeros((m, R))
                    b_gamma = np.zeros(m)
                    for idx_j in range(m):
                        j = idx_j
                        alpha = i * m + j
                        beta = j * p + k
                        A_gamma[idx_j, :] = U[alpha, :] * V[beta, :]
                        b_gamma[idx_j] = 1.0
                        
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_gamma, b_gamma, rcond=None)
                        new_W[gamma, :] = sol
                    except:
                        new_W[gamma, :] = W[gamma, :]
                W = new_W
                
            # After ALS, check if the solution is close to integer/half-integer
            # And if it is a valid decomposition
            # Snap to half-integers
            U_snapped = snap_to_half_integer(U.tolist())
            V_snapped = snap_to_half_integer(V.tolist())
            W_snapped = snap_to_half_integer(W.tolist())
            
            if check_decomposition(U_snapped, V_snapped, W_snapped, R):
                return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
            
    # If no lower rank found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
