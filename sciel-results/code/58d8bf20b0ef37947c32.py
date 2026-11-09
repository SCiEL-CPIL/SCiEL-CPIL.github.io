import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Construct the full tensor T[i, j, k] = 1 if i, j, k are valid indices
    # T is of shape (n, m, p)
    T = np.ones((n, m, p))
    
    # Try to find a lower rank decomposition using ALS
    # Lower bound for rank is roughly (n*m*p) / max(n,m,p) but let's just try a range
    # Known lower bound for matrix multiplication tensor rank is not trivially small.
    # Let's try ranks from 15 to 29.
    
    best_solution = None
    best_rank = rank_school
    
    # We will use ALS with random initialization
    rng = np.random.default_rng(42)
    
    for r in range(15, rank_school):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() > deadline:
                break
                
            # Update W: For each (k, i), W[k*n+i, :] = (T[i, :, k] * U[:, :].T) @ V
            # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # Fix U, V. Solve for W.
            # For each k, i:
            # W[k*n+i, :] = ( sum_j T[i,j,k] U[i*m+j, :] V[j*p+k, :].T ) / ( sum_j V[j*p+k, :]^2 )? No.
            # Let's reshape.
            # T_ik = sum_j T[i,j,k] U[i*m+j, :] V[j*p+k, :].T
            # This is a linear system for W[k*n+i, :].
            
            # Efficiently:
            # For each k:
            #   T_k is (n, m, r) where T_k[i, j, c] = T[i,j,k] * U[i*m+j, c]
            #   V_k is (m, p, r) where V_k[j, c] = V[j*p+k, c]
            #   Actually, for fixed k, we want to find W_k (n, r) such that:
            #   T[:, :, k] = U @ W_k @ V_k.T (element-wise? No, tensor contraction)
            #   T[i, j, k] = sum_c U[i*m+j, c] W[k*n+i, c] V[j*p+k, c]
            
            # Let's update W for all (k, i) at once.
            # W_new[k*n+i, :] = ( (T[:, :, k] * U.reshape(n, m, r)) @ V_k.T ) / (V_k @ V_k.T) ?
            # No, the contraction is over j.
            # sum_j T[i,j,k] U[i*m+j, c] V[j*p+k, d] = sum_c W[k*n+i, c] ... No.
            
            # Let's do it iteratively for clarity, it's small enough.
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # We want to solve for w = W[k*n+i, :]
                    # T[i, :, k] = sum_j U[i*m+j, :] * V[j*p+k, :]^T * w  (outer product sum)
                    # Let A = sum_j T[i,j,k] U[i*m+j, :] (1, r)
                    # Let B = V[:, k] (m, r)
                    # We want w such that A @ B.T * w? No.
                    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                    # For fixed i, k:
                    # T[i, :, k] (length m) = (U_i @ W_i @ V_k.T) ?
                    # U_i is (m, r), V_k is (m, r).
                    # sum_c U_i[j, c] V_k[j, c] W_i[c]
                    # This is element-wise product sum.
                    # Let M = U_i * V_k (m, r) element-wise.
                    # T[i, :, k] = M @ W_i
                    # So W_i = pinv(M) @ T[i, :, k]
                    
                    U_i = U[i*m:(i+1)*m, :] # (m, r)
                    V_k = V[:, k*p:k*p+p] # No, V is (m*p, r). V[j*p+k, :] is row j*p+k.
                    # V_k should be (m, r) where V_k[j, c] = V[j*p+k, c]
                    V_k = np.array([V[j*p+k, :] for j in range(m)]) # (m, r)
                    
                    M = U_i * V_k # (m, r)
                    b = T[i, :, k] # (m,)
                    
                    # Solve M @ w = b
                    try:
                        w = np.linalg.lstsq(M, b, rcond=None)[0]
                    except:
                        w = np.zeros(r)
                    W_new[k*n+i, :] = w
            
            W = W_new
            
            # Similarly update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # T[i, j, :] = sum_k U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                    # Fix V, W.
                    # V_j is (p, r) where V_j[k, c] = V[j*p+k, c]
                    # W_i is (p, r) where W_i[k, c] = W[k*n+i, c]
                    V_j = np.array([V[j*p+k, :] for k in range(p)]) # (p, r)
                    W_i = np.array([W[k*n+i, :] for k in range(p)]) # (p, r)
                    
                    M = V_j * W_i # (p, r)
                    b = T[i, j, :] # (p,)
                    
                    try:
                        u = np.linalg.lstsq(M, b, rcond=None)[0]
                    except:
                        u = np.zeros(r)
                    U_new[i*m+j, :] = u
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # T[:, j, k] = sum_i U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                    # Fix U, W.
                    U_j = np.array([U[i*m+j, :] for i in range(n)]) # (n, r)
                    W_k = np.array([W[k*n+i, :] for i in range(n)]) # (n, r)
                    
                    M = U_j * W_k # (n, r)
                    b = T[:, j, k] # (n,)
                    
                    try:
                        v = np.linalg.lstsq(M, b, rcond=None)[0]
                    except:
                        v = np.zeros(r)
                    V_new[j*p+k, :] = v
            V = V_new
            
            # Check convergence / exactness
            # Reconstruct T_hat
            T_hat = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_hat[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :]) * np.dot(np.ones(r), W[k*n+i, :]) # Wait, no.
                        # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                        T_hat[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        
            error = np.linalg.norm(T - T_hat)
            if error < 1e-9:
                # Try to round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                T_hat_round = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_hat_round[i, j, k] = np.sum(U_round[i*m+j, :] * V_round[j*p+k, :] * W_round[k*n+i, :])
                            
                if np.linalg.norm(T - T_hat_round) < 1e-9:
                    # Found exact half-integer solution
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                else:
                    # Return the floating point solution if it's exact enough? 
                    # The task says "integer/half-integer entries". 
                    # So we must return half-integers.
                    pass
                    
            if time.time() > deadline:
                break

    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
