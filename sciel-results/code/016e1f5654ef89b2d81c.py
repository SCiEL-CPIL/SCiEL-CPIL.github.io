import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank using optimization
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition
        # Use a random initialization and gradient descent
        success = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize
            for t in range(rank):
                norm_U = np.linalg.norm(U[:,t])
                norm_V = np.linalg.norm(V[:,t])
                norm_W = np.linalg.norm(W[:,t])
                if norm_U > 0 and norm_V > 0 and norm_W > 0:
                    U[:,t] /= norm_U
                    V[:,t] /= norm_V
                    W[:,t] /= norm_W
            
            # Check if the decomposition is valid
            # T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
            # This is equivalent to: for all (a,b,c), T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            
            # Compute the reconstructed tensor
            # Reconstructed[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            # This is a tensor of shape (n*m, m*p, p*n)
            
            # We can check this by: for each t, the outer product U[:,t] ⊗ V[:,t] ⊗ W[:,t]
            # Sum over t
            
            # Efficient check: T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            
            error = np.linalg.norm(T_recon - T)
            if error < 1e-6:
                success = True
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if success:
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
