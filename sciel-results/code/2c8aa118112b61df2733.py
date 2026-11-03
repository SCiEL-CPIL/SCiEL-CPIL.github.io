"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Schoolbook solution
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        if not np.allclose(T, np.ones((n,m,p)), atol=1e-10):
            return False
        return True
    
    def try_reduce_rank(start_rank, end_rank, max_time_per_rank=5.0):
        """Try to find a decomposition with rank between start_rank and end_rank."""
        nonlocal best_rank, best_U, best_V, best_W
        for target_rank in range(end_rank, start_rank - 1, -1):
            if time.time() > deadline - 20:
                break
            rank_start_time = time.time()
            
            # Initialize with random values and optimize
            np.random.seed(42 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Gradient descent to minimize ||T - 1||^2
            for it in range(200):
                if time.time() - rank_start_time > max_time_per_rank:
                    break
                if time.time() > deadline - 20:
                    break
                
                # Compute T
                T = np.zeros((n, m, p))
                for r_idx in range(target_rank):
                    u = U[:, r_idx].reshape(n, m)
                    v = V[:, r_idx].reshape(m, p)
                    w = W[:, r_idx].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                
                error = np.sum((T - 1.0)**2)
                if error < 1e-10:
                    # Found a valid decomposition
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    return True
                
                # Compute gradients (approximate)
                lr = 0.01 / (1 + it * 0.01)
                for r_idx in range(target_rank):
                    u = U[:, r_idx].reshape(n, m)
                    v = V[:, r_idx].reshape(m, p)
                    w = W[:, r_idx].reshape(p, n)
                    
                    du = np.zeros_like(u)
                    dv = np.zeros_like(v)
                    dw = np.zeros_like(w)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                grad = 2 * (T[i,j,k] - 1.0)
                                du[i,j] += grad * v[j,k] * w[k,i]
                                dv[j,k] += grad * u[i,j] * w[k,i]
                                dw[k,i] += grad * u[i,j] * v[j,k]
                    
                    U[:, r_idx] -= lr * du.reshape(-1)
                    V[:, r_idx] -= lr * dv.reshape(-1)
                    W[:, r_idx] -= lr * dw.reshape(-1)
        
        return False
    
    # Try to find lower rank decompositions
    # Start from rank 119 down to rank 100
    try_reduce_rank(100, 119, max_time_per_rank=5.0)
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
