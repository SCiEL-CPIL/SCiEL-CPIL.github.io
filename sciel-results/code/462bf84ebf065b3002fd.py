"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank = n*m*p = 40
    # We will attempt to find a lower-rank decomposition via random search.
    # If we find one, we return it; otherwise, we fall back to schoolbook.
    
    def schoolbook():
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
    
    def verify(U, V, W, tol=1e-9):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check that for all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # and for all other (i', j', k') combinations where the tensor entry is 0,
        # the sum should be 0.
        # 
        # The matrix multiplication tensor T[i, j, k] for multiplying A (n x m) by B (m x p)
        # has T[i, j, k] = 1 if the (i, k) entry of C = A*B depends on A[i, j] and B[j, k],
        # which is always true for all i, j, k. So T[i, j, k] = 1 for all i, j, k.
        # Wait, that's not right. The tensor for matrix multiplication is:
        # C[i, k] = sum_j A[i, j] * B[j, k]
        # So the tensor T has entries T[i, j, k] = 1 for all i, j, k.
        # 
        # Actually, the standard convention is:
        # T[i, j, k] = 1 if we multiply A[i, j] by B[j, k] to contribute to C[i, k].
        # So T[i, j, k] = 1 for all i, j, k in [0,n), [0,m), [0,p).
        # 
        # The decomposition is:
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[i, k, r]
        # where U has shape (n, m, rank), V has shape (m, p, rank), W has shape (n, p, rank).
        # In the flattened form:
        # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r] (per the problem statement)
        # 
        # So we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank = U_arr.shape[1]
        
        # Build the tensor and check
        # For efficiency, we can compute the tensor via einsum
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (n, p, rank)
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)  # W[k*n+i, r] -> W_3d[k, i, r]
        
        # Compute T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        # This is a bit tricky with einsum. Let's do it step by step.
        
        # First, compute U_3d[i, j, r] * V_3d[j, k, r] for all i, j, k, r
        # = sum over r of U_3d[i, j, r] * V_3d[j, k, r]
        # Let's call this M[i, j, k, r] = U_3d[i, j, r] * V_3d[j, k, r]
        # Then T[i, j, k] = sum_r M[i, j, k, r] * W_3d[k, i, r]
        
        # For small dimensions, we can just loop
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        
        return max_err < tol
    
    def random_search(max_rank, num_trials=10000):
        """Try to find a decomposition of rank max_rank."""
        np.random.seed(42)
        
        for trial in range(num_trials):
            if time.time() > _deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, max_rank) * 0.5
            V = np.random.randn(m * p, max_rank) * 0.5
            W = np.random.randn(p * n, max_rank) * 0.5
            
            # Simple gradient-free local search: try small perturbations
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(20):
                if time.time() > _deadline - 10:
                    break
                
                # Compute current error
                U_3d = U.reshape(n, m, max_rank)
                V_3d = V.reshape(m, p, max_rank)
                W_3d = W.reshape(p, n, max_rank)
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(max_rank):
                                s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-9:
                    # Found a solution!
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
                
                # Perturb
                U += np.random.randn(n * m, max_rank) * 0.1
                V += np.random.randn(m * p, max_rank) * 0.1
                W += np.random.randn(p * n, max_rank) * 0.1
            
            if best_err < 1e-9:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None
    
    # Try ranks from 39 down to 30
    for rank in range(39, 29, -1):
        if time.time() > _deadline - 30:
            break
        
        result = random_search(rank, num_trials=500)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
