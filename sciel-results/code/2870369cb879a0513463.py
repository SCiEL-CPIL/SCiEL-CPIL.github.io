import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    # Strategy: Use Strassen-like block decomposition
    # Partition A (3x4) into blocks, B (4x4) into blocks
    # Try to find low-rank decomposition of the block multiplication tensor
    
    # First, let's try a direct approach: search for rank < 48 decomposition
    # The tensor T[i,j,k] = A[i,j] * B[j,k]
    # We want T = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    
    # Let's try rank 39, 40, 41, 42, 43, 44, 45, 46, 47
    # Start with lower ranks and work up
    
    # For efficiency, we'll use the fact that we can fix U and V and solve for W
    # T_flat = sum_r (U[:,r] ⊗ V[:,r]) * W[:,r]
    # This is a linear system in W for fixed U, V
    
    def try_rank(rank):
        if time.time() > deadline - 10:
            return None
        
        # Generate random U, V with small integer entries
        # U: (n*m) x rank = 12 x rank
        # V: (m*p) x rank = 16 x rank
        # W: (p*n) x rank = 12 x rank
        
        target = np.ones((n, m, p))
        target_flat = target.flatten()
        
        # Number of trials depends on rank
        max_trials = 500 if rank <= 42 else 200
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random integer U, V in {-1, 0, 1}
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            # Build linear system for W
            # For each output (i,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Group by W entries: for fixed (k,i), W[k*n+i, r] appears in equations for all j
            # T[i,j,k] = sum_j... wait, T[i,j,k] = A[i,j]*B[j,k]
            # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # For fixed (i,k), the equation for each j is:
            # T[i,j,k] = sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
            # This is a system of m equations in rank unknowns for each (i,k) pair
            
            # Let's build the full system
            # Total equations: n*m*p = 48
            # Total unknowns: p*n*rank = 12*rank
            
            A_mat = np.zeros((n * m * p, p * n * rank))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(rank):
                            col_idx = k * n * rank + i * rank + r
                            A_mat[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
            
            # Solve A_mat @ W_flat = target_flat
            # Use least squares
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target_flat, rcond=None)
            except:
                continue
            
            # Check if solution is close to integer
            W_int = np.round(W_flat)
            error = np.linalg.norm(A_mat @ W_int - target_flat)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
        
        return None
    
    # Try ranks from 39 to 47
    for rank in range(39, 48):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            # Convert to list format
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
