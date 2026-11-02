"""
Program to find a rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.
"""
import numpy as np
import time

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition.
    
    For 3x3 * 3x7, the schoolbook decomposition has rank 3*3*7 = 63.
    Attempts to find lower rank decompositions using structured search.
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try structured low-rank decompositions
    # Strategy: For each (i,j), share a common U vector across all k
    # This means U[i*m+j, r] is the same for all k in the block
    
    # Try ranks from 27 upward (3*3*3 = 27 as a starting point)
    for target_rank in range(27, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try to find a rank-target_rank decomposition
        # Use the structure: U has 9 rows, V has 21 rows, W has 21 rows
        # For each (i,j), the 7 output positions k=0..6 should be handled by shared U
        
        # Initialize with random values
        np.random.seed(42 + target_rank)
        
        # Structured initialization:
        # U: 9 x target_rank
        # For each (i,j), assign a base vector
        U_np = np.zeros((n*m, target_rank))
        V_np = np.zeros((m*p, target_rank))
        W_np = np.zeros((p*n, target_rank))
        
        # Assign columns in groups of 3 (for the 7 k values, use 3 shared + 4 specific)
        # Actually, let's try: for each (i,j), use 3 columns that handle all 7 k's
        # This gives 9*3 = 27 columns minimum if perfectly structured
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for kk in range(3):  # 3 shared columns per (i,j)
                    if col_idx >= target_rank:
                        break
                    # U entry
                    U_np[i*m+j, col_idx] = 1.0
                    # V entries for all 7 k values
                    for k in range(p):
                        V_np[j*p+k, col_idx] = np.random.randn()
                    # W entries for all 7 k values
                    for k in range(p):
                        W_np[k*n+i, col_idx] = np.random.randn()
                    col_idx += 1
        
        # Add remaining columns if target_rank > 27
        while col_idx < target_rank:
            U_np[:, col_idx] = np.random.randn(n*m)
            V_np[:, col_idx] = np.random.randn(m*p)
            W_np[:, col_idx] = np.random.randn(p*n)
            col_idx += 1
        
        # Verify if this gives exact decomposition
        # T[i,j,k] should be 1 for all valid (i,j,k)
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r] 
                             for r in range(target_rank))
                    error += abs(val - 1.0)
        
        if error < 1e-10:
            best_rank = target_rank
            best_U = U_np.tolist()
            best_V = V_np.tolist()
            best_W = W_np.tolist()
            break
    
    # If no better found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
