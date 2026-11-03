"""Matrix multiplication 4,4,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s safety margin
    
    # Schoolbook rank as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try structured decompositions
    # Strategy: Use block structure to reduce rank
    
    # For 4x4x6, try to exploit that 6 = 4 + 2
    # Compute 4x4x4 and 4x4x2 separately
    
    # 4x4x4 decomposition using 2x2x2 Strassen blocks
    # 4x4x4 can be viewed as 2x(2x2x2) blocks
    # Using Strassen for 2x2x2 gives rank 7 per block
    
    # Let's try a direct construction for 4x4x4 with rank ~ 28-32
    # and 4x4x2 with rank ~ 16-20, total ~ 44-52
    
    # Actually, let's try a more systematic approach
    # Use the fact that the tensor is separable in certain ways
    
    # Try rank values from low to high
    # For each rank, try to find a decomposition using iterative methods
    
    def try_decomposition(rank):
        """Try to find a rank-r decomposition using iterative approach."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Target tensor: T[i,j,k] = 1 for all i,j,k
        # Reshape to (n*m, m*p, p*n) tensor
        # T[im+j, jp+k, kn+i] = 1
        
        # The constraint is: for all (i,j,k):
        # sum_c U[4i+j, c] * V[6j+k, c] * W[4k+i, c] = 1
        
        # This is a 3-way tensor completion problem
        
        # Use alternating least squares / iterative projection
        for iteration in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), minimize sum_k sum_c (U[4i+j,c] * V[6j+k,c] * W[4k+i,c] - 1)^2
            # This is a linear system in U[4i+j, :]
            for i in range(n):
                for j in range(m):
                    # For fixed (i,j), we have variables U[4i+j, 0:rank]
                    # For each k, the equation is:
                    # sum_c U[4i+j,c] * V[6j+k,c] * W[4k+i,c] = 1
                    # Let A_k[c] = V[6j+k,c] * W[4k+i,c]
                    # Then: sum_c U[4i+j,c] * A_k[c] = 1 for each k
                    # This is: A @ U[4i+j, :]^T = 1 (vector of ones)
                    # A is p x rank
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A @ x = 1 for x = U[4i+j, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = x
                    except:
                        pass
            
            # Update V: for each (j,k), minimize sum_i sum_c (U[4i+j,c] * V[6j+k,c] * W[4k+i,c] - 1)^2
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j * p + k, :] = x
                    except:
                        pass
            
            # Update W: for each (k,i), minimize sum_j sum_c (U[4i+j,c] * V[6j+k,c] * W[4k+i,c] - 1)^2
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = x
                    except:
                        pass
        
        # Check if we got a valid decomposition
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-6:
            # Convert to list of lists with rounded values
            U_list = [[round(U[i, c], 6) for c in range(rank)] for i in range(n * m)]
            V_list = [[round(V[i, c], 6) for c in range(rank)] for i in range(m * p)]
            W_list = [[round(W[i, c], 6) for c in range(rank)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    for rank in range(40, 96):
        if time.time() > deadline - 10:
            break
        
        result = try_decomposition(rank)
        if result is not None:
            best_rank = rank
            best_result = result
            break  # Found minimal rank
    
    return best_result
# EVOLVE-BLOCK-END

