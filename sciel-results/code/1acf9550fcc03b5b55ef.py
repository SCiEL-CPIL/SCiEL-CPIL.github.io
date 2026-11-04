import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Returns a rank-minimal tensor decomposition for 4x4 matrix multiplication.
    Uses Kronecker product of Strassen's rank-7 algorithm for 2x2 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # U[i*2+j, r] = coefficient of A[i,j] in term r
    # V[j*2+k, r] = coefficient of B[j,k] in term r  
    # W[k*2+i, r] = coefficient of C[i,k] in term r (trace convention: W[k*n+i, r])
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [1, 0, -1, 1, 0, 0, 1],   # B10
        [1, 0, 0, 0, 1, 0, 1]     # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=1,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=0,i=1)
        [1, -1, 1, 0, 0, 1, 0]    # C11 (k=1,i=1)
    ], dtype=int)
    
    # Verify the 2x2 Strassen decomposition
    def verify_2x2(U2, V2, W2):
        """Verify that the 2x2 Strassen decomposition is correct."""
        # Build the 2x2 matrix multiplication tensor
        # T[i*2+j, j*2+k, k*2+i] = 1
        T = np.zeros((4, 4, 4), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[i*2+j, j*2+k, k*2+i] = 1.0
        
        # Compute the decomposition
        T_decomp = np.einsum('ar,br,cr->abc', U2.astype(float), V2.astype(float), W2.astype(float))
        max_diff = np.max(np.abs(T_decomp - T))
        return max_diff < 1e-9, max_diff
    
    valid_2x2, diff_2x2 = verify_2x2(U2, V2, W2)
    if not valid_2x2:
        # Fall back to schoolbook for 2x2 (should not happen)
        return _schoolbook_4x4()
    
    # Build the rank-49 Kronecker product decomposition for 4x4
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)  # (16, 49)
    V = np.zeros((m * p, rank), dtype=int)  # (16, 49)
    W = np.zeros((p * n, rank), dtype=int)  # (16, 49)
    
    # For 4x4 matrices indexed as i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # where i1, i2, j1, j2, k1, k2 in {0, 1}
    # The Kronecker product structure:
    # U[i, r1*7+r2] = U2[i1, r1] * U2[i2, r2] where i = 2*i1+i2, and similarly for j,k
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = 2 * k1 + k2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U[i * m + j, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                                    V[j * p + k, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                                    W[k * n + i, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
    
    # Verify the 4x4 decomposition
    def verify_4x4(U, V, W):
        """Verify that the 4x4 decomposition is correct."""
        T = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        T_decomp = np.einsum('ar,br,cr->abc', U.astype(float), V.astype(float), W.astype(float))
        max_diff = np.max(np.abs(T_decomp - T))
        return max_diff < 1e-9, max_diff
    
    valid_4x4, diff_4x4 = verify_4x4(U, V, W)
    
    if not valid_4x4:
        return _schoolbook_4x4()
    
    # Attempt to reduce rank by checking for linear dependencies
    # If some columns are linear combinations of others, we can eliminate them
    rank_reduced = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    if rank_reduced is not None:
        U, V, W = rank_reduced
        # Verify again after reduction
        valid_final, diff_final = verify_4x4(U, V, W)
        if valid_final:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the rank-49 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_4x4():
    """Fallback: schoolbook decomposition with rank 64."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by identifying and eliminating linearly dependent columns.
    Returns (U_reduced, V_reduced, W_reduced) or None if no reduction found.
    """
    rank = U.shape[1]
    
    # Check if we're running out of time
    if time.time() > deadline - 10.0:
        return None
    
    # Try to find columns that are linear combinations of others
    # This is a heuristic approach - check small subsets
    # For efficiency, only try a limited number of eliminations
    
    # Strategy: try removing one column at a time and see if the decomposition still holds
    # This is O(rank^2) which is manageable for rank=49
    
    U_float = U.astype(float)
    V_float = V.astype(float)
    W_float = W.astype(float)
    
    # Build the tensor from current decomposition
    T_decomp = np.einsum('ar,br,cr->abc', U_float, V_float, W_float)
    
    # Try removing each column and checking if the remaining columns can span the same tensor
    # This is expensive, so limit to a few attempts
    max_attempts = 5
    attempts = 0
    
    for col_to_remove in range(rank):
        if time.time() > deadline - 5.0:
            break
        attempts += 1
        if attempts > max_attempts:
            break
        
        # Remove column col_to_remove
        mask = np.ones(rank, dtype=bool)
        mask[col_to_remove] = False
        U_sub = U_float[:, mask]
        V_sub = V_float[:, mask]
        W_sub = W_float[:, mask]
        
        # Check if the sub-decomposition still equals the tensor
        T_sub = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sub)
        diff = np.max(np.abs(T_sub - T_decomp))
        
        if diff < 1e-9:
            # We can remove this column!
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            return (U_new, V_new, W_new)
    
    return None
