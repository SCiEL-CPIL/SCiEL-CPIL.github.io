"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c]."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=atol)

    def schoolbook():
        """Return the standard rank-56 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_strassen_block_decomposition():
        """
        Try a Strassen-like block decomposition.
        Partition the 14 columns into 7 pairs and apply 7-term Strassen per pair.
        Total rank: 7 * 7 = 49.
        """
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        
        for l in range(7):
            col_start = 7 * l
            k1 = 2 * l
            k2 = 2 * l + 1
            
            # M1: (A00+A11)(B0,k1 + B1,k2)
            U[0, col_start] = 1
            U[3, col_start] = 1
            V[k1, col_start] = 1
            V[p + k2, col_start] = 1
            W[k1 * n, col_start] = 1
            W[k2 * n + 1, col_start] = 1
            
            # M2: (A10+A11) * B0,k1
            U[2, col_start+1] = 1
            U[3, col_start+1] = 1
            V[k1, col_start+1] = 1
            W[k1 * n + 1, col_start+1] = 1
            W[k2 * n + 1, col_start+1] = -1
            
            # M3: A00 * (B0,k2 - B1,k2)
            U[0, col_start+2] = 1
            V[k2, col_start+2] = 1
            V[p + k2, col_start+2] = -1
            W[k2 * n, col_start+2] = 1
            W[k2 * n + 1, col_start+2] = 1
            
            # M4: A11 * (B1,k1 - B0,k1)
            U[3, col_start+3] = 1
            V[p + k1, col_start+3] = 1
            V[k1, col_start+3] = -1
            W[k1 * n, col_start+3] = 1
            W[k1 * n + 1, col_start+3] = 1
            
            # M5: (A00+A01) * B1,k2
            U[0, col_start+4] = 1
            U[1, col_start+4] = 1
            V[p + k2, col_start+4] = 1
            W[k1 * n, col_start+4] = -1
            W[k2 * n, col_start+4] = 1
            
            # M6: (A10-A00) * (B0,k1 + B0,k2)
            U[2, col_start+5] = 1
            U[0, col_start+5] = -1
            V[k1, col_start+5] = 1
            V[k2, col_start+5] = 1
            W[k2 * n + 1, col_start+5] = 1
            
            # M7: (A01-A11) * (B1,k1 + B1,k2)
            U[1, col_start+6] = 1
            U[3, col_start+6] = -1
            V[p + k1, col_start+6] = 1
            V[p + k2, col_start+6] = 1
            W[k1 * n, col_start+6] = 1
        
        return U, V, W

    def try_rank_reduction():
        """
        Try to reduce rank below 49 by finding linear dependencies among the 49 columns.
        If the 49 rank-1 tensors are linearly dependent, we can express some as combinations
        of others, potentially reducing the rank.
        
        Strategy: Check if the 49 vectors (flattened tensors) are linearly independent.
        If not, find a basis and use it.
        """
        if time.time() > deadline:
            return None
        
        # Start with the Strassen-49 decomposition
        U49, V49, W49 = try_strassen_block_decomposition()
        
        if not verify(U49, V49, W49):
            return None
        
        # Flatten each rank-1 tensor into a vector
        # Tensor T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Flatten: T_r[a,b,c] -> T_r[a * size_b * size_c + b * size_c + c]
        flat_dim = size_a * size_b * size_c
        tensors = np.zeros((49, flat_dim), dtype=np.float64)
        
        for r in range(49):
            T_r = np.einsum('a,b,c->abc', U49[:,r], V49[:,r], W49[:,r])
            tensors[r] = T_r.flatten()
        
        # Check rank of the 49x(4*28*28) matrix
        # Use SVD to find the numerical rank
        try:
            U_svd, s, Vt = np.linalg.svd(tensors, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        
        # Find numerical rank (singular values > tolerance)
        tol = 1e-10
        num_rank = np.sum(s > tol * s[0])
        
        if num_rank >= 49:
            # All 49 tensors are linearly independent, no reduction possible
            return None
        
        # Use only the first num_rank singular vectors as basis
        # But we need to express the original tensor as sum of rank-1 terms
        # This is tricky - let's try a different approach
        
        # Alternative: Try to find if any column is a linear combination of others
        # by solving for coefficients
        # This is expensive, so skip for now
        return None

    # First try the constructive rank-49 Strassen decomposition
    U, V, W = try_strassen_block_decomposition()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If that didn't work, check with looser tolerance
    if verify(U, V, W, atol=1e-6):
        # Try to clean up
        U_clean = np.round(U * 2) / 2
        V_clean = np.round(V * 2) / 2
        W_clean = np.round(W * 2) / 2
        if verify(U_clean, W_clean, W_clean):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 56
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
