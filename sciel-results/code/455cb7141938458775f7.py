"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools

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
        Try to reduce rank below 49 by finding linear dependencies among the 49 rank-1 tensors.
        Uses a structural approach: since the 2x2x14 tensor is the tensor product of
        the 2x2x2 tensor and the identity tensor of size 14, we can exploit this structure.
        
        The key insight: the 49 rank-1 terms form a direct sum of 7 copies of the rank-7
        Strassen decomposition. If there are linear dependencies, they must respect the
        block structure or involve cross-block combinations.
        
        We use a more efficient approach: instead of checking all 49 terms at once,
        we check each block separately and then look for cross-block dependencies.
        """
        if time.time() > deadline:
            return None
        
        # Start with the Strassen-49 decomposition
        U_curr, V_curr, W_curr = try_strassen_block_decomposition()
        
        if not verify(U_curr, V_curr, W_curr):
            return None
        
        current_rank = 49
        reduced_U = U_curr.astype(np.float64).copy()
        reduced_V = V_curr.astype(np.float64).copy()
        reduced_W = W_curr.astype(np.float64).copy()
        
        # Build the tensor matrix for all 49 terms
        tensors = np.zeros((current_rank, size_a * size_b * size_c), dtype=np.float64)
        for r in range(current_rank):
            T_r = np.einsum('a,b,c->abc', reduced_U[:,r], reduced_V[:,r], reduced_W[:,r])
            tensors[r] = T_r.flatten()
        
        # Find the numerical rank
        try:
            U_svd, s, Vt = np.linalg.svd(tensors, full_matrices=True)
        except np.linalg.LinAlgError:
            return None
        
        # Determine numerical rank
        tol = 1e-10
        if len(s) == 0:
            return None
        num_rank = np.sum(s > tol * s[0])
        
        if num_rank >= current_rank:
            # No null space, cannot reduce further
            return None
        
        # Get null space vectors
        null_dim = current_rank - num_rank
        null_vecs = Vt[num_rank:, :]
        
        # Try to find a valid reduction by eliminating one term at a time
        for nv in null_vecs:
            if time.time() > deadline:
                break
            
            # Find index k with largest |nv[k]|
            k = np.argmax(np.abs(nv))
            if abs(nv[k]) < 1e-8:
                continue
            
            # Try to eliminate term k
            # If sum_r nv[r] T_r = 0, then T_k = -sum_{i≠k} (nv[i]/nv[k]) T_i
            # We can remove term k if we can absorb it into the others
            # This is valid if the resulting decomposition still has rank-1 terms
            
            # Check if we can simply zero out term k and rescale others
            # This works if nv[k] is the dominant coefficient
            
            # Create a new decomposition without term k
            new_U = reduced_U.copy()
            new_V = reduced_V.copy()
            new_W = reduced_W.copy()
            
            # Set term k to zero
            new_U[:, k] = 0
            new_V[:, k] = 0
            new_W[:, k] = 0
            
            # Rescale other terms to compensate
            for i in range(current_rank):
                if i == k:
                    continue
                if abs(nv[i]) > 1e-10:
                    scale = -nv[i] / nv[k]
                    # This is a heuristic; proper absorption requires more care
                    # For now, just check if zeroing out works
                    pass
            
            # Verify if the new decomposition is valid
            if verify(new_U, new_V, new_W):
                # Success! We reduced the rank
                # Remove the zero column
                mask = np.any(np.abs(new_U) > 1e-10, axis=0) | np.any(np.abs(new_V) > 1e-10, axis=0) | np.any(np.abs(new_W) > 1e-10, axis=0)
                idx = np.where(mask)[0]
                return new_U[:, idx], new_V[:, idx], new_W[:, idx]
        
        return None

    # First try the constructive rank-49 Strassen decomposition
    U, V, W = try_strassen_block_decomposition()
    if verify(U, V, W):
        # Try to reduce rank further
        reduced = try_rank_reduction()
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if verify(U_r, V_r, W_r):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If that didn't work, check with looser tolerance
    if verify(U, V, W, atol=1e-6):
        # Try to clean up
        U_clean = np.round(U * 2) / 2
        V_clean = np.round(V * 2) / 2
        W_clean = np.round(W * 2) / 2
        if verify(U_clean, V_clean, W_clean):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 56
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
