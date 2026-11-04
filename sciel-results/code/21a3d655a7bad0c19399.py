"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 130
    
    # Build the full matrix multiplication tensor T[i,j,k] = 1 if this entry exists
    # The tensor is of shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # The tensor entry is 1 if the indices are consistent (i.e., they form a valid triple)
    
    # Actually, the matrix multiplication tensor has entries:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k
    # All other entries are 0.
    # But wait - that's not quite right. The tensor for matrix multiplication is:
    # M = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # So T[a,b,c] = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    
    # Let's build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now we want to find the minimum rank r such that T can be written as
    # T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c has shape (n*m,), v_c has shape (m*p,), w_c has shape (p*n,)
    
    # Strategy: Use iterative rank-1 peeling with integer/half-integer constraints.
    # This is a hard problem in general, but for small tensors we can try:
    # 1. Start with known good decompositions (Strassen-based)
    # 2. Try to reduce rank by finding linear dependencies
    
    # First, let's build a Strassen-based decomposition as a starting point
    def _build_strassen_decomposition(n, m, p):
        """Build a decomposition using Strassen for 2x2x2 blocks and schoolbook for the rest."""
        # For n=3, m=4, p=8:
        # We can partition as:
        # i: [0,2), [2,3) -> sizes 2, 1
        # j: [0,2), [2,4) -> sizes 2, 2  
        # k: [0,2), [2,4), [4,6), [6,8) -> sizes 2, 2, 2, 2
        
        # Blocks:
        # (i0=0,i1=2, j0=0,j1=2, k0=0,k1=2): 2x2x2 -> Strassen (rank 7)
        # (i0=0,i1=2, j0=0,j1=2, k0=2,k1=4): 2x2x2 -> Strassen (rank 7)
        # (i0=0,i1=2, j0=0,j1=2, k0=4,k1=6): 2x2x2 -> Strassen (rank 7)
        # (i0=0,i1=2, j0=0,j1=2, k0=6,k1=8): 2x2x2 -> Strassen (rank 7)
        # (i0=0,i1=2, j0=2,j1=4, k0=0,k1=2): 2x2x2 -> Strassen (rank 7)
        # (i0=0,i1=2, j0=2,j1=4, k0=2,k1=4): 2x2x2 -> Strassen (rank 7)
        # (i0=0,i1=2, j0=2,j1=4, k0=4,k1=6): 2x2x2 -> Strassen (rank 7)
        # (i0=0,i1=2, j0=2,j1=4, k0=6,k1=8): 2x2x2 -> Strassen (rank 7)
        # (i0=2,i1=3, j0=0,j1=2, k0=0,k1=2): 1x2x2 -> schoolbook (rank 4)
        # (i0=2,i1=3, j0=0,j1=2, k0=2,k1=4): 1x2x2 -> schoolbook (rank 4)
        # (i0=2,i1=3, j0=0,j1=2, k0=4,k1=6): 1x2x2 -> schoolbook (rank 4)
        # (i0=2,i1=3, j0=0,j1=2, k0=6,k1=8): 1x2x2 -> schoolbook (rank 4)
        # (i0=2,i1=3, j0=2,j1=4, k0=0,k1=2): 1x2x2 -> schoolbook (rank 4)
        # (i0=2,i1=3, j0=2,j1=4, k0=2,k1=4): 1x2x2 -> schoolbook (rank 4)
        # (i0=2,i1=3, j0=2,j1=4, k0=4,k1=6): 1x2x2 -> schoolbook (rank 4)
        # (i0=2,i1=3, j0=2,j1=4, k0=6,k1=8): 1x2x2 -> schoolbook (rank 4)
        
        # Total: 8*7 + 8*4 = 56 + 32 = 88
        
        rank = 88
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen terms for 2x2x2
        # The 7 Strassen products for computing C = A*B where A,B are 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # In terms of coefficients for U (A side), V (B side), W (C side):
        # Each term t has:
        #   U coefficients: which A entries are used (with signs)
        #   V coefficients: which B entries are used (with signs)  
        #   W coefficients: which C entries are affected (with signs)
        
        strassen_terms = [
            # (U_coeffs for 4 A entries, V_coeffs for 4 B entries, W_coeffs for 4 C entries)
            # A entries: A00, A01, A10, A11 -> indices 0,1,2,3
            # B entries: B00, B01, B10, B11 -> indices 0,1,2,3
            # C entries: C00, C01, C10, C11 -> indices 0,1,2,3
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),    # M2
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),   # M3
            ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, -1, 0]),  # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
            ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
            ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),   # M7
        ]
        
        col = 0
        # Process all blocks
        for i0 in [0, 2]:
            for j0 in [0, 2]:
                for k0 in [0, 2, 4, 6]:
                    i1 = i0 + 2 if i0 == 0 else 3
                    j1 = j0 + 2
                    k1 = k0 + 2
                    di = i1 - i0
                    dj = j1 - j0
                    dk = k1 - k0
                    
                    if di == 2 and dj == 2 and dk == 2:
                        # Use Strassen
                        for t in range(7):
                            u_c, v_c, w_c = strassen_terms[t]
                            # Fill U
                            for di_idx in range(2):
                                for dj_idx in range(2):
                                    idx = (i0 + di_idx) * m + (j0 + dj_idx)
                                    U[idx, col] = u_c[di_idx * 2 + dj_idx]
                            # Fill V
                            for dj_idx in range(2):
                                for dk_idx in range(2):
                                    idx = (j0 + dj_idx) * p + (k0 + dk_idx)
                                    V[idx, col] = v_c[dj_idx * 2 + dk_idx]
                            # Fill W
                            for di_idx in range(2):
                                for dk_idx in range(2):
                                    idx = (k0 + dk_idx) * n + (i0 + di_idx)
                                    W[idx, col] = w_c[di_idx * 2 + dk_idx]
                            col += 1
                    else:
                        # Schoolbook
                        for di_idx in range(di):
                            for dj_idx in range(dj):
                                for dk_idx in range(dk):
                                    u_idx = (i0 + di_idx) * m + (j0 + dj_idx)
                                    v_idx = (j0 + dj_idx) * p + (k0 + dk_idx)
                                    w_idx = (k0 + dk_idx) * n + (i0 + di_idx)
                                    U[u_idx, col] = 1
                                    V[v_idx, col] = 1
                                    W[w_idx, col] = 1
                                    col += 1
        
        return U, V, W, rank
    
    # Build initial Strassen-based decomposition
    U, V, W, rank = _build_strassen_decomposition(n, m, p)
    
    # Verify
    def _verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Now try to reduce rank by finding linear dependencies among the columns
    # If columns c1 and c2 are linearly dependent in a specific way, we might merge them
    # More practically: try to find if any rank-1 component can be eliminated
    
    # Alternative: use the fact that we can search for better decompositions
    # by trying different block partitions
    
    # Try different partition strategies
    def _try_partition(i_splits, j_splits, k_splits):
        """Try a specific partition and return the rank."""
        # i_splits, j_splits, k_splits are lists of (start, end)
        rank = 0
        for i0, i1 in i_splits:
            for j0, j1 in j_splits:
                for k0, k1 in k_splits:
                    di, dj, dk = i1-i0, j1-j0, k1-k0
                    if di == 2 and dj == 2 and dk == 2:
                        rank += 7
                    else:
                        rank += di * dj * dk
        return rank
    
    # Generate possible splits
    def _get_splits(N):
        """Get all ways to split [0,N) into segments of size 1 or 2."""
        if N == 0:
            yield []
            return
        if N == 1:
            yield [(0, 1)]
            return
        if N == 2:
            yield [(0, 2)]
            yield [(0, 1), (1, 2)]
            return
        # For N > 2, try first segment of size 1 or 2
        for rest in _get_splits(N - 1):
            yield [(0, 1)] + [(s+1, e+1) for s, e in rest]
        if N >= 2:
            for rest in _get_splits(N - 2):
                yield [(0, 2)] + [(s+2, e+2) for s, e in rest]
    
    # Try all combinations of splits
    i_split_options = list(_get_splits(n))
    j_split_options = list(_get_splits(m))
    k_split_options = list(_get_splits(p))
    
    for isegs in i_split_options:
        for jsegs in j_split_options:
            for ksegs in k_split_options:
                if time.time() > deadline - 20:
                    break
                
                r = _try_partition(isegs, jsegs, ksegs)
                if r < best_rank:
                    # Build this decomposition
                    U_new = np.zeros((n*m, r))
                    V_new = np.zeros((m*p, r))
                    W_new = np.zeros((p*n, r))
                    
                    strassen_terms = [
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
                        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
                        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, -1, 0]),
                        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
                        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
                    ]
                    
                    col = 0
                    for i0, i1 in isegs:
                        for j0, j1 in jsegs:
                            for k0, k1 in ksegs:
                                di, dj, dk = i1-i0, j1-j0, k1-k0
                                if di == 2 and dj == 2 and dk == 2:
                                    for t in range(7):
                                        u_c, v_c, w_c = strassen_terms[t]
                                        for di_idx in range(2):
                                            for dj_idx in range(2):
                                                idx = (i0 + di_idx) * m + (j0 + dj_idx)
                                                U_new[idx, col] = u_c[di_idx * 2 + dj_idx]
                                        for dj_idx in range(2):
                                            for dk_idx in range(2):
                                                idx = (j0 + dj_idx) * p + (k0 + dk_idx)
                                                V_new[idx, col] = v_c[dj_idx * 2 + dk_idx]
                                        for di_idx in range(2):
                                            for dk_idx in range(2):
                                                idx = (k0 + dk_idx) * n + (i0 + di_idx)
                                                W_new[idx, col] = w_c[di_idx * 2 + dk_idx]
                                        col += 1
                                else:
                                    for di_idx in range(di):
                                        for dj_idx in range(dj):
                                            for dk_idx in range(dk):
                                                u_idx = (i0 + di_idx) * m + (j0 + dj_idx)
                                                v_idx = (j0 + dj_idx) * p + (k0 + dk_idx)
                                                w_idx = (k0 + dk_idx) * n + (i0 + di_idx)
                                                U_new[u_idx, col] = 1
                                                V_new[v_idx, col] = 1
                                                W_new[w_idx, col] = 1
                                                col += 1
                    
                    if col == r and _verify(U_new, V_new, W_new, n, m, p):
                        best_rank = r
                        best_U, best_V, best_W = U_new, V_new, W_new
    
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
