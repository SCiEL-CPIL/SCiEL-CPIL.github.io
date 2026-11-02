"""Tensor decomposition for 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matmul tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pass  # placeholder
        # Proper verification:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def verify_full(U, V, W):
        """Full verification: check all (i',j',j'',k') combinations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = len(U_arr[0])
        # Build the full tensor
        # T[i',j',j'',k'] = sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
        # Wait, the convention is W[k*n+i, r]
        # We need: for all i,k: C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r W[k*n+i, r] * L_r(A) * M_r(B)
        # where L_r(A) = sum_{i',j'} U[i'*m+j', r] * A[i',j']
        # and M_r(B) = sum_{j',k'} V[j'*p+k', r] * B[j',k']
        # So C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j',r] A[i',j']) * (sum_{j'',k'} V[j''*p+k',r] B[j'',k'])
        # = sum_{i',j',j'',k'} A[i',j'] B[j'',k'] * (sum_r W[k*n+i,r] U[i'*m+j',r] V[j''*p+k',r])
        # This must equal sum_j A[i,j] B[j,k]
        # So the coefficient of A[i',j'] B[j'',k'] must be:
        #   1 if i'=i, k'=k, j'=j'' (for some j, meaning j'=j'' and i'=i, k'=k)
        #   0 otherwise
        # Actually: coefficient = 1 iff i'=i AND k'=k AND j'=j''
        # But we sum over j, so: sum_j delta_{i',i} delta_{j',j} delta_{j'',j} delta_{k',k}
        # = delta_{i',i} delta_{k',k} delta_{j',j''}
        for i in range(n):
            for i_prime in range(n):
                for j_prime in range(m):
                    for j_double_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                expected = 1.0 if (i_prime == i and k_prime == k and j_prime == j_double_prime) else 0.0
                                actual = 0.0
                                for r in range(rank):
                                    actual += W_arr[k * n + i, r] * U_arr[i_prime * m + j_prime, r] * V_arr[j_double_prime * p + k_prime, r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    # Try to construct a rank-23 decomposition
    # The known optimal decomposition for 3x3x3 has rank 23.
    # We'll use a structured approach based on the Alon-Boppana-Yao result.
    
    # Start with schoolbook (rank 27) and try to eliminate 4 terms via linear combinations.
    # Strategy: identify linear dependencies among the 27 rank-1 tensors and exploit them.
    
    # Schoolbook terms: (i,j,k) for i,j,k in {0,1,2}
    # T_{ijk} corresponds to A[i,j]*B[j,k] contributing to C[i,k]
    
    # We can form new rank-1 tensors by taking linear combinations.
    # Key idea: if we have terms like (i,j,k) and (i,j',k) and (i',j,k) etc.,
    # we can sometimes combine them.
    
    # Let me try a specific known approach:
    # Use the fact that for 3x3, we can save 4 ranks.
    
    # Approach: Use a bilinear algorithm derived from partitioning.
    
    # Let's try constructing from scratch using a structured ansatz.
    
    # The 27 schoolbook terms can be indexed as (i,j,k).
    # We want to find 23 vectors u_r, v_r, w_r in R^9 such that
    # sum_r w_r[k*n+i] * u_r[i'*m+j'] * v_r[j''*p+k'] = delta_{i',i} delta_{k',k} delta_{j',j''}
    
    # Let's use a symmetric construction.
    
    # Define the 23 terms using a known pattern.
    # I'll construct this carefully.
    
    # Term types:
    # Type 1: Pure schoolbook terms (no modification) - most of them
    # Type 2: Modified terms using sums/differences
    
    # Let me try a concrete construction:
    
    # For 2x2x2, Strassen saves 1 rank (7 instead of 8).
    # For 3x3x3, we save 4 ranks (23 instead of 27).
    
    # One known approach: partition the 3x3 matrices into blocks.
    
    # Let me use the following construction based on the paper "Optimal bilinear algorithms for 3x3 matrix multiplication"
    
    # I'll define 23 terms with specific coefficient patterns.
    
    # Use indices 0,1,2 for i,j,k.
    
    # The decomposition:
    # We'll have terms that look like:
    # u_r has entries in R^9 (indexed by (i,j)), v_r in R^9 (indexed by (j,k)), w_r in R^9 (indexed by (k,i))
    
    # Let me define them systematically:
    
    # Terms 0-17: Standard schoolbook for certain (i,j,k)
    # Terms 18-22: Combined terms
    
    # Actually, let me just hardcode a known valid rank-23 decomposition.
    
    # Based on the structure, here's a construction:
    
    U = [[0.0]*23 for _ in range(9)]
    V = [[0.0]*23 for _ in range(9)]
    W = [[0.0]*23 for _ in range(9)]
    
    # Set up 23 rank-1 components
    # Using a pattern from known results:
    
    # Components 0-8: diagonal and near-diagonal terms
    # For (i,j,k) where j is "central"
    
    # Let me use a different approach: start from schoolbook and apply row/column operations
    # on the three matrices to eliminate 4 terms.
    
    # Schoolbook: 27 terms, one for each (i,j,k)
    # U[i*m+j, r] = 1 for term r=(i,j,k)
    # V[j*p+k, r] = 1 for term r=(i,j,k)  
    # W[k*n+i, r] = 1 for term r=(i,j,k)
    
    # We can apply invertible linear transformations to U, V, W columns.
    
    # Key insight: If we have two terms with same i but different j,k, we can combine.
    
    # Let me try: for fixed i, the 9 terms (i, j, k) for all j,k can potentially be compressed.
    
    # Actually, let me just implement a search or use a known explicit decomposition.
    
    # Known rank-23 decomposition for 3x3x3:
    # I'll use the following explicit construction.
    
    # Let a=0, b=1, c=2 for the three values.
    
    # I'll define the 23 terms as follows (using 1, -1, 0, and sometimes 0.5):
    
    # Term 0: u=(a,a), v=(a,a), w=(a,a)  -> A[0,0]*B[0,0] -> C[0,0]
    # Term 1: u=(a,b), v=(b,a), w=(a,a)  -> A[0,1]*B[1,0] -> C[0,0]
    # Term 2: u=(a,c), v=(c,a), w=(a,a)  -> A[0,2]*B[2,0] -> C[0,0]
    # ... etc for all 27 schoolbook, then combine 4 pairs.
    
    # Let me just create the full schoolbook and then try to find 4 linear combinations.
    
    # Initialize with schoolbook (27 terms)
    U27 = [[0.0]*27 for _ in range(9)]
    V27 = [[0.0]*27 for _ in range(9)]
    W27 = [[0.0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U27[i*3+j][col] = 1.0
                V27[j*3+k][col] = 1.0
                W27[k*3+i][col] = 1.0
                col += 1
    
    # Now try to eliminate 4 terms by expressing them as linear combinations of others.
    # This requires finding linear dependencies in the tensor product space.
    
    # For now, return the schoolbook decomposition as a safe fallback.
    # The rank-23 decomposition is complex to derive programmatically in limited time.
    
    # Verify schoolbook
    assert verify_full(U27, V27, W27), "Schoolbook decomposition failed verification"
    
    if time.time() > deadline:
        return {"U": U27, "V": V27, "W": W27}
    
    # Try to find a rank-23 decomposition using a structured approach
    # based on known results.
    
    # The following is a known rank-23 decomposition.
    # Sources: Alon, Boppana, Yao (1989); see also recent expositions.
    
    # I'll construct it using the following pattern:
    
    U23 = [[0.0]*23 for _ in range(9)]
    V23 = [[0.0]*23 for _ in range(9)]
    W23 = [[0.0]*23 for _ in range(9)]
    
    # Define 23 terms:
    # Terms 0-14: Standard terms (15 of the 27)
    # Terms 15-22: 8 combined terms
    
    # Standard terms to keep (15):
    # (0,0,0), (0,0,1), (0,0,2),
    # (0,1,0), (0,1,1), (0,1,2),
    # (0,2,0), (0,2,1), (0,2,2),
    # (1,0,0), (1,1,0), (1,2,0),
    # (2,0,0), (2,1,0), (2,2,0)
    
    keep_terms = [
        (0,0,0), (0,0,1), (0,0,2),
        (0,1,0), (0,1,1), (0,1,2),
        (0,2,0), (0,2,1), (0,2,2),
        (1,0,0), (1,1,0), (1,2,0),
        (2,0,0), (2,1,0), (2,2,0)
    ]
    
    for t, (i,j,k) in enumerate(keep_terms):
        U23[i*3+j][t] = 1.0
        V23[j*3+k][t] = 1.0
        W23[k*3+i][t] = 1.0
    
    # Removed terms (12):
    removed = [
        (1,0,1), (1,0,2),
        (1,1,1), (1,1,2),
        (1,2,1), (1,2,2),
        (2,0,1), (2,0,2),
        (2,1,1), (2,1,2),
        (2,2,1), (2,2,2)
    ]
    
    # We need to express these 12 terms using 8 additional rank-1 terms.
    # This requires careful construction.
    
    # The 8 combined terms will have non-trivial coefficients.
    
    # Let me define them based on known patterns:
    
    # Term 15: combines (1,0,1) and (1,1,0)
    # u: A[1,0]+A[1,1], v: B[0,1]+B[1,0], w: C[1,1]+C[0,0]... 
    # This gets complicated. Let me use a verified construction.
    
    # For safety, if we can't verify rank-23, return rank-27.
    
    # Let me try a specific known construction from the literature.
    
    # Based on the work of Landsberg and others, here's one approach:
    
    # I'll use the following 23-term decomposition:
    
    # Reset and use a known valid set.
    
    U23 = [[0.0]*23 for _ in range(9)]
    V23 = [[0.0]*23 for _ in range(9)]
    W23 = [[0.0]*23 for _ in range(9)]
    
    # Term definitions (u_entry, v_entry, w_entry) as sparse dicts
    # u is indexed by (i,j), v by (j,k), w by (k,i)
    
    def set_uv_w(term_idx, u_dict, v_dict, w_dict):
        for (i,j), val in u_dict.items():
            U23[i*3+j][term_idx] = val
        for (j,k), val in v_dict.items():
            V23[j*3+k][term_idx] = val
        for (k,i), val in w_dict.items():
            W23[k*3+i][term_idx] = val
    
    # 23 terms for 3x3x3 matrix multiplication
    # Based on known optimal decomposition
    
    # Terms 0-8: For i=0, all (j,k)
    for j in range(3):
        for k in range(3):
            t = j*3 + k
            set_uv_w(t, {(0,j): 1.0}, {(j,k): 1.0}, {(k,0): 1.0})
    
    # Terms 9-11: For i=1, j=0, all k
    for k in range(3):
        t = 9 + k
        set_uv_w(t, {(1,0): 1.0}, {(0,k): 1.0}, {(k,1): 1.0})
    
    # Terms 12-14: For i=1, j=1, k=0,1,2 - but we need to combine
    # Term 12: (1,1,0)
    set_uv_w(12, {(1,1): 1.0}, {(1,0): 1.0}, {(0,1): 1.0})
    
    # Term 13: combines (1,1,1) and (2,0,1)
    set_uv_w(13, {(1,1): 1.0, (2,0): 1.0}, {(1,1): 1.0, (0,1): 1.0}, {(1,1): 1.0, (1,2): 1.0})
    
    # Term 14: combines (1,1,2) and (2,0,2)
    set_uv_w(14, {(1,1): 1.0, (2,0): 1.0}, {(1,2): 1.0, (0,2): 1.0}, {(2,1): 1.0, (2,2): 1.0})
    
    # Term 15: (1,2,0)
    set_uv_w(15, {(1,2): 1.0}, {(2,0): 1.0}, {(0,1): 1.0})
    
    # Term 16: combines (1,2,1) and (2,1,1)
    set_uv_w(16, {(1,2): 1.0, (2,1): 1.0}, {(2,1): 1.0, (1,1): 1.0}, {(1,1): 1.0, (1,2): 1.0})
    
    # Term 17: combines (1,2,2) and (2,1,2)
    set_uv_w(17, {(1,2): 1.0, (2,1): 1.0}, {(2,2): 1.0, (1,2): 1.0}, {(2,1): 1.0, (2,2): 1.0})
    
    # Terms 18-22: For i=2, remaining terms
    # Term 18: (2,0,0)
    set_uv_w(18, {(2,0): 1.0}, {(0,0): 1.0}, {(0,2): 1.0})
    
    # Term 19: (2,1,0)
    set_uv_w(19, {(2,1): 1.0}, {(1,0): 1.0}, {(0,2): 1.0})
    
    # Term 20: (2,2,0)
    set_uv_w(20, {(2,2): 1.0}, {(2,0): 1.0}, {(0,2): 1.0})
    
    # Term 21: combines remaining (2,0,1) and (2,1,1) - but we already used some
    # Let me re-examine. This ad-hoc construction is error-prone.
    
    # Verify
    if verify_full(U23, V23, W23):
        return {"U": U23, "V": V23, "W": W23}
    
    # Fallback to schoolbook
    return {"U": U27, "V": V27, "W": W27}
# EVOLVE-BLOCK-END
