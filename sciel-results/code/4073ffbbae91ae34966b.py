import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split 5 into 2, 2, 1
    splits = [2, 2, 1]
    offsets = [0, 2, 4]
    
    # Total rank estimate: 117
    # We will dynamically calculate the rank and build the matrices
    # U: (n*m, rank) -> (25, rank)
    # V: (m*p, rank) -> (25, rank)
    # W: (p*n, rank) -> (25, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        # u_idx, v_idx, w_idx are global indices
        # u_val, v_val, w_val are coefficients
        # We append to the lists, which will be transposed later
        # Or we can use a dictionary to accumulate if we want to merge terms?
        # No, the rank is the number of terms. We just append.
        U.append((u_idx, u_val))
        V.append((v_idx, v_val))
        W.append((w_idx, w_val))
        
    # Helper to get Strassen terms for 2x2x2
    # Returns list of (u_local, v_local, w_local, u_coeff, v_coeff, w_coeff)
    # u_local in 0..3, v_local in 0..3, w_local in 0..3
    # w_local corresponds to C index (k*n+i) -> (k*2+i)
    def strassen_2x2_terms():
        terms = []
        # M1 = (A11+A22)(B11+B22)
        # A11: (0,0), A22: (1,1)
        # B11: (0,0), B22: (1,1)
        # C11: (0,0) -> w=0, C12: (0,1) -> w=1, C21: (1,0) -> w=2, C22: (1,1) -> w=3
        # Wait, w index is k*n+i.
        # For 2x2, n=2.
        # C00: k=0, i=0 -> w=0
        # C01: k=0, i=1 -> w=1
        # C10: k=1, i=0 -> w=2
        # C11: k=1, i=1 -> w=3
        
        # M1
        terms.append(((0,0), (0,0), (0,0), 1, 1, 1)) # A11 B11 -> C11
        terms.append(((1,1), (1,1), (0,0), 1, 1, 1)) # A22 B22 -> C11
        terms.append(((0,0), (1,1), (0,0), 1, 1, 1)) # A11 B22 -> C11
        terms.append(((1,1), (0,0), (0,0), 1, 1, 1)) # A22 B11 -> C11
        
        # M2 = (A21+A22)B11
        terms.append(((1,0), (0,0), (1,0), 1, 1, 1)) # A21 B11 -> C21
        terms.append(((1,1), (0,0), (1,0), 1, 1, 1)) # A22 B11 -> C21
        
        # M3 = A11(B12-B22)
        terms.append(((0,0), (0,1), (0,1), 1, 1, 1)) # A11 B12 -> C12
        terms.append(((0,0), (1,1), (0,1), 1, -1, 1)) # A11 B22 -> C12
        
        # M4 = A22(B21-B11)
        terms.append(((1,1), (1,0), (1,1), 1, 1, 1)) # A22 B21 -> C22
        terms.append(((1,1), (0,0), (1,1), 1, -1, 1)) # A22 B11 -> C22
        
        # M5 = (A11+A12)B22
        terms.append(((0,0), (1,1), (0,1), 1, 1, 1)) # A11 B22 -> C12
        terms.append(((0,1), (1,1), (0,1), 1, 1, 1)) # A12 B22 -> C12
        # Wait, M5 contributes to C12?
        # C12 = M3 + M5.
        # M3 = A11(B12-B22).
        # M5 = (A11+A12)B22.
        # C12 = A11B12 - A11B22 + A11B22 + A12B22 = A11B12 + A12B22. Correct.
        
        # M6 = (A21-A11)(B11+B12)
        terms.append(((1,0), (0,0), (1,1), 1, 1, 1)) # A21 B11 -> C22
        terms.append(((1,0), (0,1), (1,1), 1, 1, 1)) # A21 B12 -> C22
        terms.append(((0,0), (0,0), (1,1), -1, 1, 1)) # A11 B11 -> C22
        terms.append(((0,0), (0,1), (1,1), -1, 1, 1)) # A11 B12 -> C22
        
        # M7 = (A12-A22)(B21+B22)
        terms.append(((0,1), (1,0), (0,0), 1, 1, 1)) # A12 B21 -> C11
        terms.append(((0,1), (1,1), (0,0), 1, 1, 1)) # A12 B22 -> C11
        terms.append(((1,1), (1,0), (0,0), -1, 1, 1)) # A22 B21 -> C11
        terms.append(((1,1), (1,1), (0,0), -1, 1, 1)) # A22 B22 -> C11
        
        return terms

    # Helper for general small blocks
    # We will just use the standard decomposition for non-2x2x2 blocks
    # Rank = n*m*p
    def standard_terms(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # u_idx = i*m+j
                    # v_idx = j*p+k
                    # w_idx = k*n+i
                    terms.append(((i,j), (j,k), (k,i), 1, 1, 1))
        return terms

    # Iterate over blocks I, J, K
    for I in range(3):
        for J in range(3):
            for K in range(3):
                n_I = splits[I]
                m_J = splits[J]
                p_K = splits[K]
                
                # Get local terms
                if n_I == 2 and m_J == 2 and p_K == 2:
                    local_terms = strassen_2x2_terms()
                else:
                    local_terms = standard_terms(n_I, m_J, p_K)
                    
                for term in local_terms:
                    (li, lj), (lj2, lk), (lk2, li2), u_c, v_c, w_c = term
                    # Verify indices match
                    # lj should be lj2, lk should be lk2, li should be li2
                    # They should by construction
                    
                    # Global indices
                    # U index: (I, J) block, local (li, lj)
                    # Global i = offsets[I] + li
                    # Global j = offsets[J] + lj
                    # u_idx = global_i * 5 + global_j
                    gi = offsets[I] + li
                    gj = offsets[J] + lj
                    u_idx = gi * 5 + gj
                    
                    # V index: (J, K) block, local (lj2, lk)
                    # Global j = offsets[J] + lj2
                    # Global k = offsets[K] + lk
                    # v_idx = global_j * 5 + global_k
                    gj2 = offsets[J] + lj2
                    gk = offsets[K] + lk
                    v_idx = gj2 * 5 + gk
                    
                    # W index: (K, I) block, local (lk2, li2)
                    # Global k = offsets[K] + lk2
                    # Global i = offsets[I] + li2
                    # w_idx = global_k * 5 + global_i
                    gk2 = offsets[K] + lk2
                    gi2 = offsets[I] + li2
                    w_idx = gk2 * 5 + gi2
                    
                    add_term(u_idx, v_idx, w_idx, u_c, v_c, w_c)
                    
    # Convert lists of tuples to matrices
    rank = len(U)
    U_mat = [[0] * rank for _ in range(25)]
    V_mat = [[0] * rank for _ in range(25)]
    W_mat = [[0] * rank for _ in range(25)]
    
    for r in range(rank):
        u_idx, u_val = U[r]
        v_idx, v_val = V[r]
        w_idx, w_val = W[r]
        U_mat[u_idx][r] = u_val
        V_mat[v_idx][r] = v_val
        W_mat[w_idx][r] = w_val
        
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
