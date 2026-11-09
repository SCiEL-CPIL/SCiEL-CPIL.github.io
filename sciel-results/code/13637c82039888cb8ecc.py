import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition recursively.
    # State: (n, m, p, i_offset, j_offset, k_offset, U, V, W)
    # But it's easier to just build the full matrices directly by appending rows/cols.
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We'll use a list of tuples (u_row, v_row, w_row, coeff)
    # Actually, U, V, W are matrices.
    # Let's accumulate columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # u_idx is index in U (0 to n*m-1)
        # v_idx is index in V (0 to m*p-1)
        # w_idx is index in W (0 to p*n-1)
        U_cols.append((u_idx, coeff))
        V_cols.append((v_idx, coeff))
        W_cols.append((w_idx, coeff))
        
    # Helper to add a rank-1 term
    def add_rank1(i, j, k, coeff=1):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        add_term(u_idx, v_idx, w_idx, coeff)

    # Strassen decomposition for 2x2x2
    # Local indices i in 0..1, j in 0..1, k in 0..1
    # M1 = A11 * (B21 + B22)
    # M2 = (A11 + A21) * B11
    # M3 = (A12 + A22) * B22
    # M4 = A22 * (B12 - B11)
    # M5 = (A11 + A12) * (B22 - B21)
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A22 - A21) * (B21 + B22)
    
    def add_strassen(i_base, j_base, k_base):
        # i_base, j_base, k_base are offsets for the 2x2x2 block
        # i in {i_base, i_base+1}
        # j in {j_base, j_base+1}
        # k in {k_base, k_base+1}
        
        # We need to map local (li, lj, lk) to global (i, j, k)
        # i = i_base + li, etc.
        
        # M1: A11 * (B21 + B22)
        # This corresponds to sum of two rank-1 terms? No, Strassen combines them.
        # The tensor decomposition of Strassen is 7 rank-1 terms.
        # Term 1: A11 * B21 + A11 * B22
        # Wait, the "multiplication" is the rank-1 tensor.
        # In Strassen, the "multiplications" are:
        # M1 = A11 * (B21 + B22) -> This is not a rank-1 term in the original tensor space.
        # The rank-1 terms are the "multiplications" of the linear combinations.
        
        # Let's list the 7 rank-1 terms of Strassen:
        # 1. A11 * (B21 + B22) is NOT a term. The term is A11 * B21 + A11 * B22?
        # No. The algorithm computes M1 = A11 * (B21 + B22).
        # M1 is a 2x2 matrix? No, A11 is 1x1, B21 is 1x1. M1 is 1x1.
        # So M1 = A11 * B21 + A11 * B22.
        # This is a sum of two rank-1 terms in the scalar multiplication tensor.
        # But Strassen reduces the number of scalar multiplications.
        
        # The 7 multiplications are:
        # M1 = A11 * (B21 + B22)
        # M2 = (A11 + A21) * B11
        # M3 = (A12 + A22) * B22
        # M4 = A22 * (B12 - B11)
        # M5 = (A11 + A12) * (B22 - B21)
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A22 - A21) * (B21 + B22)
        
        # Each Mi is a product of two linear combinations.
        # In the tensor decomposition, each Mi contributes a rank-1 term?
        # No. The tensor is $\sum M_i \otimes U_i \otimes V_i$?
        # Actually, the decomposition is:
        # $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$
        # where $u_r, v_r, w_r$ are vectors.
        
        # For Strassen, the 7 terms are:
        # 1. $A_{11} (B_{21} + B_{22})$ -> $u = (1,0,0,0)$? No.
        
        # Let's use the standard Strassen decomposition vectors.
        # Indices for A: 11, 12, 21, 22 (local 0,1,2,3)
        # Indices for B: 11, 12, 21, 22
        # Indices for C: 11, 12, 21, 22
        
        # Term 1: M1 = A11 * (B21 + B22)
        # U vector: A11 is 1. So U = [1, 0, 0, 0]
        # V vector: B21 + B22 is 1, 1. So V = [0, 0, 1, 1] (indices 11,12,21,22 -> 0,1,2,3)
        # Wait, B indices: 11(0), 12(1), 21(2), 22(3).
        # B21 is index 2. B22 is index 3.
        # So V = [0, 0, 1, 1].
        # W vector: C11 = M1 - M4 - M6 + M7?
        # C11 = M1 - M4 - M6 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M3 - M5 + M7
        
        # So for M1:
        # W = [1, 0, 0, 0] (coefficient 1 for C11)
        
        # Let's list all 7:
        # M1: U=[1,0,0,0], V=[0,0,1,1], W=[1,0,0,0]
        # M2: U=[1,0,1,0], V=[1,0,0,0], W=[0,0,1,0]
        # M3: U=[0,1,0,1], V=[0,0,0,1], W=[0,1,0,1]
        # M4: U=[0,0,0,1], V=[0,1,-1,0], W=[-1,0,0,0]
        # M5: U=[1,1,0,0], V=[0,0,-1,1], W=[0,1,0,-1]
        # M6: U=[-1,0,1,0], V=[1,1,0,0], W=[-1,0,0,0]
        # M7: U=[0,0,-1,1], V=[0,0,1,1], W=[0,0,0,1]
        
        # We need to map local indices to global.
        # Local A idx 0->i_base, 1->i_base, 2->i_base+1, 3->i_base+1?
        # No, local A 11->0, 12->1, 21->2, 22->3.
        # Global i = i_base + local_i_row, j = j_base + local_j_col.
        # Local 11: i=0, j=0. Global i=i_base, j=j_base.
        # Local 12: i=0, j=1. Global i=i_base, j=j_base+1.
        # Local 21: i=1, j=0. Global i=i_base+1, j=j_base.
        # Local 22: i=1, j=1. Global i=i_base+1, j=j_base+1.
        
        # U vector entries correspond to A indices.
        # U[0] is A11. U[1] is A12. U[2] is A21. U[3] is A22.
        
        # For each term (u_vec, v_vec, w_vec):
        # We add rank-1 terms for each non-zero entry?
        # No, the term is a single rank-1 tensor in the decomposed space.
        # But U, V, W in the output are matrices of shape (n*m, rank).
        # Each column in U, V, W corresponds to one rank-1 term.
        # So for each Strassen term, we create one column in U, V, W.
        
        # U_col: length n*m. Non-zero at positions corresponding to A entries.
        # V_col: length m*p. Non-zero at positions corresponding to B entries.
        # W_col: length p*n. Non-zero at positions corresponding to C entries.
        
        terms = [
            ([1,0,0,0], [0,0,1,1], [1,0,0,0]),
            ([1,0,1,0], [1,0,0,0], [0,0,1,0]),
            ([0,1,0,1], [0,0,0,1], [0,1,0,1]),
            ([0,0,0,1], [0,1,-1,0], [-1,0,0,0]),
            ([1,1,0,0], [0,0,-1,1], [0,1,0,-1]),
            ([-1,0,1,0], [1,1,0,0], [-1,0,0,0]),
            ([0,0,-1,1], [0,0,1,1], [0,0,0,1])
        ]
        
        for u_vec, v_vec, w_vec in terms:
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            # Map local A indices to global
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            local_to_global_A = [
                (i_base + 0, j_base + 0),
                (i_base + 0, j_base + 1),
                (i_base + 1, j_base + 0),
                (i_base + 1, j_base + 1)
            ]
            for l_idx, val in enumerate(u_vec):
                if val != 0:
                    gi, gj = local_to_global_A[l_idx]
                    U_col[gi * m + gj] = val
            
            # Map local B indices to global
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            local_to_global_B = [
                (j_base + 0, k_base + 0),
                (j_base + 0, k_base + 1),
                (j_base + 1, k_base + 0),
                (j_base + 1, k_base + 1)
            ]
            for l_idx, val in enumerate(v_vec):
                if val != 0:
                    gj, gk = local_to_global_B[l_idx]
                    V_col[gj * p + gk] = val
            
            # Map local C indices to global
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            local_to_global_C = [
                (k_base + 0, i_base + 0),
                (k_base + 0, i_base + 1),
                (k_base + 1, i_base + 0),
                (k_base + 1, i_base + 1)
            ]
            for l_idx, val in enumerate(w_vec):
                if val != 0:
                    gk, gi = local_to_global_C[l_idx]
                    W_col[gk * n + gi] = val
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Schoolbook for 1x2x2
    # A (1x2), B (2x2)
    def add_schoolbook_122(i_base, j_base, k_base):
        # i in {i_base}, j in {j_base, j_base+1}, k in {k_base, k_base+1}
        for lj in range(2):
            for lk in range(2):
                gi = i_base
                gj = j_base + lj
                gk = k_base + lk
                add_rank1(gi, gj, gk, 1)

    # Schoolbook for 3x2x1
    # A (3x2), B (2x1)
    def add_schoolbook_321(i_base, j_base, k_base):
        # i in {i_base, i_base+1, i_base+2}, j in {j_base, j_base+1}, k in {k_base}
        for li in range(3):
            for lj in range(2):
                gi = i_base + li
                gj = j_base + lj
                gk = k_base
                add_rank1(gi, gj, gk, 1)

    # Build the full decomposition
    # m=4 split into 2+2
    # For each half (j_offset 0, 2):
    #   p=7 split into 2,2,2,1
    #   For each p_block:
    #     if size 2:
    #       n=3 split into 2+1
    #       add_strassen(i_base=0, j_base=j_offset, k_base=k_offset)
    #       add_schoolbook_122(i_base=2, j_base=j_offset, k_base=k_offset)
    #     if size 1:
    #       add_schoolbook_321(i_base=0, j_base=j_offset, k_base=k_offset)

    # j_offset 0
    # k_offset 0 (size 2)
    add_strassen(0, 0, 0)
    add_schoolbook_122(2, 0, 0)
    # k_offset 2 (size 2)
    add_strassen(0, 0, 2)
    add_schoolbook_122(2, 0, 2)
    # k_offset 4 (size 2)
    add_strassen(0, 0, 4)
    add_schoolbook_122(2, 0, 4)
    # k_offset 6 (size 1)
    add_schoolbook_321(0, 0, 6)
    
    # j_offset 2
    # k_offset 0 (size 2)
    add_strassen(0, 2, 0)
    add_schoolbook_122(2, 2, 0)
    # k_offset 2 (size 2)
    add_strassen(0, 2, 2)
    add_schoolbook_122(2, 2, 2)
    # k_offset 4 (size 2)
    add_strassen(0, 2, 4)
    add_schoolbook_122(2, 2, 4)
    # k_offset 6 (size 1)
    add_schoolbook_321(0, 2, 6)
    
    # Convert lists to matrices
    rank = len(U_cols)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for idx, val in enumerate(U_cols[r]):
            if val != 0:
                U[idx, r] = val
        for idx, val in enumerate(V_cols[r]):
            if val != 0:
                V[idx, r] = val
        for idx, val in enumerate(W_cols[r]):
            if val != 0:
                W[idx, r] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
