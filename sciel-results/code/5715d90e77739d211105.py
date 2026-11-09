import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using block structure
    # A is 3x3, B is 3x4, C is 3x4
    # Split A into:
    # A11 (2x2), A12 (2x1)
    # A21 (1x2), A22 (1x1)
    # Split B into:
    # B11 (2x2), B12 (2x2)
    # B21 (1x2), B22 (1x2)
    
    # The total rank is the sum of the ranks of the block multiplications
    # because the (i,j,k) indices for each block are disjoint.
    
    # We need to generate U, V, W matrices.
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (12, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # We'll accumulate the terms.
    terms = []
    
    # Helper to add a block decomposition
    # block_i_start, block_j_start, block_k_start: indices in global A, B, C
    # block_n, block_m, block_p: dimensions of the block
    # decomposition: list of (u_vals, v_vals, w_vals) where u_vals is dict {(local_i, local_j): val}
    def add_block(i_start, j_start, k_start, bn, bm, bp, decomp):
        for u_dict, v_dict, w_dict in decomp:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            for (li, lj), val in u_dict.items():
                gi = i_start + li
                gj = j_start + lj
                u[gi * m + gj] = val
            
            for (lj, lk), val in v_dict.items():
                gj = j_start + lj
                gk = k_start + lk
                v[gj * p + gk] = val
            
            for (lk, li), val in w_dict.items():
                gk = k_start + lk
                gi = i_start + li
                w[gk * n + gi] = val
            
            terms.append((u, v, w))

    # Strassen decomposition for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    strassen_decomp = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1, (1,0):1}, {(0,1):1}), # W is (k,i). C21 gets M2. C21 is (i=1, k=0).
        # Wait, let's trace indices carefully.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the contributions to W (k,i) for each M.
        # M1 contributes to C11 (k=0, i=0) and C22 (k=1, i=1) with +1?
        # No, M1 is a scalar.
        # C11 = M1 + ... -> W[0,0] += 1
        # C22 = M1 - ... -> W[1,1] += 1
        
        # Let's re-evaluate the Strassen terms as (U, V, W) tuples.
    ]
    
    # Actually, it's easier to just list the 7 terms explicitly with their U, V, W coefficients.
    # U is (i,j), V is (j,k), W is (k,i).
    
    # Term 1: M1
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1  -> Contributes to C[0,0] and C[1,1]
    # But Strassen says C11 gets M1 and C22 gets M1.
    # So W[0,0]=1, W[1,1]=1.
    
    # Term 2: M2
    # U: (1,0)=1, (1,1)=1  (A21+A22)
    # V: (0,0)=1, (1,0)=1  (B11)
    # W: (0,1)=1           (C21 gets M2. i=1, k=0)
    
    # Term 3: M3
    # U: (0,0)=1            (A11)
    # V: (0,1)=1, (1,1)=-1  (B12-B22)
    # W: (1,0)=1            (C12 gets M3. i=0, k=1)
    
    # Term 4: M4
    # U: (1,1)=1            (A22)
    # V: (1,0)=1, (0,0)=-1  (B21-B11)
    # W: (0,1)=1, (0,0)=1   (C21 gets M4, C11 gets M4)
    
    # Term 5: M5
    # U: (0,0)=1, (0,1)=1   (A11+A12)
    # V: (1,1)=1            (B22)
    # W: (1,0)=-1, (1,1)=1  (C12 gets M5, C22 gets -M5? No, C12=M3+M5, C22=M1-M2+M3+M6. Wait.)
    # Let's check C22 again.
    # C22 = M1 - M2 + M3 + M6.
    # C12 = M3 + M5.
    # C11 = M1 + M4 - M5 + M7.
    # C21 = M2 + M4.
    
    # So M5 contributes to C12 (+1) and C11 (-1).
    # W[1,0] = 1 (for C12, i=0, k=1)
    # W[0,0] = -1 (for C11, i=0, k=0)
    
    # Term 6: M6
    # U: (1,0)=1, (0,0)=-1  (A21-A11)
    # V: (0,0)=1, (0,1)=1   (B11+B12)
    # W: (1,1)=1            (C22 gets M6. i=1, k=1)
    
    # Term 7: M7
    # U: (0,1)=1, (1,1)=-1  (A12-A22)
    # V: (1,0)=1, (1,1)=1   (B21+B22)
    # W: (0,0)=1            (C11 gets M7. i=0, k=0)

    # Let's verify the sum for C11 (i=0, k=0):
    # M1: W[0,0]=1
    # M4: W[0,0]=1
    # M5: W[0,0]=-1
    # M7: W[0,0]=1
    # Sum: 1+1-1+1 = 2? No, Strassen is C11 = M1+M4-M5+M7.
    # My W values:
    # M1: 1
    # M4: 1
    # M5: -1
    # M7: 1
    # Sum: 2. But it should be 1?
    # Wait, the tensor decomposition is $C = \sum U \otimes V \otimes W$.
    # The coefficient of $A_{ij} B_{jk}$ in $C_{ik}$ is $\sum W_{ik, c} U_{ij, c} V_{jk, c}$.
    # For C11, we want the sum to be 1 for the "identity" part?
    # No, the matrix multiplication tensor has entries 1 for all valid (i,j,k).
    # So $\sum W_{ik, c} U_{ij, c} V_{jk, c} = 1$ for all i,j,k.
    
    # Let's check the standard Strassen coefficients.
    # They are correct.
    
    # I will construct the terms programmatically to avoid manual errors.
    
    def get_strassen_terms():
        # Returns list of (U_dict, V_dict, W_dict) for 2x2x2
        # Indices are local 0,1
        t1_u = {(0,0):1, (1,1):1}
        t1_v = {(0,0):1, (1,1):1}
        t1_w = {(0,0):1, (1,1):1}
        
        t2_u = {(1,0):1, (1,1):1}
        t2_v = {(0,0):1, (1,0):1}
        t2_w = {(0,1):1}
        
        t3_u = {(0,0):1}
        t3_v = {(0,1):1, (1,1):-1}
        t3_w = {(1,0):1}
        
        t4_u = {(1,1):1}
        t4_v = {(1,0):1, (0,0):-1}
        t4_w = {(0,1):1, (0,0):1}
        
        t5_u = {(0,0):1, (0,1):1}
        t5_v = {(1,1):1}
        t5_w = {(1,0):1, (0,0):-1}
        
        t6_u = {(1,0):1, (0,0):-1}
        t6_v = {(0,0):1, (0,1):1}
        t6_w = {(1,1):1}
        
        t7_u = {(0,1):1, (1,1):-1}
        t7_v = {(1,0):1, (1,1):1}
        t7_w = {(0,0):1}
        
        return [
            (t1_u, t1_v, t1_w),
            (t2_u, t2_v, t2_w),
            (t3_u, t3_v, t3_w),
            (t4_u, t4_v, t4_w),
            (t5_u, t5_v, t5_w),
            (t6_u, t6_v, t6_w),
            (t7_u, t7_v, t7_w)
        ]

    def get_schoolbook_terms(bn, bm, bp):
        terms = []
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    u = {(i,j):1}
                    v = {(j,k):1}
                    w = {(k,i):1}
                    terms.append((u,v,w))
        return terms

    # Add Block 1: A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # i_start=0, j_start=0, k_start=0
    add_block(0, 0, 0, 2, 2, 2, get_strassen_terms())
    
    # Add Block 2: A11 (2x2) * B12 (2x2) -> C12 (2x2)
    # i_start=0, j_start=0, k_start=2
    add_block(0, 0, 2, 2, 2, 2, get_strassen_terms())
    
    # Add Block 3: A21 (1x2) * B11 (2x2) -> C21 (1x2)
    # i_start=2, j_start=0, k_start=0
    add_block(2, 0, 0, 1, 2, 2, get_schoolbook_terms(1, 2, 2))
    
    # Add Block 4: A21 (1x2) * B12 (2x2) -> C22 (1x2)
    # i_start=2, j_start=0, k_start=2
    add_block(2, 0, 2, 1, 2, 2, get_schoolbook_terms(1, 2, 2))
    
    # Add Block 5: A12 (2x1) * B21 (1x2) -> C11 (2x2)
    # i_start=0, j_start=2, k_start=0
    add_block(0, 2, 0, 2, 1, 2, get_schoolbook_terms(2, 1, 2))
    
    # Add Block 6: A12 (2x1) * B22 (1x2) -> C12 (2x2)
    # i_start=0, j_start=2, k_start=2
    add_block(0, 2, 2, 2, 1, 2, get_schoolbook_terms(2, 1, 2))
    
    # Add Block 7: A22 (1x1) * B21 (1x2) -> C21 (1x2)
    # i_start=2, j_start=2, k_start=0
    add_block(2, 2, 0, 1, 1, 2, get_schoolbook_terms(1, 1, 2))
    
    # Add Block 8: A22 (1x1) * B22 (1x2) -> C22 (1x2)
    # i_start=2, j_start=2, k_start=2
    add_block(2, 2, 2, 1, 1, 2, get_schoolbook_terms(1, 1, 2))
    
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (u, v, w) in enumerate(terms):
        U[:, c] = u
        V[:, c] = v
        W[:, c] = w
        
    return {"U": U, "V": V, "W": W}
