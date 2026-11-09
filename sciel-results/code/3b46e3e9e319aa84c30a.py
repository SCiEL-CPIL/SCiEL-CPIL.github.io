"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (6, rank)
    # V: (m*p, rank) -> (9, rank)
    # W: (p*n, rank) -> (6, rank)
    
    # We construct a rank 17 decomposition
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_idx: (i, j)
    # v_idx: (j, k)
    # w_idx: (k, i)
    # coeffs: (cu, cv, cw)
    def add_term(i, j, k, cu, cv, cw):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = cu
        V[v_idx][col] = cv
        W[w_idx][col] = cw
        col += 1

    # Block 1: Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Local A: i in {0,1}, j in {0,1}
    # Local B: j in {0,1}, k in {0,1}
    # Local C: i in {0,1}, k in {0,1}
    
    # M1 = (A[1,1]) (B[0,0] + B[1,1])
    # A[1,1] -> i=1, j=1
    # B[0,0] -> j=0, k=0; B[1,1] -> j=1, k=1
    # C[0,1] += M1; C[1,1] += M1
    # We need to represent M1 as sum of rank 1 terms?
    # No, M1 is a product of a sum.
    # M1 = A[1,1] * B[0,0] + A[1,1] * B[1,1]
    # This creates 2 terms.
    # Wait, Strassen's algorithm computes 7 products of sums.
    # Each product is a scalar.
    # The terms in the tensor decomposition are the rank-1 tensors corresponding to the monomials in the expansion?
    # No, the rank-1 tensors correspond to the "intermediate" products.
    # In the tensor decomposition, each rank-1 term is $u \otimes v \otimes w$.
    # For Strassen, the 7 terms are:
    # 1. (A11) * (B00+B11) -> This is NOT a rank 1 tensor in the original A,B,C space.
    # It is a linear combination.
    # To express this as a sum of rank-1 tensors, we must expand the sums.
    # (A11)(B00+B11) = A11 B00 + A11 B11.
    # So each Strassen "product" expands into 1 or 2 rank-1 terms depending on the number of terms in the factors.
    # M1: A[1,1] (1 term) * (B[0,0]+B[1,1]) (2 terms) -> 2 rank-1 terms.
    # M2: (A[1,0]+A[1,1]) (2 terms) * B[0,0] (1 term) -> 2 rank-1 terms.
    # M3: (A[0,0]-A[1,0]) (2 terms) * (B[1,0]+B[1,1]) (2 terms) -> 4 rank-1 terms.
    # M4: A[0,0] (1) * B[1,1] (1) -> 1 rank-1 term.
    # M5: (A[0,0]+A[1,1]) (2) * (B[0,0]-B[1,1]) (2) -> 4 rank-1 terms.
    # M6: (A[0,1]-A[0,0]) (2) * (B[0,0]+B[1,0]) (2) -> 4 rank-1 terms.
    # M7: (A[1,0]-A[0,1]) (2) * (B[1,1]+B[0,1]) (2) -> 4 rank-1 terms.
    # Total terms = 2+2+4+1+4+4+4 = 21?
    # This is just the schoolbook expansion!
    # Strassen reduces the number of *multiplications*, but if we expand them into rank-1 tensors over the base field, we get back to the schoolbook count if we don't use the structure to cancel or combine?
    # No, the rank of the tensor is the minimum number of rank-1 tensors.
    # Strassen's algorithm provides a decomposition of rank 7 for the $2 \times 2 \times 2$ tensor.
    # The 7 terms are NOT monomials. They are bilinear forms.
    # However, the tensor decomposition requires rank-1 tensors $u \otimes v \otimes w$.
    # The Strassen decomposition is:
    # $T = \sum_{s=1}^7 u_s \otimes v_s \otimes w_s$.
    # The $u_s, v_s, w_s$ are vectors with coefficients $\pm 1$.
    # We must use these vectors directly.
    
    # So, for Block 1, we add 7 terms.
    # We need to map the local Strassen vectors to the global indices.
    
    # Strassen Vectors for 2x2:
    # Indices: A: 00, 01, 10, 11. B: 00, 01, 10, 11. C: 00, 01, 10, 11.
    # M1: u=[0,0,0,1] (A11), v=[1,0,0,1] (B00+B11), w=[0,1,0,1] (C01+C11)
    # M2: u=[0,0,1,1] (A10+A11), v=[1,0,0,0] (B00), w=[0,0,1,0] (C10)
    # M3: u=[1,0,-1,0] (A00-A10), v=[0,0,1,1] (B10+B11), w=[0,0,1,-1] (C10-C11) -- Wait, C10-C11?
    # Let's check the standard Strassen formulas again.
    # C00 = M4 + M5 + M7 - M2 - M6
    # C01 = M1 + M6
    # C10 = M2 + M3
    # C11 = M1 - M3 + M4
    
    # Vectors (u, v, w) such that sum u_s v_s w_s^T = C.
    # Term 1: u=(0,0,0,1), v=(1,0,0,1), w=(0,1,0,1)
    #   u*v = [0,0,0,1] * [1,0,0,1] -> Only A11 and B00,B11 interact.
    #   w contributes to C01 and C11.
    #   Check: C01 += A11(B00+B11). C11 += A11(B00+B11).
    #   In C01 formula: M1 is present. In C11 formula: M1 is present. Correct.
    
    # Term 2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
    #   C10 += (A10+A11)B00.
    #   In C10 formula: M2 is present. Correct.
    
    # Term 3: u=(1,0,-1,0), v=(0,0,1,1), w=(0,0,1,-1)
    #   C10 += (A00-A10)(B10+B11). C11 -= (A00-A10)(B10+B11).
    #   In C10: M3 present. In C11: -M3 present. Correct.
    
    # Term 4: u=(1,0,0,0), v=(0,0,0,1), w=(1,0,0,1)
    #   C00 += A00 B11. C11 += A00 B11.
    #   In C00: M4 present. In C11: M4 present. Correct.
    
    # Term 5: u=(1,0,0,1), v=(1,0,0,-1), w=(1,0,0,0)
    #   C00 += (A00+A11)(B00-B11).
    #   In C00: M5 present. Correct.
    
    # Term 6: u=(0,1,-1,0), v=(1,0,1,0), w=(0,1,0,0)
    #   C01 += (A01-A00)(B00+B10).
    #   In C01: M6 present. Correct.
    
    # Term 7: u=(0,-1,1,0), v=(0,1,0,1), w=(1,0,0,0)
    #   C00 += (A10-A01)(B01+B11).
    #   In C00: M7 present. Correct.
    
    # We will add these 7 terms.
    
    # Mapping local (r,c) to global (i,j) for A
    # Local A 00 -> Global (0,0)
    # Local A 01 -> Global (0,1)
    # Local A 10 -> Global (1,0)
    # Local A 11 -> Global (1,1)
    
    # Mapping local (r,c) to global (j,k) for B
    # Local B 00 -> Global (0,0)
    # Local B 01 -> Global (0,1)
    # Local B 10 -> Global (1,0)
    # Local B 11 -> Global (1,1)
    
    # Mapping local (r,c) to global (k,i) for C
    # Local C 00 -> Global (0,0)
    # Local C 01 -> Global (1,0)
    # Local C 10 -> Global (0,1)
    # Local C 11 -> Global (1,1)
    # Note: W index is k*n + i.
    # Local C row r, col c.
    # Global i=r, k=c.
    # W index = c*n + r.
    
    # We define the 7 terms for Block 1
    terms_block1 = [
        # (u_local, v_local, w_local)
        # u: [00, 01, 10, 11]
        # v: [00, 01, 10, 11]
        # w: [00, 01, 10, 11]
        ((0,0,0,1), (1,0,0,1), (0,1,0,1)), # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,0)), # M2
        ((1,0,-1,0), (0,0,1,1), (0,0,1,-1)), # M3
        ((1,0,0,0), (0,0,0,1), (1,0,0,1)), # M4
        ((1,0,0,1), (1,0,0,-1), (1,0,0,0)), # M5
        ((0,1,-1,0), (1,0,1,0), (0,1,0,0)), # M6
        ((0,-1,1,0), (0,1,0,1), (1,0,0,0)), # M7
    ]
    
    for u_l, v_l, w_l in terms_block1:
        # Expand u_l, v_l, w_l into global terms?
        # No, u_l is a vector. It has multiple non-zeros.
        # A rank-1 term in the tensor is defined by SINGLE non-zero entries?
        # NO. A rank-1 tensor is $u \otimes v \otimes w$.
        # $u$ can have multiple non-zeros.
        # So we can add this term directly.
        
        # We need to add $u \otimes v \otimes w$ to the decomposition.
        # This means for every non-zero in u, v, w, we update the matrix?
        # No, the matrices U, V, W store the vectors.
        # U[:, col] = u_global
        # V[:, col] = v_global
        # W[:, col] = w_global
        
        # Construct u_global
        u_g = [0] * (n*m)
        # Map local index to global
        # Local 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
        mapping_u = [(0,0), (0,1), (1,0), (1,1)]
        for idx, val in enumerate(u_l):
            if val != 0:
                i, j = mapping_u[idx]
                u_g[i*m + j] = val
                
        v_g = [0] * (m*p)
        mapping_v = [(0,0), (0,1), (1,0), (1,1)]
        for idx, val in enumerate(v_l):
            if val != 0:
                j, k = mapping_v[idx]
                v_g[j*p + k] = val
                
        w_g = [0] * (p*n)
        mapping_w = [(0,0), (0,1), (1,0), (1,1)] # Local C (r,c) -> Global (i=r, k=c)
        for idx, val in enumerate(w_l):
            if val != 0:
                r, c = mapping_w[idx]
                i, k = r, c
                w_g[k*n + i] = val
                
        U[:, col] = u_g # Pseudo-code, need to assign
        # In Python:
        for r in range(n*m): U[r][col] = u_g[r]
        for r in range(m*p): V[r][col] = v_g[r]
        for r in range(p*n): W[r][col] = w_g[r]
        col += 1
        
    # Block 2: A_L (cols 0,1) * B_T2 (rows 0,1, col 2) -> C (col 2)
    # This is 2x2 times 2x1.
    # Rank 4.
    # Terms:
    # 1. A[0,0] B[0,2] -> C[0,2]
    # 2. A[0,1] B[1,2] -> C[0,2]
    # 3. A[1,0] B[0,2] -> C[1,2]
    # 4. A[1,1] B[1,2] -> C[1,2]
    
    # Term 1
    add_term(0, 0, 2, 1, 1, 1)
    # Term 2
    add_term(0, 1, 2, 1, 1, 1)
    # Term 3
    add_term(1, 0, 2, 1, 1, 1)
    # Term 4
    add_term(1, 1, 2, 1, 1, 1)
    
    # Block 3: A_R (col 2) * B_B (row 2) -> C (all cols)
    # 2x1 times 1x3.
    # Rank 6.
    # Terms:
    # 1. A[0,2] B[2,0] -> C[0,0]
    # 2. A[0,2] B[2,1] -> C[0,1]
    # 3. A[0,2] B[2,2] -> C[0,2]
    # 4. A[1,2] B[2,0] -> C[1,0]
    # 5. A[1,2] B[2,1] -> C[1,1]
    # 6. A[1,2] B[2,2] -> C[1,2]
    
    add_term(0, 2, 0, 1, 1, 1)
    add_term(0, 2, 1, 1, 1, 1)
    add_term(0, 2, 2, 1, 1, 1)
    add_term(1, 2, 0, 1, 1, 1)
    add_term(1, 2, 1, 1, 1, 1)
    add_term(1, 2, 2, 1, 1, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
