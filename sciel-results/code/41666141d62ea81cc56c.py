import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: 2+3 split
    # A: 5x5, B: 5x5, C: 5x5
    # A00 (2x2), A01 (2x3), A10 (3x2), A11 (3x3)
    # B00 (2x2), B01 (2x3), B10 (3x2), B11 (3x3)
    # C00 (2x2), C01 (2x3), C10 (3x2), C11 (3x3)
    
    # Ranks:
    # A00*B00 -> C00: Strassen 2x2x2 (Rank 7)
    # A01*B10 -> C00: Schoolbook 2x3x2 (Rank 12)
    # A00*B01 -> C01: Schoolbook 2x2x3 (Rank 12)
    # A01*B11 -> C01: Schoolbook 2x3x3 (Rank 18)
    # A10*B00 -> C10: Schoolbook 3x2x2 (Rank 12)
    # A11*B10 -> C10: Schoolbook 3x3x2 (Rank 18)
    # A10*B01 -> C11: Schoolbook 3x2x3 (Rank 18)
    # A11*B11 -> C11: Schoolbook 3x3x3 (Rank 27)
    
    total_rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook term for block A[ar:ar+an, ac:ac+am] * B[br:br+bm, bc:bc+bp] -> C[cr:cr+cn, cc:cc+cp]
    # Note: A is n x m, B is m x p, C is n x p
    # A block: rows ar..ar+an-1, cols ac..ac+am-1
    # B block: rows br..br+bm-1, cols bc..bc+bp-1
    # C block: rows cr..cr+cn-1, cols cc..cc+cp-1
    # am == bm (inner dimension)
    def add_schoolbook_block(an, am, bp, ar, ac, br, bc, cr, cc):
        nonlocal col
        for i in range(an):
            for j in range(am):
                for k in range(bp):
                    # A index: (ar+i, ac+j)
                    a_idx = (ar + i) * m + (ac + j)
                    # B index: (br+j, bc+k)
                    b_idx = (br + j) * p + (bc + k)
                    # C index: (cr+i, cc+k) -> W index: (cc+k)*n + (cr+i)
                    c_idx = (cc + k) * n + (cr + i)
                    
                    U[a_idx][col] = 1
                    V[b_idx][col] = 1
                    W[c_idx][col] = 1
                    col += 1

    # Helper to add Strassen block for 2x2x2
    # A block: 2x2 at (ar, ac)
    # B block: 2x2 at (br, bc)
    # C block: 2x2 at (cr, cc)
    def add_strassen_block(ar, ac, br, bc, cr, cc):
        nonlocal col
        # Strassen intermediates M1..M7
        # Each Mk is a product of linear forms in A and B.
        # We decompose each Mk into rank-1 tensors.
        
        # Define the linear forms for A and B for each Mk
        # A indices: (0,0), (0,1), (1,0), (1,1) relative to block
        # B indices: (0,0), (0,1), (1,0), (1,1) relative to block
        
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Coefficients for A in each Mk
        # u_mk[i, j] where i,j in 0..1
        u_forms = [
            [[1, 0], [0, 1]],   # M1: A00 + A11
            [[0, 0], [1, 1]],   # M2: A10 + A11
            [[1, 0], [0, 0]],   # M3: A00
            [[0, 0], [0, 1]],   # M4: A11
            [[1, 0], [1, 0]],   # M5: A00 + A10
            [[-1, 0], [1, 0]],  # M6: A10 - A00
            [[0, 1], [0, -1]]   # M7: A01 - A11
        ]
        
        # Coefficients for B in each Mk
        # v_mk[i, j] where i,j in 0..1
        v_forms = [
            [[1, 0], [0, 1]],   # M1: B00 + B11
            [[1, 0], [0, 0]],   # M2: B00
            [[0, 1], [0, -1]],  # M3: B10 - B11
            [[0, 1], [-1, 0]],  # M4: B01 - B00
            [[0, 0], [0, 1]],   # M5: B11
            [[1, 1], [0, 0]],   # M6: B00 + B01
            [[0, 1], [0, 1]]    # M7: B10 + B11
        ]
        
        # Coefficients for C in each Mk
        # w_mk[i, j] where i,j in 0..1 (C00, C01, C10, C11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        w_forms = [
            [[1, 0], [0, 1]],   # M1
            [[0, 0], [1, 0]],   # M2
            [[0, 1], [0, 0]],   # M3
            [[1, 0], [0, 0]],   # M4
            [[-1, 1], [0, 0]],  # M5
            [[0, 0], [0, 1]],   # M6
            [[1, 0], [0, 0]]    # M7
        ]
        
        for k in range(7):
            # Expand Mk = (sum u A) (sum v B)
            # This is a sum of terms u[i,j] * v[p,q] * A[i,j] * B[p,q]
            # Each such term contributes to C with weight w[i,k]
            # Wait, the C index is (i, k) in the block?
            # The tensor term is sum_{i,j,p,q} u[i,j] v[p,q] w[i,q] A[i,j] B[p,q] C[i,q]?
            # No, the term is u A v B w C.
            # The contribution to the tensor is u[i,j] v[p,q] w[i,q] E_{ij} E_{pq} E_{iq}?
            # We need to add rank-1 tensors.
            # A rank-1 tensor is defined by vectors u_r, v_r, w_r.
            # Here, for a fixed k, we have a product of two linear forms.
            # We can decompose this product into rank-1 tensors by iterating over the components.
            # Term: (sum_{i,j} u[i,j] A[i,j]) * (sum_{p,q} v[p,q] B[p,q]) * (sum_{r,s} w[r,s] C[r,s])
            # This is a tensor of rank 1? No, it's a product of three linear forms.
            # It is a rank-1 tensor in the space of (A, B, C).
            # So for each k, we add ONE rank-1 tensor?
            # No, the forms are sums.
            # u_total = sum u[i,j] E_{ij}.
            # v_total = sum v[p,q] E_{pq}.
            # w_total = sum w[r,s] E_{rs}.
            # The term is u_total \otimes v_total \otimes w_total.
            # This is a single rank-1 tensor!
            
            # So for each k, we add one rank-1 tensor with coefficients:
            # U_vec[i,j] = u_forms[k][i][j]
            # V_vec[p,q] = v_forms[k][p][q]
            # W_vec[r,s] = w_forms[k][r][s]
            
            # Map local indices to global
            for i in range(2):
                for j in range(2):
                    a_idx = (ar + i) * m + (ac + j)
                    U[a_idx][col] = u_forms[k][i][j]
                    
            for p in range(2):
                for q in range(2):
                    b_idx = (br + p) * p + (bc + q)
                    V[b_idx][col] = v_forms[k][p][q]
                    
            for r in range(2):
                for s in range(2):
                    c_idx = (cc + s) * n + (cr + r)
                    W[c_idx][col] = w_forms[k][r][s]
                    
            col += 1

    # Block 1: A00 (2x2) * B00 (2x2) -> C00 (2x2)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A01 (2x3) * B10 (3x2) -> C00 (2x2)
    add_schoolbook_block(2, 3, 2, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A00 (2x2) * B01 (2x3) -> C01 (2x3)
    add_schoolbook_block(2, 2, 3, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A01 (2x3) * B11 (3x3) -> C01 (2x3)
    add_schoolbook_block(2, 3, 3, 0, 2, 2, 2, 0, 2)
    
    # Block 5: A10 (3x2) * B00 (2x2) -> C10 (3x2)
    add_schoolbook_block(3, 2, 2, 2, 0, 0, 0, 2, 0)
    
    # Block 6: A11 (3x3) * B10 (3x2) -> C10 (3x2)
    add_schoolbook_block(3, 3, 2, 2, 2, 2, 0, 2, 0)
    
    # Block 7: A10 (3x2) * B01 (2x3) -> C11 (3x3)
    add_schoolbook_block(3, 2, 3, 2, 0, 0, 2, 2, 2)
    
    # Block 8: A11 (3x3) * B11 (3x3) -> C11 (3x3)
    add_schoolbook_block(3, 3, 3, 2, 2, 2, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
