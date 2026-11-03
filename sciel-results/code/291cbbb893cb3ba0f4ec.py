import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    # C_{2x6} part: 42 rank
    # C_{1x6} part: 24 rank
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Part 1: C_{2x6} = A_{2x4} B_{4x6}
    # A_{2x4} = [A11 A12; A21 A22] (2x2 blocks)
    # B_{4x6} = [B11 B12 B13; B21 B22 B23] (2x2 blocks, 3 cols)
    
    # Block 1: A1 * B1 (A1 is 2x2, B1 is 2x6)
    # A1 rows: 0,1. Cols: 0,1
    # B1 rows: 0,1. Cols: 0,1,2,3,4,5 (3 blocks of 2)
    
    # Strassen for 2x2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # We need to map the local indices to global indices.
    # Global U index: i*m + j
    # Global V index: j*p + k
    # Global W index: k*n + i
    
    # A1 is at i=0,1 and j=0,1.
    # B1 is at j=0,1 and k=0..5.
    
    # Sub-block 1 of B1: k=0,1
    # a11=U[0*4+0]=U[0], a12=U[0*4+1]=U[1]
    # a21=U[1*4+0]=U[4], a22=U[1*4+1]=U[5]
    # b11=V[0*6+0]=V[0], b12=V[0*6+1]=V[1]
    # b21=V[1*6+0]=V[6], b22=V[1*6+1]=V[7]
    # c11=W[0*3+0]=W[0], c12=W[0*3+1]=W[1]
    # c21=W[1*3+0]=W[3], c22=W[1*3+1]=W[4]
    
    # This manual mapping is error-prone. I will write a helper to generate the terms.
    
    def strassen_222(a_i, a_j, b_j, b_k, c_k, c_i):
        # a_i: start row of A block (2x2)
        # a_j: start col of A block (2x2)
        # b_j: start row of B block (2x2)
        # b_k: start col of B block (2x2)
        # c_k: start col of C block (2x2)
        # c_i: start row of C block (2x2)
        
        # A indices
        a11_idx = (a_i) * m + (a_j)
        a12_idx = (a_i) * m + (a_j + 1)
        a21_idx = (a_i + 1) * m + (a_j)
        a22_idx = (a_i + 1) * m + (a_j + 1)
        
        # B indices
        b11_idx = (b_j) * p + (b_k)
        b12_idx = (b_j) * p + (b_k + 1)
        b21_idx = (b_j + 1) * p + (b_k)
        b22_idx = (b_j + 1) * p + (b_k + 1)
        
        # C indices
        c11_idx = (c_k) * n + (c_i)
        c12_idx = (c_k + 1) * n + (c_i)
        c21_idx = (c_k) * n + (c_i + 1)
        c22_idx = (c_k + 1) * n + (c_i + 1)
        
        # M1
        add_term(a11_idx, b11_idx, c11_idx, 1, 1, 1)
        add_term(a11_idx, b22_idx, c11_idx, 1, 1, 1)
        add_term(a22_idx, b11_idx, c11_idx, 1, 1, 1)
        add_term(a22_idx, b22_idx, c11_idx, 1, 1, 1)
        # Wait, M1 is (a11+a22)(b11+b22). This expands to 4 terms.
        # But in tensor decomposition, we want rank-1 terms of the form u v w.
        # (a11+a22) is a linear form. (b11+b22) is a linear form.
        # So M1 corresponds to 1 rank-1 term in the decomposition of the tensor?
        # No, the term is (a11+a22) * (b11+b22) * (c11+c22+c21+c12 ...)?
        # Strassen computes M1, then combines M's to get C.
        # The tensor decomposition requires expressing the tensor T = sum C_{ij} A_{ik} B_{kj} as sum R_t U_t V_t W_t.
        # Each M_t is a scalar. The contribution to C is linear in M_t.
        # So if C_11 = M_1 + ..., then the term M_1 contributes to C_11.
        # The rank-1 term is (a11+a22) * (b11+b22) * (1 for c11, 1 for c22, etc).
        
        # I need to add the terms that make up M1.
        # M1 = (a11+a22)(b11+b22).
        # This is a single product of two linear forms.
        # So it is ONE rank-1 component of the tensor?
        # Yes, if we define the U, V, W vectors to contain the coefficients of the linear forms.
        
        # So for M1:
        # U part: 1 at a11, 1 at a22.
        # V part: 1 at b11, 1 at b22.
        # W part: 1 at c11, 1 at c12, 1 at c21, 1 at c22?
        # No, M1 contributes to C11, C12, C21, C22 with specific coefficients.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes +1 to C11, +1 to C22.
        # M2 contributes +1 to C21, -1 to C22.
        # M3 contributes +1 to C12, +1 to C22.
        # M4 contributes +1 to C11, +1 to C21.
        # M5 contributes -1 to C11, +1 to C12.
        # M6 contributes +1 to C22.
        # M7 contributes +1 to C11.
        
        # So for each M_t, we add ONE rank-1 term.
        # The U vector has the coefficients of the A linear form.
        # The V vector has the coefficients of the B linear form.
        # The W vector has the coefficients of the C linear form (how M_t contributes to C).
        
        # M1: U: a11, a22. V: b11, b22. W: c11, c22.
        add_term(a11_idx, b11_idx, c11_idx, 1, 1, 1) # This is wrong. add_term adds to a specific column.
        # I need to construct the U, V, W vectors for the column.
        
    # The add_term function above is for a single non-zero entry.
    # I need a function to add a full rank-1 term defined by linear forms.
    
    def add_rank1_term(u_coeff, v_coeff, w_coeff):
        # u_coeff: list of (idx, val)
        # v_coeff: list of (idx, val)
        # w_coeff: list of (idx, val)
        nonlocal col
        for idx, val in u_coeff:
            U[idx][col] += val
        for idx, val in v_coeff:
            V[idx][col] += val
        for idx, val in w_coeff:
            W[idx][col] += val
        col += 1

    # Strassen 2x2x2
    def strassen_block(a_i, a_j, b_j, b_k, c_k, c_i):
        a11 = (a_i, 1); a12 = (a_i, 2); a21 = (a_i+1, 1); a22 = (a_i+1, 2)
        b11 = (b_j, 1); b12 = (b_j, 2); b21 = (b_j+1, 1); b22 = (b_j+1, 2)
        c11 = (c_i, 1); c12 = (c_i, 2); c21 = (c_i+1, 1); c22 = (c_i+1, 2)
        
        # M1: (a11+a22)(b11+b22) -> C11, C22
        add_rank1_term(
            [((a_i)*m + a_j, 1), ((a_i+1)*m + a_j+1, 1)],
            [((b_j)*p + b_k, 1), ((b_j+1)*p + b_k+1, 1)],
            [((c_k)*n + c_i, 1), ((c_k+1)*n + c_i+1, 1)]
        )
        
        # M2: (a21+a22)b11 -> C21, C22(-1)
        add_rank1_term(
            [((a_i+1)*m + a_j, 1), ((a_i+1)*m + a_j+1, 1)],
            [((b_j)*p + b_k, 1)],
            [((c_k)*n + c_i+1, 1), ((c_k+1)*n + c_i+1, -1)]
        )
        
        # M3: a11(b12-b22) -> C12, C22
        add_rank1_term(
            [((a_i)*m + a_j, 1)],
            [((b_j)*p + b_k+1, 1), ((b_j+1)*p + b_k+1, -1)],
            [((c_k+1)*n + c_i, 1), ((c_k+1)*n + c_i+1, 1)]
        )
        
        # M4: a22(b21-b11) -> C11, C21
        add_rank1_term(
            [((a_i+1)*m + a_j+1, 1)],
            [((b_j+1)*p + b_k, 1), ((b_j)*p + b_k, -1)],
            [((c_k)*n + c_i, 1), ((c_k)*n + c_i+1, 1)]
        )
        
        # M5: (a11+a12)b22 -> C11(-1), C12
        add_rank1_term(
            [((a_i)*m + a_j, 1), ((a_i)*m + a_j+1, 1)],
            [((b_j+1)*p + b_k+1, 1)],
            [((c_k)*n + c_i, -1), ((c_k+1)*n + c_i, 1)]
        )
        
        # M6: (a21-a11)(b11+b12) -> C22
        add_rank1_term(
            [((a_i+1)*m + a_j, 1), ((a_i)*m + a_j, -1)],
            [((b_j)*p + b_k, 1), ((b_j)*p + b_k+1, 1)],
            [((c_k+1)*n + c_i+1, 1)]
        )
        
        # M7: (a12-a22)(b21+b22) -> C11
        add_rank1_term(
            [((a_i)*m + a_j+1, 1), ((a_i+1)*m + a_j+1, -1)],
            [((b_j+1)*p + b_k, 1), ((b_j+1)*p + b_k+1, 1)],
            [((c_k)*n + c_i, 1)]
        )

    # Part 1: C_{2x6}
    # A_{2x4} is rows 0,1.
    # B_{4x6} is rows 0..3.
    # Split A into two 2x2 blocks: cols 0,1 and 2,3.
    # Split B into two 2x6 blocks: rows 0,1 and 2,3.
    
    # Block A1 (cols 0,1) * B1 (rows 0,1)
    # B1 is 2x6, split into 3 2x2 blocks: k=0,1; k=2,3; k=4,5.
    for k_block in range(3):
        strassen_block(0, 0, 0, k_block*2, k_block*2, 0)
        
    # Block A2 (cols 2,3) * B2 (rows 2,3)
    for k_block in range(3):
        strassen_block(0, 2, 2, k_block*2, k_block*2, 0)

    # Part 2: C_{1x6}
    # A_{1x4} is row 2.
    # B_{4x6} is rows 0..3.
    # Split A into two 1x2 blocks: cols 0,1 and 2,3.
    # Split B into two 2x6 blocks: rows 0,1 and 2,3.
    
    # Schoolbook 1x2x2
    def schoolbook_122_block(a_i, a_j, b_j, b_k, c_k, c_i):
        # A: a1, a2
        # B: b11, b12; b21, b22
        # C: c1, c2
        # c1 = a1 b11 + a2 b21
        # c2 = a1 b12 + a2 b22
        
        # M1: a1 b11 -> c1
        add_rank1_term(
            [((a_i)*m + a_j, 1)],
            [((b_j)*p + b_k, 1)],
            [((c_k)*n + c_i, 1)]
        )
        # M2: a2 b21 -> c1
        add_rank1_term(
            [((a_i)*m + a_j+1, 1)],
            [((b_j+1)*p + b_k, 1)],
            [((c_k)*n + c_i, 1)]
        )
        # M3: a1 b12 -> c2
        add_rank1_term(
            [((a_i)*m + a_j, 1)],
            [((b_j)*p + b_k+1, 1)],
            [((c_k+1)*n + c_i, 1)]
        )
        # M4: a2 b22 -> c2
        add_rank1_term(
            [((a_i)*m + a_j+1, 1)],
            [((b_j+1)*p + b_k+1, 1)],
            [((c_k+1)*n + c_i, 1)]
        )

    # A1 (cols 0,1) * B1 (rows 0,1)
    for k_block in range(3):
        schoolbook_122_block(2, 0, 0, k_block*2, k_block*2, 2)
        
    # A2 (cols 2,3) * B2 (rows 2,3)
    for k_block in range(3):
        schoolbook_122_block(2, 2, 2, k_block*2, k_block*2, 2)

    return {"U": U, "V": V, "W": W}
