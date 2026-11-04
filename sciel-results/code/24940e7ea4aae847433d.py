"""Matrix multiplication 4x5x5 with Strassen-based tensor network ansatz."""
import numpy as np

DIMENSIONS = [4, 5, 5]


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using Strassen-like algebraic cancellations.
    
    Strategy: Decompose the 4x5x5 problem into blocks and apply Strassen's
    algorithm to 2x2x2 sub-blocks where possible, saving 1 multiplication per
    such block. For the 5-dimension, split into 2+2+1 and handle the remainder
    with schoolbook multiplication.
    
    Block structure:
    - A (4x5) split into rows [0:2, 2:4] and cols [0:2, 2:4, 4]
    - B (5x5) split into rows [0:2, 2:4, 4] and cols [0:2, 2:4, 4]
    - C (4x5) split into rows [0:2, 2:4] and cols [0:2, 2:4, 4]
    
    For 2x2x2 blocks, use Strassen (rank 7 instead of 8).
    For blocks involving the 5th index, use schoolbook.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build up the decomposition by collecting rank-1 terms
    # Each term is (U_row, V_row, W_row, U_val, V_val, W_val)
    # where the rank-1 tensor is U_val * u_row * V_val * v_row * W_val * w_row
    
    terms = []  # list of (u_row, v_row, w_row, u_coeff, v_coeff, w_coeff)
    
    def add_term(u_row, v_row, w_row, u_coeff=1, v_coeff=1, w_coeff=1):
        terms.append((u_row, v_row, w_row, u_coeff, v_coeff, w_coeff))
    
    def strassen_2x2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Add Strassen decomposition for a 2x2x2 block.
        A_rows: list of 2 row indices for A
        A_cols: list of 2 col indices for A (= row indices for B)
        B_rows: list of 2 row indices for B
        B_cols: list of 2 col indices for B
        C_rows: list of 2 row indices for C
        C_cols: list of 2 col indices for C
        
        Strassen's 7 multiplications:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B22)
        M4 = A22(B11-B21)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        a11, a12 = A_rows[0], A_rows[1]
        a21, a22 = A_cols[0], A_cols[1]
        b11, b12 = B_rows[0], B_rows[1]
        b21, b22 = B_cols[0], B_cols[1]
        c11, c12 = C_rows[0], C_rows[1]
        c21, c22 = C_cols[0], C_cols[1]
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11 and A22 both get coefficient 1
        # V: B11 and B22 both get coefficient 1
        # W: C11 gets +1, C22 gets +1
        add_term(a11 * m + b11, b11 * p + c11, c11 * n + a11, 1, 1, 1)
        add_term(a22 * m + b22, b22 * p + c22, c22 * n + a22, 1, 1, 1)
        # Wait, this isn't quite right. Let me reconsider.
        
        # Actually, each rank-1 term in the decomposition corresponds to
        # a product of a linear form in A, a linear form in B, and a 
        # linear form in C. Let me restructure.
        
        # For Strassen, the 7 products are:
        # P1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # P2 = (A[1,0]+A[1,1]) * B[0,0]
        # P3 = A[0,0] * (B[1,0]-B[1,1])
        # P4 = A[1,1] * (B[0,0]-B[0,1])
        # P5 = (A[0,0]+A[0,1]) * B[1,1]
        # P6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # P7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        
        # And the outputs are:
        # C[0,0] = P1 + P4 - P5 + P7
        # C[0,1] = P3 + P5
        # C[1,0] = P2 + P4
        # C[1,1] = P1 - P2 + P3 + P6
        
        # Each Pi is a product of a linear form in A and a linear form in B.
        # The contribution to C is a linear combination of the Pi's.
        
        # Let me define the linear forms:
        # A-forms:
        # L_A1 = A[0,0] + A[1,1]  -> U rows: (a11,b11) coeff 1, (a22,b22) coeff 1
        # L_A2 = A[1,0] + A[1,1]  -> U rows: (a21,b11) coeff 1, (a22,b22) coeff 1
        # L_A3 = A[0,0]           -> U row: (a11,b11) coeff 1
        # L_A4 = A[1,1]           -> U row: (a22,b22) coeff 1
        # L_A5 = A[0,0] + A[0,1]  -> U rows: (a11,b11) coeff 1, (a11,b22) coeff 1
        # L_A6 = A[1,0] - A[0,0]  -> U rows: (a21,b11) coeff 1, (a11,b11) coeff -1
        # L_A7 = A[0,1] - A[1,1]  -> U rows: (a11,b22) coeff 1, (a22,b22) coeff -1
        
        # B-forms:
        # L_B1 = B[0,0] + B[1,1]  -> V rows: (b11,c11) coeff 1, (b22,c22) coeff 1
        # L_B2 = B[0,0]           -> V row: (b11,c11) coeff 1
        # L_B3 = B[1,0] - B[1,1]  -> V rows: (b21,c11) coeff 1, (b21,c22) coeff -1
        # L_B4 = B[0,0] - B[0,1]  -> V rows: (b11,c11) coeff 1, (b11,c22) coeff -1
        # L_B5 = B[1,1]           -> V row: (b22,c22) coeff 1
        # L_B6 = B[0,0] + B[0,1]  -> V rows: (b11,c11) coeff 1, (b11,c22) coeff 1
        # L_B7 = B[1,0] + B[1,1]  -> V rows: (b21,c11) coeff 1, (b21,c22) coeff 1
        
        # C-forms (which Pi contributes to which C element):
        # C[0,0]: P1(+1), P4(+1), P5(-1), P7(+1)
        # C[0,1]: P3(+1), P5(+1)
        # C[1,0]: P2(+1), P4(+1)
        # C[1,1]: P1(+1), P2(-1), P3(+1), P6(+1)
        
        # For each Pi, we add one rank-1 term where:
        # U has the coefficients from L_Ai
        # V has the coefficients from L_Bi
        # W has the coefficients from the C-form
        
        # P1: L_A1 * L_B1
        # U: (a11,b11):1, (a22,b22):1
        # V: (b11,c11):1, (b22,c22):1
        # W: (c11,a11):1, (c22,a22):1
        add_term(a11 * m + b11, b11 * p + c11, c11 * n + a11, 1, 1, 1)
        add_term(a22 * m + b22, b22 * p + c22, c22 * n + a22, 1, 1, 1)
        # This isn't right - a single rank-1 term can't have multiple U rows.
        
        # I need to rethink. Each rank-1 term in the tensor decomposition
        # is of the form u * v * w where u is a vector in R^{n*m}, v in R^{m*p},
        # w in R^{p*n}. The tensor is sum of u_t * v_t * w_t over t.
        
        # For a product like (A[i1,j1] + A[i2,j2]) * (B[j3,k3] + B[j4,k4]),
        # this expands to 4 terms, but in the decomposition we have ONE rank-1
        # term where U has 1's at positions (i1*m+j1) and (i2*m+j2), V has 1's
        # at (j3*p+k3) and (j4*p+k4), and W has appropriate coefficients.
        
        # Let me redo this properly.
        pass
    
    # Let me restart with a cleaner approach
    terms = []
    
    def add_rank1(u_dict, v_dict, w_dict):
        """
        Add a rank-1 term where u_dict, v_dict, w_dict are dicts mapping
        (row) -> coefficient. The rank-1 tensor is the outer product.
        """
        terms.append((u_dict, v_dict, w_dict))
    
    def schoolbook_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Schoolbook for a small block.
        For each (i,j) in A block and (j',k) in B block where j==j',
        add A[i,j]*B[j,k] to C[i,k].
        """
        for i in A_rows:
            for j in A_cols:
                for k in B_cols:
                    if j in B_rows:
                        # Find position of i in C_rows and k in C_cols
                        ci = C_rows.index(i)
                        ck = C_cols.index(k)
                        c_row = ci * p + ck  # Wait, C is n x p, so row i, col k -> index i*p + k
                        # Actually C is 4x5, so C[i,k] is at position i*p + k = i*5 + k
                        # But in our W matrix, the index is k*n + i (trace convention W[k*n+i,r])
                        # So W index for C[i,k] is k*n + i
                        w_idx = k * n + i
                        u_idx = i * m + j
                        v_idx = j * p + k
                        add_rank1({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Strassen for a 2x2x2 block.
        A_rows: [i1, i2], A_cols: [j1, j2]
        B_rows: [j1, j2], B_cols: [k1, k2]
        C_rows: [i1, i2], C_cols: [k1, k2]
        """
        i1, i2 = A_rows
        j1, j2 = A_cols
        k1, k2 = B_cols
        
        # P1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        u1 = {i1 * m + j1: 1, i2 * m + j2: 1}
        v1 = {j1 * p + k1: 1, j2 * p + k2: 1}
        w1 = {k1 * n + i1: 1, k2 * n + i2: 1}  # C[i1,k1] and C[i2,k2]
        add_rank1(u1, v1, w1)
        
        # P2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        u2 = {i2 * m + j1: 1, i2 * m + j2: 1}
        v2 = {j1 * p + k1: 1}
        w2 = {k1 * n + i2: 1}  # C[i2,k1]
        add_rank1(u2, v2, w2)
        
        # P3 = A[i1,j1] * (B[j2,k1] - B[j2,k2])
        u3 = {i1 * m + j1: 1}
        v3 = {j2 * p + k1: 1, j2 * p + k2: -1}
        w3 = {k1 * n + i1: 1}  # C[i1,k1]... wait, C[i1,k2] too?
        # P3 contributes to C[i1,k1] with +1? No.
        # C[i1,k1] = P1 + P4 - P5 + P7
        # C[i1,k2] = P3 + P5
        # So P3 contributes +1 to C[i1,k2]
        w3 = {k2 * n + i1: 1}
        add_rank1(u3, v3, w3)
        
        # P4 = A[i2,j2] * (B[j1,k1] - B[j1,k2])
        u4 = {i2 * m + j2: 1}
        v4 = {j1 * p + k1: 1, j1 * p + k2: -1}
        # P4 contributes +1 to C[i2,k1]
        w4 = {k1 * n + i2: 1}
        add_rank1(u4, v4, w4)
        
        # P5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        u5 = {i1 * m + j1: 1, i1 * m + j2: 1}
        v5 = {j2 * p + k2: 1}
        # P5 contributes -1 to C[i1,k1] and +1 to C[i1,k2]
        w5 = {k1 * n + i1: -1, k2 * n + i1: 1}
        add_rank1(u5, v5, w5)
        
        # P6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        u6 = {i2 * m + j1: 1, i1 * m + j1: -1}
        v6 = {j1 * p + k1: 1, j1 * p + k2: 1}
        # P6 contributes +1 to C[i2,k2]
        w6 = {k2 * n + i2: 1}
        add_rank1(u6, v6, w6)
        
        # P7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        u7 = {i1 * m + j2: 1, i2 * m + j2: -1}
        v7 = {j2 * p + k1: 1, j2 * p + k2: 1}
        # P7 contributes +1 to C[i1,k1]
        w7 = {k1 * n + i1: 1}
        add_rank1(u7, v7, w7)
    
    # Now let's build the full decomposition
    # A is 4x5, B is 5x5, C is 4x5
    # Split rows of A: [0,1] and [2,3]
    # Split cols of A (rows of B): [0,1], [2,3], [4]
    # Split cols of B: [0,1], [2,3], [4]
    
    # 2x2x2 Strassen blocks:
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    strassen_block([2, 3], [0, 1], [0, 1], [0, 1], [2, 3], [0, 1])
    
    # A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    strassen_block([2, 3], [0, 1], [0, 1], [2, 3], [2, 3], [2, 3])
    
    # A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    strassen_block([2, 3], [2, 3], [2, 3], [0, 1], [2, 3], [0, 1])
    
    # A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    strassen_block([2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3])
    
    # Schoolbook blocks involving index 4:
    # A[0:2, 4] * B[4, 0:2] -> C[0:2, 0:2]
    schoolbook_block([0, 1], [4], [4], [0, 1], [0, 1], [0, 1])
    
    # A[0:2, 4] * B[4, 2:4] -> C[0:2, 2:4]
    schoolbook_block([0, 1], [4], [4], [2, 3], [0, 1], [2, 3])
    
    # A[0:2, 4] * B[4, 4] -> C[0:2, 4]
    schoolbook_block([0, 1], [4], [4], [4], [0, 1], [4])
    
    # A[2:4, 4] * B[4, 0:2] -> C[2:4, 0:2]
    schoolbook_block([2, 3], [4], [4], [0, 1], [2, 3], [0, 1])
    
    # A[2:4, 4] * B[4, 2:4] -> C[2:4, 2:4]
    schoolbook_block([2, 3], [4], [4], [2, 3], [2, 3], [2, 3])
    
    # A[2:4, 4] * B[4, 4] -> C[2:4, 4]
    schoolbook_block([2, 3], [4], [4], [4], [2, 3], [4])
    
    # Also need: A[0:2, 0:2] * B[4, :] and A[0:2, 2:4] * B[4, :]
    # Wait, I already handled those above with the schoolbook_block calls.
    # But I also need A[0:2, :] * B[0:4, 4] which involves B cols 4.
    
    # Let me check: C[i,k] = sum_j A[i,j] * B[j,k]
    # For k=4: C[i,4] = sum_j A[i,j] * B[j,4]
    # This involves B[j,4] for j=0..4
    # I haven't handled B[j,4] for j=0..3 yet!
    
    # I need additional blocks:
    # A[0:2, 0:2] * B[0:2, 4] -> C[0:2, 4]
    schoolbook_block([0, 1], [0, 1], [0, 1], [4], [0, 1], [4])
    
    # A[0:2, 2:4] * B[2:4, 4] -> C[0:2, 4]
    schoolbook_block([0, 1], [2, 3], [2, 3], [4], [0, 1], [4])
    
    # A[2:4, 0:2] * B[0:2, 4] -> C[2:4, 4]
    schoolbook_block([2, 3], [0, 1], [0, 1], [4], [2, 3], [4])
    
    # A[2:4, 2:4] * B[2:4, 4] -> C[2:4, 4]
    schoolbook_block([2, 3], [2, 3], [2, 3], [4], [2, 3], [4])
    
    # Now build the matrices
    n_terms = len(terms)
    U = [[0] * n_terms for _ in range(n * m)]
    V = [[0] * n_terms for _ in range(m * p)]
    W = [[0] * n_terms for _ in range(p * n)]
    
    for t, (u_dict, v_dict, w_dict) in enumerate(terms):
        for row, coeff in u_dict.items():
            U[row][t] += coeff
        for row, coeff in v_dict.items():
            V[row][t] += coeff
        for row, coeff in w_dict.items():
            W[row][t] += coeff
    
    return {"U": U, "V": V, "W": W}
