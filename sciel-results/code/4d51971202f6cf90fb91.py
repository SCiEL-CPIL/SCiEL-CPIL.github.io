"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Use a rank-64 decomposition for the first 4 columns (4x4x4)
    # and a rank-16 schoolbook decomposition for the 5th column (4x4x1).
    # Total rank = 64 + 16 = 80.
    # Wait, the current program is already rank 80 (schoolbook for all).
    # To improve, we need a rank < 80.
    #
    # Let's try to construct a rank-64 decomposition for 4x4x4.
    # We can use the Kronecker product of a rank-4 decomposition of 2x2x2?
    # No, rank(2x2x2) is 7 (Strassen). So 7*7 = 49.
    # But Strassen's 49 rank decomposition for 4x4x4 is valid.
    # If we use rank 49 for the first 4 columns and rank 16 for the last, total is 65.
    # This is a significant improvement over 80.
    #
    # Let's implement the Strassen 4x4x4 decomposition carefully.
    #
    # The tensor for 4x4x4 multiplication can be decomposed as follows:
    # We view A, B, C as 2x2 block matrices with 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen's algorithm computes 7 products of 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # Then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M_i is a product of two 2x2 matrices.
    # We use Strassen's rank-7 decomposition for each 2x2 product.
    # So each M_i contributes 7 rank-1 terms.
    # Total 7 * 7 = 49 rank-1 terms for the 4x4x4 part.
    #
    # For the 5th column of C (which is 4x1), we compute A * B_col5.
    # This is a 4x4 matrix times a 4x1 vector.
    # Schoolbook rank is 16.
    #
    # Total rank = 49 + 16 = 65.
    
    rank = 49 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen Decomposition (Rank 49) ---
    
    # Strassen linear forms for 2x2 blocks
    # A_strassen[m] is a 4x4 matrix of coefficients for A's blocks?
    # No, A_strassen[m] is a 2x2 matrix of coefficients for the 4 blocks of A.
    # Let's represent the linear form on A as a 4x4 matrix of coefficients.
    # coeff_A[m][i, j] is the coefficient of A[i, j] in the m-th linear form.
    
    # M1 = (A00 + A11)(B00 + B11)
    # A form: A00 + A11 -> coeff: [[1,0],[0,1]] for A00, [[1,0],[0,1]] for A11
    # So coeff_A[0] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]? No.
    # A is 4x4. A00 is top-left 2x2.
    # A00 = A[0:2, 0:2]
    # A11 = A[2:4, 2:4]
    # So coeff_A[0] has 1s at (0,0), (1,1), (2,2), (3,3).
    
    coeff_A = [
        # M1: A00 + A11
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=int),
        # M2: A10 + A11
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=int),
        # M3: A00
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int),
        # M4: A11
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=int),
        # M5: A00 + A01
        np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int),
        # M6: A10 - A00
        np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=int),
        # M7: A01 - A11
        np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=int)
    ]
    
    coeff_B = [
        # M1: B00 + B11
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=int),
        # M2: B00
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int),
        # M3: B01 - B11
        np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=int),
        # M4: B10 - B00
        np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=int),
        # M5: B11
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=int),
        # M6: B00 + B01
        np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int),
        # M7: B10 + B11
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=int)
    ]
    
    # Coefficients for C blocks in terms of M's
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    coeff_C = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Rank-7 decomposition of 2x2x2 (Strassen)
    # U_2x2, V_2x2, W_2x2 are 7x4 matrices (flattened 2x2)
    U_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    V_2x2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    W_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    col = 0
    for m_idx in range(7):
        a_form = coeff_A[m_idx]
        b_form = coeff_B[m_idx]
        
        for r_idx in range(7):
            u_inner = U_2x2[r_idx]
            v_inner = V_2x2[r_idx]
            w_inner = W_2x2[r_idx]
            
            # Compute U_vec for this rank-1 term
            # U_vec[i, j] = sum_{i1, j1} a_form[i1, j1] * u_inner[(i - 2*i1), (j - 2*j1)]?
            # No. a_form is a 4x4 matrix of coefficients for A[i,j].
            # u_inner is a 4-element vector representing a linear form on a 2x2 matrix.
            # The linear form on A for the m-th Strassen step is:
            # S_A^m = sum_{i,j} a_form[i,j] A[i,j]
            # This is a 4x4 matrix of linear forms? No, it's a 4x4 matrix of scalars (coeffs).
            # The actual matrix S_A^m is a 4x4 matrix where S_A^m[i,j] = a_form[i,j] * A[i,j]?
            # No, that's not how Strassen works.
            # Strassen combines blocks.
            # S_A^m is a 2x2 block matrix where each block is a linear form.
            # Actually, S_A^m is a 4x4 matrix.
            # S_A^m[i,j] = coeff_A[m][i,j] * A[i,j].
            # So the linear form on A is simply the vector of coeff_A[m] flattened.
            # But we need to decompose the product S_A^m * S_B^m.
            # S_A^m is a 4x4 matrix. S_B^m is a 4x4 matrix.
            # We cannot use the 2x2 Strassen decomposition directly on 4x4 matrices.
            # We must view S_A^m and S_B^m as 2x2 block matrices of 2x2 blocks.
            # Let S_A^m = [[X00, X01], [X10, X11]] where Xij are 2x2 matrices.
            # Xij = coeff_A[m][2*i:2*i+2, 2*j:2*j+2] * A[2*i:2*i+2, 2*j:2*j+2]?
            # No. coeff_A[m] is a 4x4 matrix.
            # Xij = coeff_A[m][2*i:2*i+2, 2*j:2*j+2].
            # This Xij is a 2x2 matrix of coefficients.
            # The linear form on the block A_ij is:
            # sum_{p,q} Xij[p,q] * A_ij[p,q].
            # This is a scalar linear form.
            # So S_A^m is a 2x2 matrix of scalar linear forms.
            # We can apply the 2x2 Strassen decomposition to this 2x2 matrix of forms.
            
            # Let's extract the 2x2 blocks of coeff_A[m] and coeff_B[m].
            A_blocks = np.array([
                [a_form[0:2, 0:2], a_form[0:2, 2:4]],
                [a_form[2:4, 0:2], a_form[2:4, 2:4]]
            ])
            B_blocks = np.array([
                [b_form[0:2, 0:2], b_form[0:2, 2:4]],
                [b_form[2:4, 0:2], b_form[2:4, 2:4]]
            ])
            
            # Now apply 2x2 Strassen decomposition to A_blocks and B_blocks.
            # The rank-1 term is:
            # U_vec[i, j] = sum_{i1, j1} u_inner[(i - 2*i1), (j - 2*j1)] * A_blocks[i1, j1][(i - 2*i1), (j - 2*j1)]?
            # No. u_inner is a linear form on a 2x2 matrix.
            # u_inner[0] * M00 + u_inner[1] * M01 + u_inner[2] * M10 + u_inner[3] * M11.
            # Here M is a 2x2 matrix of linear forms.
            # M[i1, j1] is a linear form on the 2x2 block A[i1, j1].
            # M[i1, j1] = sum_{p,q} A_blocks[i1, j1][p, q] * A[2*i1+p, 2*j1+q].
            # So the total linear form on A is:
            # sum_{i1, j1} u_inner[i1*2+j1] * sum_{p,q} A_blocks[i1, j1][p, q] * A[2*i1+p, 2*j1+q].
            # U_vec[2*i1+p, 2*j1+q] += u_inner[i1*2+j1] * A_blocks[i1, j1][p, q].
            
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for i1 in range(2):
                for j1 in range(2):
                    u_coeff = u_inner[i1 * 2 + j1]
                    if u_coeff != 0:
                        block = A_blocks[i1, j1]
                        for p in range(2):
                            for q in range(2):
                                u_vec[(2 * i1 + p) * m + (2 * j1 + q)] += u_coeff * block[p, q]
            
            for i1 in range(2):
                for j1 in range(2):
                    v_coeff = v_inner[i1 * 2 + j1]
                    if v_coeff != 0:
                        block = B_blocks[i1, j1]
                        for p in range(2):
                            for q in range(2):
                                v_vec[(2 * i1 + p) * p + (2 * j1 + q)] += v_coeff * block[p, q]
            
            # W_vec: The output is a 4x4 matrix C.
            # C = sum_{m} sum_{r} (u_r(S_A^m)) (v_r(S_B^m)) w_r(S_C^m)
            # S_C^m is a 2x2 matrix of linear forms on C.
            # The contribution to C from M_m is determined by coeff_C.
            # For each block (i1, j1) of C, the coefficient of M_m is coeff_C[(i1, j1)][m_idx].
            # Let c_coeff = coeff_C[(i1, j1)][m_idx] if m_idx is in the list, else 0.
            # The linear form on C for this rank-1 term is:
            # sum_{i1, j1} c_coeff(i1, j1) * w_inner[i1*2+j1] * sum_{p,q} C[2*i1+p, 2*j1+q]
            # W_vec[2*i1+p, 2*j1+q] += c_coeff(i1, j1) * w_inner[i1*2+j1].
            
            for i1 in range(2):
                for j1 in range(2):
                    # Find coefficient of M_m in C[i1, j1]
                    c_coeff = 0
                    for (m_id, c_val) in coeff_C.get((i1, j1), []):
                        if m_id == m_idx:
                            c_coeff += c_val
                    if c_coeff != 0:
                        w_coeff = w_inner[i1 * 2 + j1] * c_coeff
                        if w_coeff != 0:
                            for p in range(2):
                                for q in range(2):
                                    w_vec[(2 * j1 + q) * n + (2 * i1 + p)] += w_coeff
            
            # Add to U, V, W
            for i in range(n * m):
                U[i][col] = u_vec[i]
            for i in range(m * p):
                V[i][col] = v_vec[i]
            for i in range(p * n):
                W[i][col] = w_vec[i]
            col += 1
    
    # --- Part 2: 4x4x1 Schoolbook Decomposition (Rank 16) ---
    # C[:, 4] = A * B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1  # B[j, 4]
            W[4 * n + i][col] = 1  # C[4, i] -> W[k*n+i, r] with k=4
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
