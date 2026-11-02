import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_U_cache = None
_V_cache = None
_W_cache = None

def _build_strassen_4x4():
    """
    Constructs the Rank-49 Strassen decomposition for 4x4 matrix multiplication.
    Returns U, V, W as numpy arrays.
    
    The matrix multiplication tensor T_{ijk} = sum_k A_{ik} B_{kj} can be decomposed 
    into rank-1 terms sum_{r=1}^{R} u_r \otimes v_r \otimes w_r.
    For 4x4 matrices, we use the recursive Strassen algorithm:
    1. Split 4x4 matrices into 2x2 blocks of 2x2 submatrices.
    2. Apply Strassen's 7-product formula to the block level.
    3. Each block product is a 2x2 matrix multiplication, which is also decomposed 
       using Strassen's 7-product formula.
    Total rank = 7 * 7 = 49.
    """
    n = m = p = 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Indexing helpers
    def u_idx(a_row, a_col):
        return a_row * m + a_col
        
    def v_idx(b_row, b_col):
        return b_row * p + b_col
        
    def w_idx(c_row, c_col):
        # Official trace convention: W[k*n+i, r] where k is col, i is row?
        # The task says: "output factor follows the official trace convention W[k*n+i,r]"
        # Usually W is the matrix that maps the products to the output C.
        # C_{ij} = sum_k A_{ik} B_{kj}.
        # The tensor is T_{ikj} = A_{ik} B_{kj} -> C_{ij}.
        # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
        # This implies C = sum_r (U_r * A) \odot (V_r * B) \odot W_r? 
        # No, standard tensor decomposition for matmul:
        # M(A,B) = sum_{r=1}^R (U_r A) \circ (V_r B) \circ W_r^T ?
        # Let's stick to the standard definition used in these competitions.
        # The matrices U, V, W are such that:
        # vec(C) = sum_{r} (W_r \otimes I) (vec(A) \circ (U_r^T vec(B)))? 
        # Actually, the standard form is:
        # C_{ij} = sum_{k} A_{ik} B_{kj}
        # Decomposition: C = sum_{r} (A U_r) \circ (V_r B) \circ W_r
        # Where U_r, V_r, W_r are matrices.
        # Here U, V, W are the factors.
        # U has shape (n*m, rank). This corresponds to the "A" part.
        # V has shape (m*p, rank). This corresponds to the "B" part.
        # W has shape (p*n, rank). This corresponds to the "C" part.
        # The entry W[k*n+i, r] suggests that the index in W is (col, row) flattened as col*n+row.
        # Standard row-major flattening of C (n x p) would be i*p + k.
        # But the prompt says W[k*n+i, r]. This is column-major flattening of C (n x p) if we consider C_{i,k}.
        # Or it's just a specific indexing convention for the W matrix rows.
        # Let's assume the row index in W corresponds to the pair (c_col, c_row) such that index = c_col * n + c_row.
        # This matches my w_idx function above: c_col * n + c_row.
        return c_col * n + c_row

    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to map these 7 "outer" products to the 16x16 blocks.
    # Each "outer" product M_t is a product of two 2x2 matrices (blocks of the 4x4).
    # Let A be 4x4, partitioned into 2x2 blocks A_{ij} (each 2x2).
    # Let B be 4x4, partitioned into 2x2 blocks B_{ij} (each 2x2).
    
    # The outer terms define how the 4x4 matrices A and B are combined to form the inputs to the 7 block-multiplications.
    # For each outer term t (0..6), we have:
    # L_t: A linear combination of the 4 blocks of A to form a 2x2 matrix A'_t
    # R_t: A linear combination of the 4 blocks of B to form a 2x2 matrix B'_t
    # C_t: A linear combination of the 4 resulting 2x2 block-products to form the contribution to C.
    
    # However, since the block size is 2x2, and we are using Strassen recursively,
    # we don't just compute A'_t * B'_t as a single scalar. We decompose it further.
    
    # Let's define the 7 Strassen steps for a 2x2 matrix multiplication.
    # Let X, Y be 2x2 matrices.
    # We want to compute X*Y.
    # We can use the 7 products:
    # P1 = (X11+X22)(Y11+Y22)
    # ... etc.
    
    # Instead of nested loops, we can precompute the 49 vectors.
    # The total rank is 49.
    # We can index the 49 terms as (t, s) where t is the outer Strassen term (0..6) and s is the inner Strassen term (0..6).
    
    # Outer Strassen definitions (for 2x2 blocks of 4x4 matrices):
    # A is 4x4. Blocks A00, A01, A10, A11 (each 2x2).
    # B is 4x4. Blocks B00, B01, B10, B11 (each 2x2).
    
    # We define the 7 outer products M_t = A'_t B'_t.
    # A'_t and B'_t are 2x2 matrices formed by linear combinations of the 4 blocks of A and B.
    # The result M_t is a 2x2 matrix (conceptually), which is then added to C with coefficients.
    
    # But M_t itself is a matrix product of two 2x2 matrices.
    # So we apply the 7 inner Strassen products to compute M_t.
    
    # Let's define the coefficients for the outer Strassen step.
    # Each term t corresponds to a product of two 2x2 matrices (formed from A and B blocks).
    # Let's represent the "left" and "right" 2x2 matrices for each outer term.
    
    # Outer Term 0: (A00 + A11)(B00 + B11)
    # Left 2x2 matrix L0:
    #   L0_00 = A00_00 + A11_00 ? No, A00 is a 2x2 block.
    #   The "matrix" A'_0 is a 2x2 matrix where:
    #   (A'_0)_{00} = A00_{00} + A11_{00} ? No.
    #   Wait, A00 is the top-left 2x2 block of A.
    #   So A00 has entries A[0,0], A[0,1], A[1,0], A[1,1].
    #   A11 has entries A[2,2], A[2,3], A[3,2], A[3,3].
    #   The sum A00 + A11 is a 2x2 matrix.
    #   So A'_0 = A00 + A11.
    #   Similarly B'_0 = B00 + B11.
    #   The product M0 = A'_0 B'_0 is a 2x2 matrix.
    #   This 2x2 matrix M0 contributes to C.
    #   Specifically, M0 contributes to C00 and C11?
    #   In standard Strassen:
    #   C00 = M1 - M2 + M4 + M5
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M3 + M6 + M7
    #   So M0 (which is M1) contributes +1 to C00 and +1 to C11.
    
    # So, for each outer term t, we have:
    # 1. A 2x2 matrix A'_t (linear combo of A's 4 blocks)
    # 2. A 2x2 matrix B'_t (linear combo of B's 4 blocks)
    # 3. A 2x2 matrix C_contrib_t (coefficients for how M_t adds to C's 4 blocks)
    
    # Then, for each inner term s (0..6), we decompose the product A'_t B'_t.
    # The product A'_t B'_t is computed using 7 scalar multiplications (inner Strassen).
    # Let P_s be the s-th inner product.
    # P_s = (A'_t)_L_s * (B'_t)_R_s
    # Where (A'_t)_L_s is a scalar linear combo of the 4 entries of A'_t.
    # And (B'_t)_R_s is a scalar linear combo of the 4 entries of B'_t.
    
    # The result of the inner Strassen for A'_t B'_t is a 2x2 matrix M_t.
    # The entries of M_t are linear combinations of P_0..P_6.
    # Let M_t[i,j] = sum_s C_inner[s, i, j] * P_s.
    
    # The final contribution to C is:
    # C += C_contrib_t \odot M_t
    # Where \odot is the Hadamard product? No, C_contrib_t is a 2x2 matrix of coefficients.
    # C_block_{i,j} += (C_contrib_t)_{i,j} * M_t[i,j]?
    # No, in Strassen, C00 = M1 - M2 + M4 + M5.
    # This means the entire 2x2 matrix M1 is added to C00, M2 is subtracted from C00, etc.
    # So C_block_{i,j} += (C_contrib_t)_{i,j} * M_t.
    # Wait, M_t is a 2x2 matrix. C_block_{i,j} is a 2x2 matrix.
    # So C_block_{i,j} += coeff * M_t.
    
    # So, for a specific (t, s):
    # The scalar product is P_{t,s} = (linear combo of A entries) * (linear combo of B entries).
    # This P_{t,s} contributes to M_t[i,j] with coefficient C_inner[s, i, j].
    # And M_t[i,j] contributes to C_block_{k,l} with coefficient (C_contrib_t)_{k,l}.
    # So the total coefficient for P_{t,s} in C_block_{k,l}[r,c] is:
    # (C_contrib_t)_{k,l} * C_inner[s, i, j] * (where i,j are the positions in M_t)
    # Wait, M_t is the result of the block multiplication.
    # M_t is a 2x2 matrix.
    # The entry M_t[i,j] is a scalar.
    # C_block_{k,l} is a 2x2 matrix.
    # The formula is C_block_{k,l} += (C_contrib_t)_{k,l} * M_t.
    # This means every entry of C_block_{k,l} is increased by (C_contrib_t)_{k,l} * M_t[i,j]?
    # No, M_t is a 2x2 matrix. C_block_{k,l} is a 2x2 matrix.
    # The standard Strassen formula is:
    # C00 = M1 - M2 + M4 + M5
    # Here M1 is a 2x2 matrix. C00 is a 2x2 matrix.
    # So C00 = M1 - M2 + ... means element-wise addition.
    # So C00[r,c] = M1[r,c] - M2[r,c] + ...
    
    # So, for a specific entry C[r,c] (where r,c are in 0..3):
    # Let k = r // 2, l = c // 2. (Block index)
    # Let i = r % 2, j = c % 2. (Local index within block)
    # C[r,c] = sum_{t=0}^6 (C_contrib_t)_{k,l} * M_t[i,j]
    # M_t[i,j] = sum_{s=0}^6 C_inner[s, i, j] * P_{t,s}
    # P_{t,s} = (sum_{a,b} U_{t,s,a,b} A[a,b]) * (sum_{d,e} V_{t,s,d,e} B[d,e])
    
    # So C[r,c] = sum_{t,s} ( (C_contrib_t)_{k,l} * C_inner[s, i, j] ) * P_{t,s}
    
    # We need to construct U, V, W vectors.
    # U[:, t*7+s] is the vector of coefficients for A.
    # V[:, t*7+s] is the vector of coefficients for B.
    # W[:, t*7+s] is the vector of coefficients for C.
    
    # W_idx(c_row, c_col) = c_col * 4 + c_row.
    # So W[row, r] is non-zero only for the specific C entry it contributes to.
    # Actually, W is a matrix of shape (16, 49).
    # W[c_idx, r] is the coefficient for C[c_idx] in the r-th term.
    # C[c_idx] is a scalar.
    # So W[c_idx, r] = (C_contrib_t)_{k,l} * C_inner[s, i, j]
    # where r = t*7+s, and c_idx corresponds to (k,l,i,j).
    
    # Let's define the inner Strassen coefficients.
    # For 2x2 matrices X, Y:
    # P1 = (X11+X22)(Y11+Y22)
    # P2 = (X21+X22)Y11
    # P3 = X11(Y12-Y22)
    # P4 = X22(Y21-Y11)
    # P5 = (X11+X12)Y22
    # P6 = (X21-X11)(Y11+Y12)
    # P7 = (X12-X22)(Y21+Y22)
    
    # X coefficients for P_s:
    # P1: X11:1, X22:1
    # P2: X21:1, X22:1
    # P3: X11:1
    # P4: X22:1
    # P5: X11:1, X12:1
    # P6: X21:1, X11:-1
    # P7: X12:1, X22:-1
    
    # Y coefficients for P_s:
    # P1: Y11:1, Y22:1
    # P2: Y11:1
    # P3: Y12:1, Y22:-1
    # P4: Y21:1, Y11:-1
    # P5: Y22:1
    # P6: Y11:1, Y12:1
    # P7: Y21:1, Y22:1
    
    # Result Z = XY coefficients for P_s (Z_ij = sum_s C_sij P_s):
    # Z11 = P1 - P2 + P4 + P5
    # Z12 = P3 + P5
    # Z21 = P2 + P4
    # Z22 = P1 - P3 + P6 + P7
    
    # Inner Coeffs C_inner[s, i, j]:
    # s=0 (P1): Z11:+1, Z22:+1
    # s=1 (P2): Z11:-1, Z21:+1
    # s=2 (P3): Z12:+1, Z22:-1
    # s=3 (P4): Z11:+1, Z21:+1
    # s=4 (P5): Z11:+1, Z12:+1
    # s=5 (P6): Z22:+1
    # s=6 (P7): Z22:+1
    
    # Let's store these as lists of dicts.
    # X_coeffs[s] = {(row, col): coeff}
    # Y_coeffs[s] = {(row, col): coeff}
    # Z_coeffs[s] = {(row, col): coeff}
    
    X_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    Y_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    Z_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): -1, (1,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,0): 1, (1,0): 1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(1,1): 1}
    ]
    
    # Outer Strassen definitions
    # A'_t and B'_t are 2x2 matrices.
    # A'_t[i,j] = sum_{block_r, block_c} L_t[block_r, block_c, i, j] * A[2*block_r+i, 2*block_c+j]
    # B'_t[i,j] = sum_{block_r, block_c} R_t[block_r, block_c, i, j] * B[2*block_r+i, 2*block_c+j]
    
    # L_t and R_t are 2x2x2x2 tensors? Or just 2x2 matrices of coefficients for the blocks?
    # No, A'_t is a 2x2 matrix. Its entries are linear combos of A's entries.
    # A'_t[i,j] is a linear combo of all 16 entries of A?
    # No, A'_t is formed from the 4 blocks.
    # A'_t[i,j] = L_t[i,j,0,0]*A00[i,j] + L_t[i,j,0,1]*A01[i,j] + ...
    # So L_t is a 2x2x2x2 tensor?
    # Actually, it's simpler:
    # For each outer term t, we define how to form the 2x2 matrix A'_t.
    # A'_t is a 2x2 matrix.
    # Each entry A'_t[i,j] is a linear combination of the 16 entries of A.
    # But in Strassen, A'_t is formed by adding/subtracting whole blocks.
    # So A'_t[i,j] = sum_{br,bc} coeff[br,bc] * A[2*br+i, 2*bc+j].
    
    # Let's define L_t as a list of 4 dicts, one for each block (br,bc).
    # L_t[br][bc] is a 2x2 matrix of coefficients?
    # No, the coefficient is the same for all i,j in the block?
    # Yes, because we are adding blocks.
    # So L_t[br][bc] is a scalar.
    # A'_t[i,j] = sum_{br,bc} L_t[br][bc] * A[2*br+i, 2*bc+j].
    
    # Outer Term 0: (A00 + A11)(B00 + B11)
    # L0: A00:1, A11:1. Others 0.
    # R0: B00:1, B11:1. Others 0.
    # C_contrib0: C00:1, C11:1. Others 0.
    
    # Let's define L_t, R_t, C_contrib_t as 2x2 matrices of scalars.
    # L_t[br][bc] is the coefficient for block (br,bc).
    
    L_outer = [
        [[1, 0], [0, 1]], # T0: A00+A11
        [[0, 0], [1, 1]], # T1: A10+A11
        [[1, 0], [0, 0]], # T2: A00
        [[0, 0], [0, 1]], # T3: A11
        [[1, 1], [0, 0]], # T4: A00+A01
        [[0, 0], [1, -1]],# T5: A10-A00 ? No. (A21-A11) in 2x2 notation is (A10-A00).
                         # Wait, standard Strassen:
                         # M6 = (A21 - A11)(B11 + B12)
                         # In 0-indexed blocks: A10 - A00.
                         # So L5[1][0]=1, L5[0][0]=-1.
        [[0, 1], [0, -1]]  # T6: A01-A11
    ]
    
    R_outer = [
        [[1, 0], [0, 1]], # T0: B00+B11
        [[1, 0], [0, 0]], # T1: B00
        [[0, 1], [0, -1]],# T2: B01-B11
        [[0, -1], [1, 0]],# T3: B10-B00
        [[0, 0], [0, 1]], # T4: B11
        [[1, 1], [0, 0]], # T5: B00+B01
        [[0, 0], [1, 1]]  # T6: B10+B11
    ]
    
    C_contrib_outer = [
        [[1, 0], [0, 1]], # T0: M1 -> C00, C11
        [[0, 0], [1, -1]],# T1: M2 -> C10, -C11 ?
                         # Standard: C10 = M2+M4, C11 = M1-M3+M6+M7.
                         # So M2 contributes +1 to C10, 0 to others?
                         # Wait, C00 = M1 - M2 + M4 + M5.
                         # So M2 contributes -1 to C00?
                         # Let's re-verify Strassen formulas.
                         # C00 = M1 - M2 + M4 + M5
                         # C01 = M3 + M5
                         # C10 = M2 + M4
                         # C11 = M1 - M3 + M6 + M7
                         
                         # T0 (M1): C00:+1, C11:+1
                         # T1 (M2): C00:-1, C10:+1
                         # T2 (M3): C01:+1, C11:-1
                         # T3 (M4): C00:+1, C10:+1
                         # T4 (M5): C00:+1, C01:+1
                         # T5 (M6): C11:+1
                         # T6 (M7): C11:+1
    ]
    
    C_contrib_outer = [
        [[1, 0], [0, 1]], # M1
        [[-1, 0], [1, 0]],# M2
        [[0, 1], [0, -1]],# M3
        [[1, 0], [1, 0]], # M4
        [[1, 1], [0, 0]], # M5
        [[0, 0], [0, 1]], # M6
        [[0, 0], [0, 1]]  # M7
    ]
    
    for t in range(7):
        for s in range(7):
            r_idx = t * 7 + s
            
            # Construct U vector (coefficients for A)
            u_vec = np.zeros(16)
            # A'_t[i,j] = sum_{br,bc} L_outer[t][br][bc] * A[2*br+i, 2*bc+j]
            # P_{t,s} uses X_coeffs[s][i,j] * A'_t[i,j]
            # So coeff for A[2*br+i, 2*bc+j] is sum_{i,j} X_coeffs[s][i,j] * L_outer[t][br][bc]
            
            for br in range(2):
                for bc in range(2):
                    l_coeff = L_outer[t][br][bc]
                    if l_coeff == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            x_coeff = X_coeffs[s].get((i,j), 0)
                            if x_coeff == 0:
                                continue
                            a_row = 2*br + i
                            a_col = 2*bc + j
                            u_vec[u_idx(a_row, a_col)] += l_coeff * x_coeff
            
            # Construct V vector (coefficients for B)
            v_vec = np.zeros(16)
            for br in range(2):
                for bc in range(2):
                    r_coeff = R_outer[t][br][bc]
                    if r_coeff == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            y_coeff = Y_coeffs[s].get((i,j), 0)
                            if y_coeff == 0:
                                continue
                            b_row = 2*br + i
                            b_col = 2*bc + j
                            v_vec[v_idx(b_row, b_col)] += r_coeff * y_coeff
            
            # Construct W vector (coefficients for C)
            w_vec = np.zeros(16)
            # C[r,c] += sum_{t,s} C_contrib_outer[t][k,l] * Z_coeffs[s][i,j] * P_{t,s}
            # where k=r//2, l=c//2, i=r%2, j=c%2
            # So for a given (t,s), the coefficient for C[r,c] is:
            # C_contrib_outer[t][r//2][c//2] * Z_coeffs[s][r%2][c%2]
            
            for c_row in range(4):
                for c_col in range(4):
                    k = c_row // 2
                    l = c_col // 2
                    i = c_row % 2
                    j = c_col % 2
                    
                    c_coeff_outer = C_contrib_outer[t][k][l]
                    if c_coeff_outer == 0:
                        continue
                    z_coeff = Z_coeffs[s].get((i,j), 0)
                    if z_coeff == 0:
                        continue
                    
                    total_coeff = c_coeff_outer * z_coeff
                    w_vec[w_idx(c_row, c_col)] += total_coeff
            
            U[:, r_idx] = u_vec
            V[:, r_idx] = v_vec
            W[:, r_idx] = w_vec
            
    return U, V, W

def construct():
    global _U_cache, _V_cache, _W_cache
    
    if _U_cache is None:
        _U_cache, _V_cache, _W_cache = _build_strassen_4x4()
        
    return {"U": _U_cache.tolist(), "V": _V_cache.tolist(), "W": _W_cache.tolist()}
# EVOLVE-BLOCK-END
