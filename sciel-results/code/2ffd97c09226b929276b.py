"""Program for rank-minimal 5x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition exactly using integer scaling by 2."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)
    
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    
    # Target tensor: T[a*m+b, b*p+c, c*n+a] = 1 for all (a,b,c)
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    
    # T_computed[i,j,k] = sum_c U2[i,c] * V2[j,c] * W2[k,c]
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    
    # Since each factor is scaled by 2, product is scaled by 8
    if np.max(np.abs(T_computed - target * 8)) > 1e-6:
        return False
    return True

def _build_strassen_2x2():
    """Return U, V, W factors for 2x2x2 Strassen (rank 7)."""
    # Strassen's 7 products for 2x2 matrices
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]  # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    V = [[0]*7 for _ in range(4)]  # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    W = [[0]*7 for _ in range(4)]  # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: A=(A00+A11), B=(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: A=(A10+A11), B=B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A=A00, B=(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A=A11, B=(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: A=(A00+A01), B=B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: A=(A10-A00), B=(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: A=(A01-A11), B=(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen's 7-term formula applied at the 2x2 block level,
    where each 2x2 block multiplication uses rank-7 Strassen.
    Total rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # For 4x4, we have 2x2 blocks. Each block is a 2x2 matrix.
    # A has blocks A00, A01, A10, A11 (each 2x2)
    # B has blocks B00, B01, B10, B11
    # C has blocks C00, C01, C10, C11
    
    # At the macro level (2x2 block multiplication), Strassen uses 7 products:
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)*B00 -> contributes to C10, C11
    # M3 = A00*(B01-B11) -> contributes to C01, C11
    # M4 = A11*(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)*B11 -> contributes to C00, C01
    # M6 = (A10-A00)*(B00+B01) -> contributes to C11
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00
    
    # Each macro product M_k is itself a 2x2 matrix product, which uses 7 Strassen terms.
    # So for each macro k, we have 7 micro terms, total 7*7 = 49.
    
    rank = 49
    U = [[0]*rank for _ in range(16)]  # 4x4 A = 16 entries
    V = [[0]*rank for _ in range(16)]  # 4x4 B = 16 entries
    W = [[0]*rank for _ in range(16)]  # 4x4 C = 16 entries
    
    # Macro-level Strassen coefficients for A-blocks (A00,A01,A10,A11)
    # and B-blocks (B00,B01,B10,B11)
    # For each macro term k, list of (a_block, coeff_a, b_block, coeff_b)
    macro_terms = [
        # M1: (A00+A11)*(B00+B11)
        [(0, 1, 0, 1), (0, 1, 1, 1), (3, 1, 0, 1), (3, 1, 1, 1)],
        # M2: (A10+A11)*B00
        [(2, 1, 0, 1), (3, 1, 0, 1)],
        # M3: A00*(B01-B11)
        [(0, 1, 1, 1), (0, 1, 3, -1)],
        # M4: A11*(B10-B00)
        [(3, 1, 2, 1), (3, 1, 0, -1)],
        # M5: (A00+A01)*B11
        [(0, 1, 3, 1), (1, 1, 3, 1)],
        # M6: (A10-A00)*(B00+B01)
        [(2, 1, 0, 1), (2, 1, 1, 1), (0, -1, 0, 1), (0, -1, 1, 1)],
        # M7: (A01-A11)*(B10+B11)
        [(1, 1, 2, 1), (1, 1, 3, 1), (3, -1, 2, 1), (3, -1, 3, 1)],
    ]
    
    # For each macro term, we expand into 7 micro Strassen terms for the 2x2 block product.
    # The 2x2 Strassen factors U2, V2, W2 give coefficients for the 2x2 matrices within each block.
    # Block A_ij has 2x2 entries at global positions:
    # A00: indices 0,1,4,5 (row 0: 0,1; row 1: 4,5)
    # A01: indices 2,3,6,7
    # A10: indices 8,9,12,13
    # A11: indices 10,11,14,15
    
    # Local 2x2 index to global index mapping for a block at (block_row, block_col)
    def block_global_index(block_row, block_col, local_row, local_col):
        """Map block position and local 2x2 index to global 4x4 index."""
        # Global row = block_row * 2 + local_row
        # Global col = block_col * 2 + local_col
        # A is stored as a*4+b where a=row, b=col
        return (block_row * 2 + local_row) * 4 + (block_col * 2 + local_col)
    
    # Same for B (m x p = 4 x 4): index = row*4 + col
    def b_global_index(block_row, block_col, local_row, local_col):
        return (block_row * 2 + local_row) * 4 + (block_col * 2 + local_col)
    
    # C is stored as c*n+i where c=col, i=row (per problem statement W[k*n+i])
    # So C index = col * 4 + row
    def c_global_index(block_col, block_row, local_col, local_row):
        return (block_col * 2 + local_col) * 4 + (block_row * 2 + local_row)
    
    col = 0
    for macro_k in range(7):
        # For this macro term, we have a linear combination of A-blocks and B-blocks.
        # Each "virtual" A-block and B-block undergoes 2x2 Strassen multiplication.
        # We need to expand each macro term into 7 micro terms.
        
        # The macro term k involves certain A-blocks with coefficients and B-blocks with coefficients.
        # For the 2x2 Strassen at the micro level, each entry of the "combined" A-block
        # is a linear combination of entries from the relevant A-blocks, and similarly for B.
        
        # Let's build the effective 2x2 A and B matrices for this macro term.
        # For each of the 4 positions in the 2x2 block (local_row, local_col in {0,1}),
        # the A entry is sum over contributing A-blocks of coeff * A_block_entry
        # The B entry is sum over contributing B-blocks of coeff * B_block_entry
        
        # Get the macro term's A-block contributions
        a_contribs = {}  # (block_row, block_col) -> coeff
        b_contribs = {}
        for (a_blk, ca, b_blk, cb) in macro_terms[macro_k]:
            a_blk_row, a_blk_col = a_blk // 2, a_blk % 2
            b_blk_row, b_blk_col = b_blk // 2, b_blk % 2
            a_contribs[(a_blk_row, a_blk_col)] = a_contribs.get((a_blk_row, a_blk_col), 0) + ca
            b_contribs[(b_blk_row, b_blk_col)] = b_contribs.get((b_blk_row, b_blk_col), 0) + cb
        
        # Now, for each of the 7 micro Strassen terms, we assign a column.
        # The micro Strassen factors U2, V2, W2 tell us which local 2x2 entries to combine.
        for micro_l in range(7):
            # U2[local_idx][micro_l] gives coefficient for A local entry local_idx
            # V2[local_idx][micro_l] gives coefficient for B local entry local_idx
            # W2[local_idx][micro_l] gives coefficient for C local entry local_idx
            
            for local_idx in range(4):
                u_coeff = U2[local_idx][micro_l]
                v_coeff = V2[local_idx][micro_l]
                w_coeff = W2[local_idx][micro_l]
                
                # local_idx maps to local (row, col) in 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                local_row, local_col = local_idx // 2, local_idx % 2
                
                # For U: sum over A-blocks that contribute to this local position
                for (a_blk_r, a_blk_c), ca in a_contribs.items():
                    if u_coeff != 0 and ca != 0:
                        global_idx = block_global_index(a_blk_r, a_blk_c, local_row, local_col)
                        U[global_idx][col] += u_coeff * ca
                
                # For V: sum over B-blocks
                for (b_blk_r, b_blk_c), cb in b_contribs.items():
                    if v_coeff != 0 and cb != 0:
                        global_idx = b_global_index(b_blk_r, b_blk_c, local_row, local_col)
                        V[global_idx][col] += v_coeff * cb
                
                # For W: the C-block receiving this micro term's output
                # The macro term k contributes to certain C-blocks.
                # We need to know which C-blocks get M_k and with what sign.
                
                # Let's determine C-block contributions for macro term k
                c_block_contribs = {}  # (c_blk_row, c_blk_col) -> sign
                if macro_k == 0:  # M1
                    c_block_contribs = {(0,0): 1, (1,1): 1}
                elif macro_k == 1:  # M2
                    c_block_contribs = {(1,0): 1, (1,1): -1}
                elif macro_k == 2:  # M3
                    c_block_contribs = {(0,1): 1, (1,1): 1}
                elif macro_k == 3:  # M4
                    c_block_contribs = {(0,0): 1, (1,0): 1}
                elif macro_k == 4:  # M5
                    c_block_contribs = {(0,0): -1, (0,1): 1}
                elif macro_k == 5:  # M6
                    c_block_contribs = {(1,1): 1}
                elif macro_k == 6:  # M7
                    c_block_contribs = {(0,0): 1}
                
                for (c_blk_r, c_blk_c), sign in c_block_contribs.items():
                    if w_coeff != 0 and sign != 0:
                        global_idx = c_global_index(c_blk_c, c_blk_r, local_col, local_row)
                        W[global_idx][col] += w_coeff * sign
            
            col += 1
    
    # Verify and return
    assert col == 49, f"Expected 49 columns, got {col}"
    return U, V, W

def _build_5x5_decomposition():
    """
    Build a rank-100 decomposition for 5x5x5 matrix multiplication.
    Strategy: Split 5 = 4 + 1.
    - Use rank-49 Strassen for the 4x4 sub-block.
    - Handle cross terms between 4-part and 1-part naively.
    - Handle the 1x1 corner naively.
    
    Cross terms:
    - A[4:5, 0:4] * B[0:4, 0:4] -> C[0:4, 4:5] (outer product, 4*4=16 terms)
    - A[0:4, 4:5] * B[4:5, 0:4] -> C[0:4, 0:4] (outer product, 4*4=16 terms)
    - A[0:4, 4:5] * B[4:5, 4:5] -> C[0:4, 4:5] (outer product, 4*1=4 terms)
    - A[4:5, 0:4] * B[0:4, 4:5] -> C[4:5, 0:4] (outer product, 1*4=4 terms)
    - A[4:5, 4:5] * B[4:5, 4:5] -> C[4:5, 4:5] (1 term)
    - A[4:5, 0:4] * B[0:4, 0:4] -> C[4:5, 0:4] already counted
    - A[0:4, 0:4] * B[4:5, 0:4] -> C[0:4, 0:4] already counted
    
    Wait, let me be more careful. C[i,j] = sum_k A[i,k] B[k,j].
    Split i into i4 (0-3) and i5 (4), k into k4 (0-3) and k5 (4), j into j4 (0-3) and j5 (4).
    
    C[i4,j4] = sum_{k4} A[i4,k4] B[k4,j4] + A[i4,k5] B[k5,j4]
    C[i4,j5] = sum_{k4} A[i4,k4] B[k4,j5] + A[i4,k5] B[k5,j5]
    C[i5,j4] = sum_{k4} A[i5,k4] B[k4,j4] + A[i5,k5] B[k5,j4]
    C[i5,j5] = sum_{k4} A[i5,k4] B[k4,j5] + A[i5,k5] B[k5,j5]
    
    So:
    - C[i4,j4] has: 4x4 Strassen (rank 49) + 4 terms (A[i4,4]*B[4,j4])
    - C[i4,j5] has: 4 terms (A[i4,k4]*B[k4,4]) + 1 term (A[i4,4]*B[4,4])
    - C[i5,j4] has: 4 terms (A[i5,k4]*B[k4,j4]) + 1 term (A[i5,4]*B[4,j4])
    - C[i5,j5] has: 4 terms (A[i5,k4]*B[k4,4]) + 1 term (A[i5,4]*B[4,4])
    
    Total cross terms: 4 + 4 + 1 + 4 + 1 + 4 + 1 = 19? Let me recount.
    
    Actually each "cross term" is a single product A[i,k]*B[k,j] for specific i,k,j.
    
    For C[i4,j4]: i4 in {0,1,2,3}, j4 in {0,1,2,3}, k5=4. That's 4*4=16 terms.
    For C[i4,j5]: i4 in {0,1,2,3}, j5=4, k4 in {0,1,2,3}. That's 4*4=16 terms.
    For C[i4,j5]: i4=4? No, i4 means i in 0-3. i5=4.
    
    Let me redo:
    - Terms with k=4 (the 5th index):
      * i in 0-3, j in 0-3: A[i,4]*B[4,j] -> 16 terms
      * i in 0-3, j=4: A[i,4]*B[4,4] -> 4 terms
      * i=4, j in 0-3: A[4,4]*B[4,j] -> 4 terms
      * i=4, j=4: A[4,4]*B[4,4] -> 1 term
      Total with k=4: 16+4+4+1 = 25
    
    - Terms with i=4, k in 0-3:
      * i=4, k in 0-3, j in 0-3: A[4,k]*B[k,j] -> 4*4=16 terms
      * i=4, k in 0-3, j=4: A[4,k]*B[k,4] -> 4 terms
      Total: 20
    
    - Terms with j=4, k in 0-3, i in 0-3:
      * i in 0-3, k in 0-3, j=4: A[i,k]*B[k,4] -> 4*4=16 terms
      Total: 16
    
    But we've double-counted some. Let me just enumerate all 125 terms and subtract the 4x4 block.
    
    4x4 block terms: i in 0-3, k in 0-3, j in 0-3. That's 4*4*4 = 64 terms.
    Total terms: 125.
    Cross terms: 125 - 64 = 61.
    
    So total rank = 49 + 61 = 110.
    """
    U4, V4, W4 = _build_strassen_4x4()
    
    rank = 49 + 61  # 110
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    # First 49 columns: 4x4 Strassen
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            V[i][c] = V4[i][c]
            W[i][c] = W4[i][c]
    
    # Remaining 61 columns: cross terms (naive)
    col = 49
    for i in range(5):
        for k in range(5):
            for j in range(5):
                if i < 4 and k < 4 and j < 4:
                    continue  # Already handled by 4x4 Strassen
                U[i*5 + k][col] = 1
                V[k*5 + j][col] = 1
                W[j*5 + i][col] = 1
                col += 1
    
    assert col == 110, f"Expected 110 columns, got {col}"
    return U, V, W

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Build the rank-110 decomposition
    U, V, W = _build_5x5_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        print(f"Found valid rank-110 decomposition")
        return {"U": U, "V": V, "W": W}
    else:
        print("Rank-110 verification failed, falling back to schoolbook")
    
    # Fallback: schoolbook rank 125
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
