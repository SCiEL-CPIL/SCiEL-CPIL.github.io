"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's algorithm for 4x4 matrix multiplication
    # We partition A and B into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M4
    # C21 = M1 + M2
    # C22 = M3 - M2 + M5 + M6
    # Where:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each Mi is a 2x2 matrix multiplication, which requires 7 scalar multiplications (Strassen).
    # Total scalar multiplications = 7 * 7 = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need to map the 49 scalar multiplications to the U, V, W matrices.
    # The indices for A are i*m + j, where i,j in {0,1,2,3}.
    # The indices for B are j*p + k, where j,k in {0,1,2,3}.
    # The indices for C are k*n + i, where i,k in {0,1,2,3}.
    
    # We define the 7 Strassen matrices for a 2x2 block multiplication.
    # Let's define the linear forms for the 2x2 Strassen algorithm.
    # For a 2x2 matrix X = [[x0, x1], [x2, x3]] and Y = [[y0, y1], [y2, y3]]
    # The 7 multiplications are:
    # m1 = (x0 + x3)(y0 + y3)
    # m2 = (x2 + x3)y0
    # m3 = x0(y1 - y3)
    # m4 = x3(y2 - y0)
    # m5 = (x0 + x1)y3
    # m6 = (x2 - x0)(y0 + y1)
    # m7 = (x1 - x3)(y2 + y3)
    
    # We need to apply this to the 7 blocks of the 4x4 multiplication.
    # Each block multiplication uses 7 scalar multiplications.
    # We will iterate through the 7 block multiplications (M1 to M7) and for each,
    # iterate through the 7 scalar multiplications of the 2x2 Strassen algorithm.
    
    # Helper to get indices
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i
    
    # Define the 7 Strassen linear forms for 2x2 matrices
    # Each form is a list of (coeff, index) for U and (coeff, index) for V
    # For a 2x2 matrix, indices are 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    
    # We will define the 7 forms for the "inner" Strassen step.
    # form 1: U: [1, 0, 0, 1], V: [1, 0, 0, 1]
    # form 2: U: [0, 0, 1, 1], V: [1, 0, 0, 0]
    # form 3: U: [1, 0, 0, 0], V: [0, 1, 0, -1]
    # form 4: U: [0, 0, 0, 1], V: [0, 0, 1, -1]
    # form 5: U: [1, 1, 0, 0], V: [0, 0, 0, 1]
    # form 6: U: [0, 0, 1, -1], V: [1, 1, 0, 0]
    # form 7: U: [0, 1, 0, -1], V: [0, 0, 1, 1]
    
    inner_strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    inner_strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # The 7 block multiplications M1..M7 are defined by linear combinations of the 4x4 blocks.
    # We need to define the U and V coefficients for each of the 7 block multiplications.
    # Each block multiplication is a 2x2 matrix multiplication.
    # The "U" for the block multiplication is a 4x4 matrix (flattened to 16) with non-zero entries in the 2x2 blocks.
    # The "V" is similar.
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We define the U and V matrices for each block multiplication.
    # U_block is a 4x4 matrix, V_block is a 4x4 matrix.
    # We flatten them to vectors of length 16.
    
    def make_block_u(blocks):
        # blocks is a dict mapping (i,j) to coeff
        u = [0] * 16
        for (i, j), coeff in blocks.items():
            u[i * 4 + j] = coeff
        return u
        
    def make_block_v(blocks):
        v = [0] * 16
        for (j, k), coeff in blocks.items():
            v[j * 4 + k] = coeff
        return v
        
    # Define the 7 block U and V matrices
    # A blocks: A11(0,0), A12(0,1), A21(1,0), A22(1,1)
    # B blocks: B11(0,0), B12(0,1), B21(1,0), B22(1,1)
    
    # M1: U = A11 + A22, V = B11 + B22
    u1 = make_block_u({(0,0): 1, (1,1): 1})
    v1 = make_block_v({(0,0): 1, (1,1): 1})
    
    # M2: U = A21 + A22, V = B11
    u2 = make_block_u({(1,0): 1, (1,1): 1})
    v2 = make_block_v({(0,0): 1})
    
    # M3: U = A11, V = B12 - B22
    u3 = make_block_u({(0,0): 1})
    v3 = make_block_v({(0,1): 1, (1,1): -1})
    
    # M4: U = A22, V = B21 - B11
    u4 = make_block_u({(1,1): 1})
    v4 = make_block_v({(1,0): 1, (0,0): -1})
    
    # M5: U = A11 + A12, V = B22
    u5 = make_block_u({(0,0): 1, (0,1): 1})
    v5 = make_block_v({(1,1): 1})
    
    # M6: U = A21 - A11, V = B11 + B12
    u6 = make_block_u({(1,0): 1, (0,0): -1})
    v6 = make_block_v({(0,0): 1, (0,1): 1})
    
    # M7: U = A12 - A22, V = B21 + B22
    u7 = make_block_u({(0,1): 1, (1,1): -1})
    v7 = make_block_v({(1,0): 1, (1,1): 1})
    
    block_Us = [u1, u2, u3, u4, u5, u6, u7]
    block_Vs = [v1, v2, v3, v4, v5, v6, v7]
    
    # Now, for each block multiplication t (0..6), and each inner Strassen form s (0..6),
    # we create a column in U, V, W.
    
    # The W matrix for the inner Strassen algorithm:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M4
    # C21 = M1 + M2
    # C22 = M3 - M2 + M5 + M6
    # The output is a 2x2 matrix.
    # We need to map the 7 scalar products to the 4 entries of the 2x2 result.
    # For each inner form s, we have a scalar product.
    # The contribution to the 2x2 result is determined by the inner Strassen W matrix.
    
    inner_strassen_w = [
        # s=0 (M1)
        [1, 0, 1, 0],  # C11, C12, C21, C22
        # s=1 (M2)
        [0, 0, 1, -1],
        # s=2 (M3)
        [0, 1, 0, 1],
        # s=3 (M4)
        [-1, 1, 0, 0],
        # s=4 (M5)
        [0, 0, 0, 1],
        # s=5 (M6)
        [1, 0, 0, 1],
        # s=6 (M7)
        [1, 0, 0, 0]
    ]
    
    # The block result C is a 2x2 matrix of 2x2 blocks.
    # C11 (top-left 2x2), C12 (top-right 2x2), C21 (bottom-left 2x2), C22 (bottom-right 2x2).
    # The indices for C are (k, i) where k is row, i is col.
    # C11 corresponds to k in {0,1}, i in {0,1}.
    # C12 corresponds to k in {0,1}, i in {2,3}.
    # C21 corresponds to k in {2,3}, i in {0,1}.
    # C22 corresponds to k in {2,3}, i in {2,3}.
    
    # For each block multiplication t, the result is a 2x2 matrix of scalars (from inner Strassen).
    # These scalars are placed in the 2x2 blocks of the 4x4 matrix C.
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            # U column
            # u_block is 4x4, inner_u is 2x2.
            # The U vector is the product of the block U and the inner U.
            # Actually, the U matrix for the scalar multiplication is the composition.
            # U_scalar[i, j] = sum_{block_i, block_j} U_block[block_i, block_j] * U_inner[inner_i, inner_j]
            # Since the block U is sparse (only 2x2 blocks non-zero), we can compute it.
            
            u_col = [0] * 16
            for bi in range(2):
                for bj in range(2):
                    coeff_block = block_Us[t][bi * 4 + bj]
                    if coeff_block != 0:
                        for ii in range(2):
                            for jj in range(2):
                                coeff_inner = inner_strassen_u[s][ii * 2 + jj]
                                if coeff_inner != 0:
                                    # Global index: (bi*2 + ii, bj*2 + jj)
                                    gi = bi * 2 + ii
                                    gj = bj * 2 + jj
                                    u_col[gi * 4 + gj] += coeff_block * coeff_inner
                                # Wait, the inner Strassen U is for a 2x2 matrix.
                                # The block is 2x2.
                                # The indices of the 2x2 block are (ii, jj) where ii, jj in {0,1}.
                                # The global indices are (bi*2+ii, bj*2+jj).
                                # The inner U vector is indexed by (ii, jj) -> ii*2+jj.
                                # So we need to map the inner indices to the local block indices.
                                
            # Let's re-verify the mapping.
            # Block U is 4x4.
            # Inner U is 2x2.
            # The scalar multiplication is: sum_{a,b} U[a,b] A[a,b] * sum_{c,d} V[c,d] B[c,d]
            # U[a,b] = sum_{p,q} U_block[p,q] U_inner[a-2p, b-2q]
            
            for a in range(4):
                for b in range(4):
                    val = 0
                    for p in range(2):
                        for q in range(2):
                            ia = a - 2 * p
                            ib = b - 2 * q
                            if 0 <= ia < 2 and 0 <= ib < 2:
                                val += block_Us[t][p * 4 + q] * inner_strassen_u[s][ia * 2 + ib]
                    u_col[a * 4 + b] = val
                    U[a * 4 + b][col] = val
            
            # V column
            v_col = [0] * 16
            for c in range(4):
                for d in range(4):
                    val = 0
                    for p in range(2):
                        for q in range(2):
                            ic = c - 2 * p
                            id_ = d - 2 * q
                            if 0 <= ic < 2 and 0 <= id_ < 2:
                                val += block_Vs[t][p * 4 + q] * inner_strassen_v[s][ic * 2 + id_]
                    V[c * 4 + d][col] = val
            
            # W column
            # W is 4x4.
            # The result of the inner Strassen is a 2x2 matrix R.
            # R[0,0] = C11, R[0,1] = C12, R[1,0] = C21, R[1,1] = C22.
            # The block result is placed in the 4x4 matrix C.
            # C[bi*2+ii, bj*2+jj] = R[ii, jj] * W_block? No.
            # The W matrix for the scalar multiplication maps the scalar product to the 4x4 matrix C.
            # C[i, k] = sum_{t,s} W[i, k, t, s] * (U... * V...)
            # The contribution of the scalar product (t,s) to C[i,k] is:
            # sum_{p,q} W_inner[ii, jj] * 1 (if the block is at p,q)
            # Actually, the block result R is 2x2.
            # R[ii, jj] contributes to C[bi*2+ii, bj*2+jj].
            # So W[bi*2+ii, bj*2+jj] = W_inner[ii, jj]
            
            for ii in range(2):
                for jj in range(2):
                    w_val = inner_strassen_w[s][ii * 2 + jj]
                    if w_val != 0:
                        # The block is at (bi, bj) ? No, the block index is determined by t?
                        # No, the block multiplication t produces a 2x2 result.
                        # This result is placed in the 4x4 matrix C.
                        # But which 2x2 block of C?
                        # The Strassen algorithm for 4x4 produces C11, C12, C21, C22.
                        # C11 is the top-left 2x2 block.
                        # C12 is the top-right 2x2 block.
                        # etc.
                        # So the result of block multiplication t is NOT a single 2x2 block.
                        # Wait, M1, M2, etc. are 2x2 matrices.
                        # C11 = M1 - M4 + M6 + M7.
                        # So M1 contributes to C11.
                        # M1 is a 2x2 matrix.
                        # So the 2x2 result of M1 is added to the top-left 2x2 block of C.
                        
                        # So for block multiplication t, the resulting 2x2 matrix R is added to a specific 2x2 block of C.
                        # Which block?
                        # C11: M1, M4, M6, M7
                        # C12: M3, M4
                        # C21: M1, M2
                        # C22: M3, M2, M5, M6
                        
                        # So M1 (t=0) contributes to C11 and C21.
                        # M2 (t=1) contributes to C21 and C22.
                        # M3 (t=2) contributes to C12 and C22.
                        # M4 (t=3) contributes to C11 and C12.
                        # M5 (t=4) contributes to C22.
                        # M6 (t=5) contributes to C11 and C22.
                        # M7 (t=6) contributes to C11.
                        
                        # So for each t, we need to know which blocks of C it contributes to.
                        
                        # Let's define the mapping of t to (block_i, block_j, coeff)
                        # block_i, block_j in {0,1}
                        
                        # t=0 (M1): C11(0,0) coeff 1, C21(1,0) coeff 1
                        # t=1 (M2): C21(1,0) coeff 1, C22(1,1) coeff -1
                        # t=2 (M3): C12(0,1) coeff 1, C22(1,1) coeff 1
                        # t=3 (M4): C11(0,0) coeff -1, C12(0,1) coeff 1
                        # t=4 (M5): C22(1,1) coeff 1
                        # t=5 (M6): C11(0,0) coeff 1, C22(1,1) coeff 1
                        # t=6 (M7): C11(0,0) coeff 1
                        
                        # We need to add the inner W values to these blocks.
                        
                        pass # Placeholder

    # Re-implementing W mapping correctly
    block_contributions = {
        0: [(0, 0, 1), (1, 0, 1)],   # M1 -> C11, C21
        1: [(1, 0, 1), (1, 1, -1)],  # M2 -> C21, C22
        2: [(0, 1, 1), (1, 1, 1)],   # M3 -> C12, C22
        3: [(0, 0, -1), (0, 1, 1)],  # M4 -> C11, C12
        4: [(1, 1, 1)],              # M5 -> C22
        5: [(0, 0, 1), (1, 1, 1)],   # M6 -> C11, C22
        6: [(0, 0, 1)]               # M7 -> C11
    }

    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            # U and V are already set
            
            # W
            w_col = [0] * 16
            for (bi, bj, coeff_block) in block_contributions[t]:
                for ii in range(2):
                    for jj in range(2):
                        w_val = coeff_block * inner_strassen_w[s][ii * 2 + jj]
                        if w_val != 0:
                            gi = bi * 2 + ii
                            gj = bj * 2 + jj
                            w_col[gi * 4 + gj] += w_val
                            
            for idx in range(16):
                W[idx][col] = w_col[idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
