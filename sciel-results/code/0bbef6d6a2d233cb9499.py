"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36
    
    def verify(U, V, W):
        """Verify that the decomposition correctly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(len(U[0])))
                    if abs(val - 1) > 1e-9:
                        return False
        # Check non-support entries are 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        a = i * m + j
                        b = j * p + k
                        c = k2 * n + i
                        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(len(U[0])))
                        if abs(val) > 1e-9:
                            return False
        return True
    
    def strassen_4x4x4():
        """
        Rank-49 Strassen decomposition for 4x4 * 4x4 = 4x4.
        Uses 2-level Strassen: 7x7x7 -> but we need 49 for 4x4x4.
        Actually, standard 2-level Strassen gives rank 7^2=49 for 4x4x4.
        """
        # 2-level Strassen: split each 4x4 into 2x2 blocks
        # Level 1: 7 Strassen products for 2x2x2
        # Level 2: each 2x2x2 uses 7 products -> total 7*7=49
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen 2x2x2 coefficients
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # For 2x2x2, the 7 Strassen terms have these (i,j,k) contributions:
        # Each term r contributes to C[i,j] = sum_r U[i,r]*V[j,r]*W[c_idx,r]
        # where c_idx depends on which output block
        
        # Let's build it carefully.
        # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11
        # B is 4x4, split into 2x2 blocks: B00, B01, B10, B11
        # C = A*B, split into 2x2 blocks: C00, C01, C10, C11
        #
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M7
        
        # Level 2: each 2x2x2 block multiplication uses 7 Strassen terms
        # Total: 7 (level 1) * 7 (level 2) = 49 terms
        
        # Strassen 2x2x2 factor matrices (for 2x2*2x2=2x2):
        # U is 4x7 (indices: i*2+j for row, r for column)
        # V is 4x7 (indices: j*2+k for row, r for column)  
        # W is 4x7 (indices: k*2+i for row, r for column)
        
        # Standard Strassen 2x2x2:
        # Term 0: (A00+A11)(B00+B11) -> contributes to C00(+), C11(+)
        # Term 1: (A10+A11)B00 -> contributes to C10(+)
        # Term 2: A00(B10-B11) -> contributes to C01(+)
        # Term 3: A11(B01-B00) -> contributes to C00(-), C11(-)
        # Term 4: (A00+A01)B11 -> contributes to C01(+), C11(+)
        # Term 5: (A10-A00)(B00+B10) -> contributes to C00(+)
        # Term 6: (A01-A11)(B01+B11) -> contributes to C00(-), C11(+)
        
        # Build 2x2x2 Strassen factors:
        # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # Term 0: (A00+A11)(B00+B11)
        # U: A00 -> (0,0), A11 -> (1,1)
        U2[0, 0] = 1  # A[0,0]
        U2[3, 0] = 1  # A[1,1]
        # V: B00 -> (0,0), B11 -> (1,1)
        V2[0, 0] = 1  # B[0,0]
        V2[3, 0] = 1  # B[1,1]
        # W: C00 -> (0,0), C11 -> (1,1)
        W2[0, 0] = 1  # C[0,0]
        W2[3, 0] = 1  # C[1,1]
        
        # Term 1: (A10+A11)B00
        U2[2, 1] = 1  # A[1,0]
        U2[3, 1] = 1  # A[1,1]
        V2[0, 1] = 1  # B[0,0]
        W2[2, 1] = 1  # C[1,0]
        
        # Term 2: A00(B10-B11)
        U2[0, 2] = 1  # A[0,0]
        V2[2, 2] = 1  # B[1,0]
        V2[3, 2] = -1  # B[1,1]
        W2[1, 2] = 1  # C[0,1]
        
        # Term 3: A11(B01-B00)
        U2[3, 3] = 1  # A[1,1]
        V2[1, 3] = 1  # B[0,1]
        V2[0, 3] = -1  # B[0,0]
        W2[0, 3] = -1  # C[0,0]
        W2[3, 3] = -1  # C[1,1]
        
        # Term 4: (A00+A01)B11
        U2[0, 4] = 1  # A[0,0]
        U2[1, 4] = 1  # A[0,1]
        V2[3, 4] = 1  # B[1,1]
        W2[1, 4] = 1  # C[0,1]
        W2[3, 4] = 1  # C[1,1]
        
        # Term 5: (A10-A00)(B00+B10)
        U2[2, 5] = 1  # A[1,0]
        U2[0, 5] = -1  # A[0,0]
        V2[0, 5] = 1  # B[0,0]
        V2[2, 5] = 1  # B[1,0]
        W2[0, 5] = 1  # C[0,0]
        
        # Term 6: (A01-A11)(B01+B11)
        U2[1, 6] = 1  # A[0,1]
        U2[3, 6] = -1  # A[1,1]
        V2[1, 6] = 1  # B[0,1]
        V2[3, 6] = 1  # B[1,1]
        W2[0, 6] = -1  # C[0,0]
        W2[3, 6] = 1  # C[1,1]
        
        # Now build 4x4x4 from 2x2x2 using Strassen at the block level
        # A (4x4) split into 2x2 blocks:
        # A00 = A[0:2,0:2], A01 = A[0:2,2:4]
        # A10 = A[2:4,0:2], A11 = A[2:4,2:4]
        # Similarly for B and C.
        
        # Level 1 Strassen (7 block-level terms), each using level 2 (7 sub-terms)
        # Total 49 terms.
        
        # Block-level Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M7
        
        # For each level-1 term, we need to compute a 2x2x2 product
        # where the "A" and "B" inputs are linear combinations of blocks.
        
        # Let's define the level-1 structure:
        # For term t (0-6), we have:
        #   A_input = sum of certain A blocks with coefficients
        #   B_input = sum of certain B blocks with coefficients
        #   C_output = sum of certain C blocks with coefficients
        
        # Level 1 Strassen terms (block level):
        # t=0: A_in = A00+A11, B_in = B00+B11, C_out = C00(+1) + C11(+1)
        # t=1: A_in = A10+A11, B_in = B00,       C_out = C10(+1)
        # t=2: A_in = A00,        B_in = B10-B11, C_out = C01(+1)
        # t=3: A_in = A11,        B_in = B01-B00, C_out = C00(-1) + C11(-1)
        # t=4: A_in = A00+A01,   B_in = B11,      C_out = C01(+1) + C11(+1)
        # t=5: A_in = A10-A00,   B_in = B00+B10,  C_out = C00(+1)
        # t=6: A_in = A01-A11,   B_in = B01+B11,  C_out = C00(-1) + C11(+1)
        
        # For each level-1 term t, we use the 7 level-2 Strassen terms.
        # The level-2 factors U2, V2, W2 are applied within each 2x2 block.
        
        # A block A00 corresponds to A indices [0,1] x [0,1] -> flat indices 0,1,4,5
        # A block A01 corresponds to A indices [0,1] x [2,3] -> flat indices 2,3,6,7
        # A block A10 corresponds to A indices [2,3] x [0,1] -> flat indices 8,9,12,13
        # A block A11 corresponds to A indices [2,3] x [2,3] -> flat indices 10,11,14,15
        
        # Similarly for B and C.
        
        # Block to flat index mapping:
        # A block (bi, bj) where bi,bj in {0,1}:
        #   flat rows: bi*2 + [0,1], flat cols: bj*2 + [0,1]
        #   flat indices: (bi*2+0)*4 + (bj*2+0), (bi*2+0)*4 + (bj*2+1),
        #                  (bi*2+1)*4 + (bj*2+0), (bi*2+1)*4 + (bj*2+1)
        
        def block_flat_indices(block_row, block_col, dim=4):
            """Return flat indices for a 2x2 block in a 4x4 matrix."""
            rows = [block_row * 2 + r for r in range(2)]
            cols = [block_col * 2 + c for c in range(2)]
            return [r * dim + c for r in rows for c in cols]
        
        # Level 1 term definitions:
        # (a_blocks, b_blocks, c_blocks) where each is list of (block, coeff)
        level1_terms = [
            # t=0: A00+A11, B00+B11, C00+C11
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # t=1: A10+A11, B00, C10
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
            # t=2: A00, B10-B11, C01
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1)]),
            # t=3: A11, B01-B00, C00-C11
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,-1),(1,1,-1)]),
            # t=4: A00+A01, B11, C01+C11
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),
            # t=5: A10-A00, B00+B10, C00
            ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(0,0,1)]),
            # t=6: A01-A11, B01+B11, C00-C11
            ([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
        ]
        
        for t in range(7):
            a_blocks, b_blocks, c_blocks = level1_terms[t]
            for s in range(7):
                r = t * 7 + s  # rank index
                
                # For level-2 term s, U2 has non-zero entries at certain (i,j) positions
                # within the 2x2 block. We need to map these to the actual 4x4 indices
                # based on which blocks are involved.
                
                # U2[s] tells us which (local_i, local_j) in the 2x2 A-block to use
                # with what coefficient. We sum over all A-blocks in a_blocks.
                for (a_br, a_bc, a_coeff) in a_blocks:
                    # For each non-zero in U2[:, s]
                    for local_i in range(2):
                        for local_j in range(2):
                            u_val = U2[local_i * 2 + local_j, s]
                            if u_val != 0:
                                flat_idx = (a_br * 2 + local_i) * 4 + (a_bc * 2 + local_j)
                                U[flat_idx, r] += a_coeff * u_val
                
                for (b_br, b_bc, b_coeff) in b_blocks:
                    for local_i in range(2):
                        for local_j in range(2):
                            v_val = V2[local_i * 2 + local_j, s]
                            if v_val != 0:
                                flat_idx = (b_br * 2 + local_i) * 4 + (b_bc * 2 + local_j)
                                V[flat_idx, r] += b_coeff * v_val
                
                for (c_br, c_bc, c_coeff) in c_blocks:
                    for local_i in range(2):
                        for local_j in range(2):
                            w_val = W2[local_i * 2 + local_j, s]
                            if w_val != 0:
                                flat_idx = (c_br * 2 + local_i) * 4 + (c_bc * 2 + local_j)
                                W[flat_idx, r] += c_coeff * w_val
        
        return U, V, W, 49
    
    def schoolbook_4x4x1():
        """
        Schoolbook decomposition for 4x4 * 4x1 = 4x1.
        Rank = 4*4*1 = 16.
        """
        U = np.zeros((16, 16))
        V = np.zeros((4, 16))
        W = np.zeros((4, 16))
        
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                a = i * 4 + j  # A[i,j]
                b = j * 1 + 0  # B[j,0]
                c = 0 * 4 + i  # C[0,i]
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
        
        return U, V, W, 16
    
    def combine_blocks():
        """
        Combine two 4x4x4 Strassen decompositions and one 4x4x1 schoolbook
        for the full 4x4x9 problem.
        
        B is 4x9, split into B_left (4x4) and B_right (4x5).
        Wait, 9 = 4 + 5, but let's do 4 + 4 + 1.
        
        Actually: C = A * B where A is 4x4, B is 4x9, C is 4x9.
        Split B into B1 (4x4), B2 (4x4), B3 (4x1).
        C = A*B1 + A*B2 + A*B3.
        C1 = A*B1 (4x4), C2 = A*B2 (4x4), C3 = A*B3 (4x1).
        
        Total rank = 49 + 49 + 16 = 114.
        """
        U1, V1, W1, r1 = strassen_4x4x4()
        U2, V2, W2, r2 = strassen_4x4x4()
        U3, V3, W3, r3 = schoolbook_4x4x1()
        
        total_rank = r1 + r2 + r3
        U = np.zeros((16, total_rank))
        V = np.zeros((36, total_rank))
        W = np.zeros((36, total_rank))
        
        # Block 1: B columns 0-3, C columns 0-3
        U[:, 0:r1] = U1
        # V: B1 is 4x4, rows 0-3 of B (which is 4x9, so B[j,0..3])
        # V1 is 16x49, V1[j*4+k, r] for B[j,k] where k in 0..3
        # In the full problem, B[j, k] for k in 0..3 -> index j*9+k
        for j in range(4):
            for k in range(4):
                V[j * 9 + k, 0:r1] = V1[j * 4 + k, 0:r1]
        # W: C1 is 4x4, W1[i*4+j, r] for C[i,j] where i in 0..3
        # In the full problem, C[i, k] for k in 0..3 -> index k*4+i
        for i in range(4):
            for j in range(4):
                W[j * 4 + i, 0:r1] = W1[i * 4 + j, 0:r1]
        
        # Block 2: B columns 4-7, C columns 4-7
        U[:, r1:r1+r2] = U2
        for j in range(4):
            for k in range(4):
                V[j * 9 + (k + 4), r1:r1+r2] = V2[j * 4 + k, 0:r2]
        for i in range(4):
            for j in range(4):
                W[(j + 4) * 4 + i, r1:r1+r2] = W2[i * 4 + j, 0:r2]
        
        # Block 3: B column 8, C column 8
        U[:, r1+r2:r1+r2+r3] = U3
        for j in range(4):
            V[j * 9 + 8, r1+r2:r1+r2+r3] = V3[j, 0:r3]
        for i in range(4):
            W[8 * 4 + i, r1+r2:r1+r2+r3] = W3[i, 0:r3]
        
        return U, V, W, total_rank
    
    # Build the combined decomposition
    U, V, W, rank = combine_blocks()
    
    # Verify
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if not verify(U_list, V_list, W_list):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((nm, rank))
        V_fb = np.zeros((mp, rank))
        W_fb = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, idx] = 1
                    V_fb[b, idx] = 1
                    W_fb[c, idx] = 1
        U_list = U_fb.tolist()
        V_list = V_fb.tolist()
        W_list = W_fb.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
# EVOLVE-BLOCK-END
