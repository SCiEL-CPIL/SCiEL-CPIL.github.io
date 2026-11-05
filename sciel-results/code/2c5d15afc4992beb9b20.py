import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # --- Rank-21 decomposition for 3x3 matrix multiplication ---
    # Strategy: Split 3x3 into 2x2 block with 2x2, 2x1, 1x2, 1x1 sub-blocks.
    # A = [[A22, A21], [A12, A11]] where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1
    # B = [[B22, B21], [B12, B11]]
    # C = [[C22, C21], [C12, C11]]
    #
    # Use Strassen (rank 7) for the 2x2 blocks, and handle cross terms.
    # Actually, let's use a different approach: direct rank-21 construction.
    #
    # Known: rank(3x3) <= 23 (Strassen bound), and there exist rank-21 decompositions.
    # Let me construct one using the "Strassen-like" approach for 3x3.
    #
    # Alternative: Use the fact that 3x3 can be seen as 2x2 + 1.
    # A = [[A', a], [b', c]] where A' is 2x2, a is 2x1, b' is 1x2, c is scalar.
    # B = [[B', b], [d', e]] where B' is 2x2, b is 2x1, d' is 1x2, e is scalar.
    # C = [[C', c1], [d1, f]]
    #
    # C' = A'B' + a*d'
    # c1 = A'b + a*e
    # d1 = b'B' + c*d'
    # f = b' * b + c*e
    #
    # We need to compute: A'B' (rank 7 via Strassen), a*d' (rank 2), 
    # A'b (rank 4? no, it's a 2x1 times... wait)
    #
    # Actually, let me think more carefully. The multiplication tensor for 3x3
    # has 27 components. We want to find a rank-21 decomposition.
    #
    # Let me use a known construction. One approach:
    # View 3x3 as block matrix [[A11, A12], [A21, A22]] with 2x2, 2x1, 1x2, 1x1 blocks.
    # 
    # For the 2x2 block product A11*B11, use Strassen (7 terms).
    # For cross terms, we need additional terms.
    #
    # Total: 7 (for A11*B11) + terms for A11*B12, A12*B11, A12*B21, A21*B11, A21*B22, A22*B11, etc.
    # This gets complex. Let me use a verified rank-21 construction.
    #
    # Actually, the simplest reliable approach: use rank-23 (which is well-known) 
    # or stick with schoolbook 27. But 7*23=161 < 7*27=189.
    #
    # Let me construct a rank-23 3x3 decomposition.
    # From the literature, one construction uses the 2x2 Strassen embedded in 3x3.
    #
    # Here's a concrete rank-23 construction:
    # Split 3x3 into 2x2 + 1. The 2x2 part uses Strassen (7 terms).
    # The remaining 10 terms handle the cross products and the 1x1 part.
    # 7 + 16 = 23? Let me count more carefully.
    #
    # Actually, let me just use a verified construction. I'll hardcode a known 
    # rank-23 decomposition for 3x3 matrix multiplication.
    #
    # For now, let me use the safe approach: rank-27 schoolbook for 3x3,
    # giving 7*27 = 189 total rank. This is already an improvement from 216.
    #
    # But can I do better? Let me try to construct a rank-21 3x3 decomposition.
    #
    # Known result: The rank of 3x3 matrix multiplication is at most 23.
    # A rank-21 decomposition would be even better.
    #
    # Let me try a different strategy: use the "border rank" approach or 
    # a specific algebraic construction.
    #
    # For reliability, I'll implement the rank-23 3x3 decomposition.
    # Here's one construction from the literature:
    
    def get_3x3_rank23():
        """
        Rank-23 decomposition for 3x3 matrix multiplication.
        Uses the structure: split into 2x2 blocks and handle cross terms.
        
        A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
        B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
        C = A * B
        
        We use 23 terms. The construction is based on:
        - 7 terms for the 2x2 sub-block (top-left 2x2)
        - Additional terms for the remaining elements
        """
        n3 = 3
        rank3 = 23
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        
        # First 7 terms: Strassen on the top-left 2x2 block
        # Blocks: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3 in 3x3
        # A blocks: A[0:2,0:2] split as [[A00,A01],[A10,A11]]
        # B blocks: B[0:2,0:2] split as [[B00,B01],[B10,B11]]
        # C blocks: C[0:2,0:2] split as [[C00,C01],[C10,C11]]
        
        # Strassen P1 = (A00+A11)*(B00+B11)
        # In 3x3: A00=A[0,0], A11=A[1,1], B00=B[0,0], B11=B[1,1]
        # C00=C[0,0], C11=C[1,1]
        U3[0, 0] = 1; U3[3, 0] = 1  # A[0,0], A[1,1]
        V3[0, 0] = 1; V3[3, 0] = 1  # B[0,0], B[1,1]
        W3[0, 0] = 1; W3[3, 0] = 1  # C[0,0], C[1,1]
        
        # Strassen P2 = (A10+A11)*B00
        U3[2, 1] = 1; U3[3, 1] = 1  # A[1,0], A[1,1]
        V3[0, 1] = 1                # B[0,0]
        W3[2, 1] = 1; W3[3, 1] = -1 # C[1,0], C[1,1]
        
        # Strassen P3 = A00*(B01-B11)
        U3[0, 2] = 1                # A[0,0]
        V3[1, 2] = 1; V3[3, 2] = -1 # B[0,1], B[1,1]
        W3[1, 2] = 1; W3[3, 2] = 1  # C[0,1], C[1,1]
        
        # Strassen P4 = A11*(B10-B00)
        U3[3, 3] = 1                # A[1,1]
        V3[2, 3] = 1; V3[0, 3] = -1 # B[1,0], B[0,0]
        W3[0, 3] = 1; W3[2, 3] = 1  # C[0,0], C[1,0]
        
        # Strassen P5 = (A00+A01)*B11
        U3[0, 4] = 1; U3[1, 4] = 1  # A[0,0], A[0,1]
        V3[3, 4] = 1                # B[1,1]
        W3[1, 4] = 1; W3[0, 4] = -1 # C[0,1], C[0,0]
        
        # Strassen P6 = (A10-A00)*(B00+B01)
        U3[2, 5] = 1; U3[0, 5] = -1 # A[1,0], A[0,0]
        V3[0, 5] = 1; V3[1, 5] = 1  # B[0,0], B[0,1]
        W3[3, 5] = 1                # C[1,1]
        
        # Strassen P7 = (A01-A11)*(B10+B11)
        U3[1, 6] = 1; U3[3, 6] = -1 # A[0,1], A[1,1]
        V3[2, 6] = 1; V3[3, 6] = 1  # B[1,0], B[1,1]
        W3[0, 6] = 1                # C[0,0]
        
        # Now we need to handle the 3rd row and column.
        # Remaining C elements: C[0,2], C[1,2], C[2,0], C[2,1], C[2,2]
        # These involve A[*,2], A[2,*], B[*,2], B[2,*]
        
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
        # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
        
        # That's 15 terms if done schoolbook. We need to reduce to 16 total additional.
        # Total should be 23, so 23 - 7 = 16 more terms.
        
        # Let me use a smarter approach. I'll add 16 more terms.
        
        # Term 7: A[0,2]*B[2,2] -> C[0,2]
        U3[2, 7] = 1  # A[0,2]
        V3[8, 7] = 1  # B[2,2]
        W3[6, 7] = 1  # C[0,2] (k=2, i=0 -> 2*3+0=6)
        
        # Term 8: A[1,2]*B[2,2] -> C[1,2]
        U3[5, 8] = 1  # A[1,2]
        V3[8, 8] = 1  # B[2,2]
        W3[7, 8] = 1  # C[1,2] (k=2, i=1 -> 2*3+1=7)
        
        # Term 9: A[2,2]*B[2,2] -> C[2,2]
        U3[8, 9] = 1  # A[2,2]
        V3[8, 9] = 1  # B[2,2]
        W3[8, 9] = 1  # C[2,2] (k=2, i=2 -> 2*3+2=8)
        
        # Term 10: A[0,0]*B[0,2] -> C[0,2]
        U3[0, 10] = 1  # A[0,0]
        V3[2, 10] = 1  # B[0,2]
        W3[6, 10] = 1  # C[0,2]
        
        # Term 11: A[0,1]*B[1,2] -> C[0,2]
        U3[1, 11] = 1  # A[0,1]
        V3[5, 11] = 1  # B[1,2]
        W3[6, 11] = 1  # C[0,2]
        
        # Term 12: A[1,0]*B[0,2] -> C[1,2]
        U3[3, 12] = 1  # A[1,0]
        V3[2, 12] = 1  # B[0,2]
        W3[7, 12] = 1  # C[1,2]
        
        # Term 13: A[1,1]*B[1,2] -> C[1,2]
        U3[4, 13] = 1  # A[1,1]
        V3[5, 13] = 1  # B[1,2]
        W3[7, 13] = 1  # C[1,2]
        
        # Term 14: A[2,0]*B[0,0] -> C[2,0]
        U3[6, 14] = 1  # A[2,0]
        V3[0, 14] = 1  # B[0,0]
        W3[2, 14] = 1  # C[2,0] (k=0, i=2 -> 0*3+2=2)
        
        # Term 15: A[2,1]*B[1,0] -> C[2,0]
        U3[7, 15] = 1  # A[2,1]
        V3[3, 15] = 1  # B[1,0]
        W3[2, 15] = 1  # C[2,0]
        
        # Term 16: A[2,2]*B[2,0] -> C[2,0]
        U3[8, 16] = 1  # A[2,2]
        V3[6, 16] = 1  # B[2,0]
        W3[2, 16] = 1  # C[2,0]
        
        # Term 17: A[2,0]*B[0,1] -> C[2,1]
        U3[6, 17] = 1  # A[2,0]
        V3[1, 17] = 1  # B[0,1]
        W3[5, 17] = 1  # C[2,1] (k=1, i=2 -> 1*3+2=5)
        
        # Term 18: A[2,1]*B[1,1] -> C[2,1]
        U3[7, 18] = 1  # A[2,1]
        V3[4, 18] = 1  # B[1,1]
        W3[5, 18] = 1  # C[2,1]
        
        # Term 19: A[2,2]*B[2,1] -> C[2,1]
        U3[8, 19] = 1  # A[2,2]
        V3[7, 19] = 1  # B[2,1]
        W3[5, 19] = 1  # C[2,1]
        
        # Term 20: A[2,0]*B[0,2] -> C[2,2]
        U3[6, 20] = 1  # A[2,0]
        V3[2, 20] = 1  # B[0,2]
        W3[8, 20] = 1  # C[2,2]
        
        # Term 21: A[2,1]*B[1,2] -> C[2,2]
        U3[7, 21] = 1  # A[2,1]
        V3[5, 21] = 1  # B[1,2]
        W3[8, 21] = 1  # C[2,2]
        
        # Term 22: A[0,2]*B[2,0] -> C[0,0]? No, that's not right.
        # Wait, I need to be more careful. Let me recount.
        # 
        # Actually, this is getting messy. Let me use a cleaner approach.
        # I'll verify the decomposition and if it doesn't work, fall back to rank 27.
        
        return U3, V3, W3
    
    # Let me verify the rank-23 construction
    U3_test, V3_test, W3_test = get_3x3_rank23()
    
    def verify_3x3(U_arr, V_arr, W_arr):
        """Verify 3x3 decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == delta_{i,k}*delta_{j,i}... 
        Actually, for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        The tensor is T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
        So we need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k in 0..2
        Wait, that's not right either. The multiplication tensor has T[i,j,k] = 1 
        if the output element (i,k) gets contribution from A[i,j]*B[j,k].
        Actually, the tensor is: T[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k].
        In the decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 for all i,j,k.
        """
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(U_arr.shape[1]):
                        s += U_arr[i*3+j, r] * V_arr[j*3+k, r] * W_arr[k*3+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    valid_23, err = verify_3x3(U3_test, V3_test, W3_test)
    
    if valid_23:
        r3 = 23
        U3, V3, W3 = U3_test, V3_test, W3_test
    else:
        # Fall back to schoolbook rank 27
        r3 = 27
        U3 = np.zeros((9, r3))
        V3 = np.zeros((9, r3))
        W3 = np.zeros((9, r3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i * 9 + j * 3 + k
                    U3[i * 3 + j, col] = 1.0
                    V3[j * 3 + k, col] = 1.0
                    W3[k * 3 + i, col] = 1.0
    
    # --- Construct 6x6 decomposition using Strassen 2x2 blocks ---
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        """
        Add one Strassen term.
        a_blocks, b_blocks, c_blocks: list of (block_idx, coeff)
        Block idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        for r in range(r3):
            u_mat = U3[:, r].reshape(3, 3)
            v_mat = V3[:, r].reshape(3, 3)
            w_mat = W3[:, r].reshape(3, 3)
            
            for bi, coeff in a_blocks:
                rb, cb = bi // 2, bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            gi, gj = rb * 3 + i, cb * 3 + j
                            U6[gi * 6 + gj, start_col + r] += val
            
            for bi, coeff in b_blocks:
                rb, cb = bi // 2, bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            gj, gk = rb * 3 + j, cb * 3 + k
                            V6[gj * 6 + gk, start_col + r] += val
            
            for bi, coeff in c_blocks:
                rb, cb = bi // 2, bi % 2
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            gk, gi = cb * 3 + k, rb * 3 + i
                            W6[gk * 6 + gi, start_col + r] += val
    
    # Strassen formulas for 2x2 blocks
    # P1 = (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
    add_strassen_term(0,
                      a_blocks=[(0, 1), (3, 1)],
                      b_blocks=[(0, 1), (3, 1)],
                      c_blocks=[(0, 1), (3, 1)])
    
    # P2 = (A10+A11)*B00 -> C10(+1), C11(-1)
    add_strassen_term(r3,
                      a_blocks=[(2, 1), (3, 1)],
                      b_blocks=[(0, 1)],
                      c_blocks=[(2, 1), (3, -1)])
    
    # P3 = A00*(B01-B11) -> C01(+1), C11(+1)
    add_strassen_term(2 * r3,
                      a_blocks=[(0, 1)],
                      b_blocks=[(1, 1), (3, -1)],
                      c_blocks=[(1, 1), (3, 1)])
    
    # P4 = A11*(B10-B00) -> C00(+1), C10(+1)
    add_strassen_term(3 * r3,
                      a_blocks=[(3, 1)],
                      b_blocks=[(2, 1), (0, -1)],
                      c_blocks=[(0, 1), (2, 1)])
    
    # P5 = (A00+A01)*B11 -> C01(+1), C00(-1)
    add_strassen_term(4 * r3,
                      a_blocks=[(0, 1), (1, 1)],
                      b_blocks=[(3, 1)],
                      c_blocks=[(1, 1), (0, -1)])
    
    # P6 = (A10-A00)*(B00+B01) -> C11(+1)
    add_strassen_term(5 * r3,
                      a_blocks=[(2, 1), (0, -1)],
                      b_blocks=[(0, 1), (1, 1)],
                      c_blocks=[(3, 1)])
    
    # P7 = (A01-A11)*(B10+B11) -> C00(+1)
    add_strassen_term(6 * r3,
                      a_blocks=[(1, 1), (3, -1)],
                      b_blocks=[(2, 1), (3, 1)],
                      c_blocks=[(0, 1)])
    
    # Verify the full 6x6 decomposition
    def verify_6x6(U_arr, V_arr, W_arr):
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    s = 0.0
                    for r in range(U_arr.shape[1]):
                        s += U_arr[i * 6 + j, r] * V_arr[j * 6 + k, r] * W_arr[k * 6 + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if verify_6x6(U6, V6, W6):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    # Fallback: schoolbook rank 216
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                col = i * 36 + j * 6 + k
                U_f[i * 6 + j, col] = 1.0
                V_f[j * 6 + k, col] = 1.0
                W_f[k * 6 + i, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
