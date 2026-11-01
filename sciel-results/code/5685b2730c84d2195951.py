"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get block A[I, J] as a 2x2 list of linear forms in A
    # A[I, J] corresponds to rows 2I..2I+1, cols 2J..2J+1
    # We represent linear forms as dicts { (i, j): coeff }
    
    def get_block_A(I, J):
        block = {}
        for r in range(2):
            for c in range(2):
                i = 2 * I + r
                j = 2 * J + c
                block[(i, j)] = 1
        return block

    def get_block_B(I, J):
        block = {}
        for r in range(2):
            for c in range(2):
                j = 2 * I + r
                k = 2 * J + c
                block[(j, k)] = 1
        return block

    def add_blocks(b1, b2, sign1=1, sign2=1):
        res = {}
        for k, v in b1.items():
            res[k] = res.get(k, 0) + sign1 * v
        for k, v in b2.items():
            res[k] = res.get(k, 0) + sign2 * v
        return res

    def sub_blocks(b1, b2):
        return add_blocks(b1, b2, 1, -1)

    # Super-products q=0..6
    # Each defined by Left (2x2 matrix of blocks) and Right (2x2 matrix of blocks)
    # We store Left as a list of 4 dicts (row-major 2x2)
    # Same for Right
    
    A00 = get_block_A(0, 0)
    A01 = get_block_A(0, 1)
    A10 = get_block_A(1, 0)
    A11 = get_block_A(1, 1)
    
    B00 = get_block_B(0, 0)
    B01 = get_block_B(0, 1)
    B10 = get_block_B(1, 0)
    B11 = get_block_B(1, 1)
    
    super_L = []
    super_R = []
    
    # q=0: (A00+A11)(B00+B11)
    L0 = [add_blocks(A00, A11), add_blocks(A00, A11), add_blocks(A00, A11), add_blocks(A00, A11)] # Wait, A00+A11 is a 2x2 block sum.
    # A00 is 2x2. A11 is 2x2. Sum is 2x2.
    # My get_block returns dict of elements.
    # So L0[0] is the (0,0) element of the 2x2 matrix L0?
    # No, L0 is a 2x2 matrix. Each entry is a 2x2 block sum?
    # Strassen: M0 = (A11+A22)(B11+B22).
    # A11 is the top-left block.
    # So the "Left" operand is a single 2x2 matrix which is A00+A11.
    # So L0 is a 2x2 matrix where every entry is the same block sum? No.
    # L0 is a 2x2 matrix.
    # L0[0][0] is the element at (0,0) of the matrix (A00+A11).
    # A00+A11 is a 4x4 sparse matrix? No, it's a 2x2 matrix of blocks?
    # No, in the context of the recursive step, A00 is a 2x2 matrix.
    # So A00+A11 is a 2x2 matrix.
    # So L0 is a 2x2 matrix.
    # L0[0][0] = (A00+A11)[0][0]
    # L0[0][1] = (A00+A11)[0][1]
    # etc.
    
    # Let's re-define the linear forms for the 2x2 matrix operands.
    # For q=0:
    # Left operand X is 2x2 matrix X[i][j] = A[i][j] + A[i+2][j+2]
    # Right operand Y is 2x2 matrix Y[i][j] = B[i][j] + B[i+2][j+2]
    
    def make_sum_block_A(I, J, K, L):
        # Returns 2x2 list of dicts representing A[I,J] + A[K,L]
        res = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                res[r][c] = add_blocks(get_block_A(I, J), get_block_A(K, L))
        return res

    def make_single_block_A(I, J):
        res = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                res[r][c] = get_block_A(I, J)
        return res

    # This approach is flawed. A00 is a block.
    # In the super-product M0 = (A00+A11)(B00+B11), the operand (A00+A11) is a 2x2 matrix.
    # Its (0,0) entry is A00[0,0] + A11[0,0].
    # Its (0,1) entry is A00[0,1] + A11[0,1].
    
    # Let's just compute the 2x2 matrices X and Y for each q directly.
    
    X_list = []
    Y_list = []
    
    # q=0
    X0 = [[add_blocks(A00, A11), add_blocks(A00, A11), add_blocks(A00, A11), add_blocks(A00, A11)] for _ in range(2)] 
    # Wait, A00+A11 is the same for all 4 positions?
    # A00 is 2x2. A11 is 2x2.
    # (A00+A11) is a 2x2 matrix.
    # So X0 is just that matrix.
    # X0[0][0] = A00[0][0] + A11[0][0]
    # My add_blocks returns a dict.
    # So X0 is a 2x2 list of dicts.
    
    X0 = [
        [add_blocks(A00, A11), add_blocks(A00, A11)],
        [add_blocks(A00, A11), add_blocks(A00, A11)]
    ]
    # This is wrong. A00 and A11 are 2x2 blocks.
    # A00[0][0] is a scalar.
    # So X0[0][0] should be A00[0][0] + A11[0][0].
    # My get_block_A returns a dict of the whole 2x2 block.
    # So I need to extract individual elements.
    
    # Let's restart the representation.
    # Represent linear form as dict { (i, j): coeff }
    # X is 2x2 list of such dicts.
    
    def block_element_A(I, J, r, c):
        d = {}
        i = 2*I + r
        j = 2*J + c
        d[(i,j)] = 1
        return d

    def block_element_B(I, J, r, c):
        d = {}
        j = 2*I + r
        k = 2*J + c
        d[(j,k)] = 1
        return d

    def sum_blocks_A(I1, J1, I2, J2, s1=1, s2=1):
        # Returns 2x2 list of dicts
        res = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                d1 = block_element_A(I1, J1, r, c)
                d2 = block_element_A(I2, J2, r, c)
                res[r][c] = add_blocks(d1, d2, s1, s2)
        return res

    def single_block_A(I, J, s=1):
        res = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                d = block_element_A(I, J, r, c)
                if s != 1:
                    for k in d: d[k] *= s
                res[r][c] = d
        return res

    # q=0: X = A00+A11, Y = B00+B11
    X_list.append(sum_blocks_A(0,0,1,1))
    Y_list.append(sum_blocks_A(0,0,1,1)) # Note: B blocks use same geometry
    
    # q=1: X = A10+A11, Y = B00
    X_list.append(sum_blocks_A(1,0,1,1))
    Y_list.append(single_block_A(0,0))
    
    # q=2: X = A00, Y = B01-B11
    X_list.append(single_block_A(0,0))
    Y_list.append(sum_blocks_A(0,1,1,1,1,-1))
    
    # q=3: X = A11, Y = B10-B00
    X_list.append(single_block_A(1,1))
    Y_list.append(sum_blocks_A(1,0,0,0,1,-1))
    
    # q=4: X = A00+A01, Y = B11
    X_list.append(sum_blocks_A(0,0,0,1))
    Y_list.append(single_block_A(1,1))
    
    # q=5: X = A10-A00, Y = B00+B01
    X_list.append(sum_blocks_A(1,0,0,0,1,-1))
    Y_list.append(sum_blocks_A(0,0,0,1))
    
    # q=6: X = A01-A11, Y = B10+B11
    X_list.append(sum_blocks_A(0,1,1,1,1,-1))
    Y_list.append(sum_blocks_A(1,0,1,1))
    
    # Base Strassen Coeffs
    # m0: (X00+X11)(Y00+Y11)
    # m1: (X10+X11)Y00
    # m2: X00(Y01-Y11)
    # m3: X11(Y10-Y00)
    # m4: (X00+X01)Y11
    # m5: (X10-X00)(Y00+Y01)
    # m6: (X01-X11)(Y10+Y11)
    
    base_L_combos = [
        [(0,0), (1,1)], # indices into 2x2 X
        [(1,0), (1,1)],
        [(0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (0,0)],
        [(0,1), (1,1)]
    ]
    base_L_signs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1]
    ]
    base_R_combos = [
        [(0,0), (1,1)],
        [(0,0)],
        [(0,1), (1,1)],
        [(1,0), (0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (1,1)]
    ]
    base_R_signs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1]
    ]
    
    # Reconstruction coeffs for Z from m
    # Z00 = m0 + m3 - m4 + m6
    # Z01 = m2 + m4
    # Z10 = m1 + m3
    # Z11 = m0 - m1 + m2 + m5
    rec = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # Z00
        {2: 1, 4: 1},               # Z01
        {1: 1, 3: 1},               # Z10
        {0: 1, 1: -1, 2: 1, 5: 1}   # Z11
    ]
    
    # Super rec coeffs
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    super_rec = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11
    ]
    
    for q in range(7):
        X = X_list[q]
        Y = Y_list[q]
        for s in range(7):
            c_idx = q * 7 + s
            
            # Construct L_A
            lA = {}
            for idx, sign in zip(base_L_combos[s], base_L_signs[s]):
                r, c = idx
                block = X[r][c]
                for k, v in block.items():
                    lA[k] = lA.get(k, 0) + sign * v
            
            # Construct L_B
            lB = {}
            for idx, sign in zip(base_R_combos[s], base_R_signs[s]):
                r, c = idx
                block = Y[r][c]
                for k, v in block.items():
                    lB[k] = lB.get(k, 0) + sign * v
            
            # Fill U
            for (i, j), coeff in lA.items():
                if coeff != 0:
                    U[i*m + j][c_idx] = coeff
                    
            # Fill V
            for (j, k), coeff in lB.items():
                if coeff != 0:
                    V[j*p + k][c_idx] = coeff
            
            # Fill W
            # m_s contributes to Z_r,c' with rec[r][s]
            for r in range(2):
                alpha = rec[r].get(s, 0)
                if alpha == 0: continue
                for c_prime in range(2):
                    # Z is 2x2. rec is indexed by 0..3 for Z00, Z01, Z10, Z11
                    # My rec list is [Z00, Z01, Z10, Z11]
                    # So r=0, c_prime=0 -> rec[0]
                    # r=0, c_prime=1 -> rec[1]
                    # r=1, c_prime=0 -> rec[2]
                    # r=1, c_prime=1 -> rec[3]
                    z_idx = r * 2 + c_prime
                    contrib = rec[z_idx].get(s, 0)
                    if contrib == 0: continue
                    
                    # M_q contributes to C_IK with super_rec[I,K][q]
                    for I in range(2):
                        for K in range(2):
                            super_contrib = super_rec[I*2+K].get(q, 0)
                            if super_contrib == 0: continue
                            
                            global_i = 2 * I + r
                            global_k = 2 * K + c_prime
                            
                            w_coeff = super_contrib * contrib
                            if w_coeff != 0:
                                W[global_k*n + global_i][c_idx] = w_coeff
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
