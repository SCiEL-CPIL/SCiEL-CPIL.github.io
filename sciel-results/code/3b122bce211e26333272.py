import time

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using 2-level Strassen's algorithm.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # U: (n*m, rank) -> (16, 49)
    # V: (m*p, rank) -> (16, 49)
    # W: (p*n, rank) -> (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to represent a linear form as a dict { (i, j): coeff }
    # For A, i is row, j is col.
    # For B, i is row, j is col.
    
    def add_forms(f1, f2, sign2=1):
        res = dict(f1)
        for k, v in f2.items():
            res[k] = res.get(k, 0) + sign2 * v
        return res
    
    def block_form(br, bc, sign=1):
        """
        Create linear form for the 2x2 block at (br, bc).
        br, bc in {0, 1}.
        Returns dict { (i, j): coeff }
        """
        d = {}
        for i in range(2):
            for j in range(2):
                global_i = 2 * br + i
                global_j = 2 * bc + j
                d[(global_i, global_j)] = sign
        return d
    
    # Define the 7 block products
    # Each product k has:
    # X_k: 2x2 matrix of linear forms in A
    # Y_k: 2x2 matrix of linear forms in B
    # C_map: dict { (block_r, block_c): coeff } for contribution to C
    
    products = []
    
    # k=0: M1 = (A00 + A11)(B00 + B11)
    X0 = [[block_form(0,0) | block_form(1,1) for _ in range(2)] for _ in range(2)]
    # Correctly: X0[r][c] should be the linear form for entry (r,c) of the block sum.
    # A00 is block (0,0). A11 is block (1,1).
    # X0[0][0] = A[0,0] + A[2,2]
    # X0[0][1] = A[0,1] + A[2,3]
    # etc.
    def make_sum_block(br1, bc1, br2, bc2):
        """Returns 2x2 list of linear forms for A_br1_bc1 + A_br2_bc2"""
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                f1 = block_form(br1, bc1)
                f2 = block_form(br2, bc2)
                # We need to extract the specific entry (r,c) from the block forms?
                # No, block_form returns the whole 2x2 block as a dict of global indices.
                # But X_k is a 2x2 matrix of linear forms.
                # Each entry X_k[r][c] is a linear form.
                # The linear form for X_k[r][c] is the sum of the corresponding entries from the blocks.
                # Entry (r,c) of block (br,bc) is at global (2*br+r, 2*bc+c).
                # So f1 has key (2*br1+r, 2*bc1+c) with value 1.
                # f2 has key (2*br2+r, 2*bc2+c) with value 1.
                # So the linear form is { (2*br1+r, 2*bc1+c): 1, (2*br2+r, 2*bc2+c): 1 }
                form = {}
                form[(2*br1+r, 2*bc1+c)] = 1
                form[(2*br2+r, 2*bc2+c)] = 1
                row.append(form)
            res.append(row)
        return res

    def make_single_block(br, bc):
        """Returns 2x2 list of linear forms for A_br_bc"""
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                form = { (2*br+r, 2*bc+c): 1 }
                row.append(form)
            res.append(row)
        return res

    def make_diff_block(br1, bc1, br2, bc2, sign2=1):
        """Returns 2x2 list of linear forms for A_br1_bc1 + sign2 * A_br2_bc2"""
        res = []
        for r in range(2):
            row = []
            for c in range(2):
                form = {}
                form[(2*br1+r, 2*bc1+c)] = 1
                form[(2*br2+r, 2*bc2+c)] = sign2
                row.append(form)
            res.append(row)
        return res

    # Re-define products using these helpers
    # k=0
    X = make_sum_block(0,0, 1,1)
    Y = make_sum_block(0,0, 1,1)
    C_map = { (0,0): 1, (1,1): 1 }
    products.append((X, Y, C_map))
    
    # k=1: M2 = (A10 + A11)B00
    X = make_sum_block(1,0, 1,1)
    Y = make_single_block(0,0)
    C_map = { (1,0): 1, (1,1): -1 }
    products.append((X, Y, C_map))
    
    # k=2: M3 = A00(B01 - B11)
    X = make_single_block(0,0)
    Y = make_diff_block(0,1, 1,1, -1)
    C_map = { (0,1): 1, (1,1): 1 }
    products.append((X, Y, C_map))
    
    # k=3: M4 = A11(B10 - B00)
    X = make_single_block(1,1)
    Y = make_diff_block(1,0, 0,0, -1)
    C_map = { (0,0): 1, (1,0): 1 }
    products.append((X, Y, C_map))
    
    # k=4: M5 = (A00 + A01)B11
    X = make_sum_block(0,0, 0,1)
    Y = make_single_block(1,1)
    C_map = { (0,0): -1, (0,1): 1 }
    products.append((X, Y, C_map))
    
    # k=5: M6 = (A10 - A00)(B00 + B01)
    X = make_diff_block(1,0, 0,0, -1)
    Y = make_sum_block(0,0, 0,1)
    C_map = { (1,1): 1 }
    products.append((X, Y, C_map))
    
    # k=6: M7 = (A01 - A11)(B10 + B11)
    X = make_diff_block(0,1, 1,1, -1)
    Y = make_sum_block(1,0, 1,1)
    C_map = { (0,0): 1 }
    products.append((X, Y, C_map))
    
    # Strassen step coefficients for M_k entries
    # m00 = z0 + z3 - z4 + z6
    # m01 = z2 + z4
    # m10 = z1 + z3
    # m11 = z0 - z1 + z2 + z5
    
    strassen_D = [
        [[1, 0], [0, 1]], # z0
        [[0, 0], [1, -1]], # z1
        [[0, 1], [0, 1]], # z2
        [[1, 0], [1, 0]], # z3
        [[-1, 1], [0, 0]], # z4
        [[0, 0], [0, 1]], # z5
        [[1, 0], [0, 0]]  # z6
    ]
    
    # Strassen linear forms for z_s
    # z0 = (x00+x11)(y00+y11)
    # z1 = (x10+x11)y00
    # z2 = x00(y01-y11)
    # z3 = x11(y10-y00)
    # z4 = (x00+x01)y11
    # z5 = (x10-x00)(y00+y01)
    # z6 = (x01-x11)(y10+y11)
    
    # We need to combine the linear forms from X and Y matrices.
    # X is 2x2 list of forms. Y is 2x2 list of forms.
    
    def combine_forms(f1, f2, sign2=1):
        res = dict(f1)
        for k, v in f2.items():
            res[k] = res.get(k, 0) + sign2 * v
        return res

    for k in range(7):
        X, Y, C_map = products[k]
        for s in range(7):
            r = 7 * k + s
            
            # Determine u_r and v_r
            if s == 0:
                u = combine_forms(X[0][0], X[1][1])
                v = combine_forms(Y[0][0], Y[1][1])
            elif s == 1:
                u = combine_forms(X[1][0], X[1][1])
                v = Y[0][0]
            elif s == 2:
                u = X[0][0]
                v = combine_forms(Y[0][1], Y[1][1], -1)
            elif s == 3:
                u = X[1][1]
                v = combine_forms(Y[1][0], Y[0][0], -1)
            elif s == 4:
                u = combine_forms(X[0][0], X[0][1])
                v = Y[1][1]
            elif s == 5:
                u = combine_forms(X[1][0], X[0][0], -1)
                v = combine_forms(Y[0][0], Y[0][1])
            elif s == 6:
                u = combine_forms(X[0][1], X[1][1], -1)
                v = combine_forms(Y[1][0], Y[1][1])
            
            # Fill U and V
            for (i, j), coeff in u.items():
                U[i * m + j][r] = coeff
            
            for (i, j), coeff in v.items():
                V[i * m + j][r] = coeff # Note: V is indexed by j*p+k, which is i*4+j in our B matrix notation?
                # Wait. B is m x p. m=4, p=4.
                # V row index is j*p+k.
                # In our B matrix, row is i (0..3), col is j (0..3).
                # So B[i][j] corresponds to V[i*4+j].
                # My B linear forms are { (i, j): coeff } where i is row, j is col.
                # So V[i*4+j][r] is correct.
            
            # Fill W
            D_s = strassen_D[s]
            for (br, bc), c_coeff in C_map.items():
                for lr in range(2):
                    for lc in range(2):
                        d_coeff = D_s[lr][lc]
                        if d_coeff == 0:
                            continue
                        total_coeff = c_coeff * d_coeff
                        if total_coeff == 0:
                            continue
                        
                        # Global C index
                        i_global = 2 * br + lr
                        k_global = 2 * bc + lc
                        
                        # W index is k*n + i
                        W[k_global * n + i_global][r] = total_coeff

    return {"U": U, "V": V, "W": W}
