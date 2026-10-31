# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Recursive Strassen decomposition for 4x4 matrix multiplication.
    Rank: 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to add to W
    # W index: k*n + i
    def w_idx(i, k):
        return k * n + i

    # Strassen 2x2 factors
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Outer blocks A00, A01, A10, A11
    # A00 is rows 0-1, cols 0-1
    # A01 is rows 0-1, cols 2-3
    # A10 is rows 2-3, cols 0-1
    # A11 is rows 2-3, cols 2-3
    
    # We define the linear combinations for U and V for each of the 7 outer steps.
    # Then for each outer step, we apply the 7 inner Strassen steps.
    
    # Define A blocks as slices of indices
    # A[i, j] index = i*4 + j
    
    # Outer Step 1: M1 = (A00+A11)(B00+B11)
    # U part: A00 + A11
    # V part: B00 + B11
    
    # Let's define the 7 outer steps' U and V contributions as lists of (index, coeff)
    
    def block_indices(br, bc):
        """Return list of (global_idx, coeff) for a 2x2 block"""
        idxs = []
        for i in range(2):
            for j in range(2):
                gi = (br*2 + i) * 4 + (bc*2 + j)
                idxs.append((gi, 1))
        return idxs

    A00 = block_indices(0, 0)
    A01 = block_indices(0, 1)
    A10 = block_indices(1, 0)
    A11 = block_indices(1, 1)
    
    B00 = block_indices(0, 0)
    B01 = block_indices(0, 1)
    B10 = block_indices(1, 0)
    B11 = block_indices(1, 1)
    
    def combine_blocks(blocks_coeffs):
        """
        blocks_coeffs: list of (block_list, coeff)
        Returns list of (global_idx, total_coeff)
        """
        d = {}
        for block, coeff in blocks_coeffs:
            for idx, c in block:
                d[idx] = d.get(idx, 0) + coeff * c
        return [(idx, c) for idx, c in d.items() if c != 0]

    # Outer Steps
    # 1: (A00+A11), (B00+B11)
    # 2: (A10+A11), B00
    # 3: A00, (B01-B11)
    # 4: A11, (B10-B00)
    # 5: (A00+A01), B11
    # 6: (A10-A00), (B00+B01)
    # 7: (A01-A11), (B10+B11)
    
    outer_U = [
        combine_blocks([(A00, 1), (A11, 1)]),
        combine_blocks([(A10, 1), (A11, 1)]),
        combine_blocks([(A00, 1)]),
        combine_blocks([(A11, 1)]),
        combine_blocks([(A00, 1), (A01, 1)]),
        combine_blocks([(A10, 1), (A00, -1)]),
        combine_blocks([(A01, 1), (A11, -1)])
    ]
    
    outer_V = [
        combine_blocks([(B00, 1), (B11, 1)]),
        combine_blocks([(B00, 1)]),
        combine_blocks([(B01, 1), (B11, -1)]),
        combine_blocks([(B10, 1), (B00, -1)]),
        combine_blocks([(B11, 1)]),
        combine_blocks([(B00, 1), (B01, 1)]),
        combine_blocks([(B10, 1), (B11, 1)])
    ]
    
    # Inner Strassen Steps for 2x2 matrices X, Y
    # X = [[X00, X01], [X10, X11]]
    # Y = [[Y00, Y01], [Y10, Y11]]
    # Z = XY
    
    # Inner Step 1: (X00+X11)(Y00+Y11)
    # Inner Step 2: (X10+X11)Y00
    # Inner Step 3: X00(Y01-Y11)
    # Inner Step 4: X11(Y10-Y00)
    # Inner Step 5: (X00+X01)Y11
    # Inner Step 6: (X10-X00)(Y00+Y01)
    # Inner Step 7: (X01-X11)(Y10+Y11)
    
    # We need to map the "local" indices of X and Y (which are 2x2 blocks of the original A and B)
    # to the global indices.
    # The "local" index (li, lj) in the 2x2 block corresponds to global offset.
    # But here, X and Y are already linear combinations of global A and B elements.
    # So, for a specific outer step t, X is a 2x2 matrix where each entry is a linear combo of global A.
    # Wait, the structure is:
    # Outer step t defines a 2x2 matrix X_t and Y_t.
    # X_t[i][j] is a linear combo of A elements.
    # Y_t[i][j] is a linear combo of B elements.
    # Then we multiply X_t * Y_t using Strassen (rank 7).
    # The result Z_t is a 2x2 matrix of linear combos of products.
    # Then we sum Z_t with coefficients to get C.
    
    # Let's build U, V, W for each (t, k) pair.
    
    col = 0
    for t in range(7):
        # X_t and Y_t are 2x2 matrices of linear combos.
        # We need to extract the "local" 2x2 structure of X_t and Y_t.
        # X_t is formed by combining A blocks.
        # A00 is at (0,0), A01 at (0,1), etc.
        # So X_t[0][0] comes from A00, X_t[0][1] from A01, etc.
        # Actually, the outer step defines X_t as a 2x2 matrix where:
        # X_t[0][0] = A00 (or combo)
        # X_t[0][1] = A01 (or combo)
        # X_t[1][0] = A10 (or combo)
        # X_t[1][1] = A11 (or combo)
        
        # Let's define the coefficients for each A block in X_t
        # X_t[i][j] = sum_{br, bc} coeff[i][j][br][bc] * A_{br, bc}
        # This is getting complicated to represent generically.
        
        # Simpler approach:
        # For each outer step t, we know which A blocks are involved.
        # e.g., t=0: X = [[A00, 0], [0, A11]] + [[0,0],[0,0]]? No.
        # X = A00 + A11 means X[0][0] = A00, X[1][1] = A11, X[0][1]=0, X[1][0]=0.
        # Wait, Strassen operates on the 2x2 matrix structure.
        # If X = A00 + A11, it means the 2x2 matrix X is:
        # [[A00, 0], [0, A11]]? No.
        # The matrix X is a 2x2 matrix of scalars? No, X is a 2x2 matrix of 2x2 blocks?
        # No, the recursive step treats the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
        # Let A = [[A00, A01], [A10, A11]].
        # Strassen computes M1 = (A00+A11)(B00+B11).
        # Here A00+A11 is a 2x2 matrix sum.
        # So the "left operand" for the inner multiplication is the 2x2 matrix (A00+A11).
        # The "right operand" is (B00+B11).
        # The inner multiplication is the product of two 2x2 matrices.
        # So we need to decompose the product of two 2x2 matrices (X and Y) into 7 scalar multiplications.
        # X and Y are 2x2 matrices.
        
        # So for t=0:
        # X = [[A00, 0], [0, A11]]? No.
        # X is the matrix [[A00, 0], [0, A11]]?
        # A00 is a 2x2 block.
        # So X is a 2x2 matrix where the (0,0) entry is the 2x2 block A00, and (1,1) is A11?
        # No, that would be a 4x4 matrix.
        # The recursion is:
        # Level 1: 4x4 matrices. Split into 2x2 blocks.
        # Level 2: 2x2 matrices (the blocks). Multiply them using Strassen.
        
        # So, for each outer step t, we have a 2x2 matrix X_t and Y_t.
        # X_t is a 2x2 matrix where each entry is a linear combination of the original A elements?
        # No, X_t is a 2x2 matrix where each entry is a 2x2 block?
        # No, the "inner" Strassen works on 2x2 matrices of scalars.
        # But here, the "scalars" are the elements of the 2x2 blocks?
        # Yes.
        # So X_t is a 2x2 matrix.
        # X_t[0][0] corresponds to the top-left element of the block A00?
        # No, X_t is a 2x2 matrix of "blocks"?
        # Let's clarify the structure.
        
        # Standard Strassen for 4x4:
        # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
        # M1 = (A00+A11)(B00+B11)
        # Here A00+A11 is a 2x2 matrix.
        # We need to compute the product of two 2x2 matrices: X = A00+A11 and Y = B00+B11.
        # The product Z = X Y is a 2x2 matrix.
        # Z is added to C00 and C11.
        
        # So, the inner Strassen decomposition applies to the 2x2 matrices X and Y.
        # X and Y are 2x2 matrices.
        # The elements of X are linear combinations of elements of A.
        # Specifically, X[0][0] = A00[0][0] + A11[0][0] = A[0][0] + A[2][2].
        # X[0][1] = A00[0][1] + A11[0][1] = A[0][1] + A[2][3].
        # X[1][0] = A00[1][0] + A11[1][0] = A[1][0] + A[3][2].
        # X[1][1] = A00[1][1] + A11[1][1] = A[1][1] + A[3][3].
        
        # So for each outer step t, we can determine the linear combinations for the 4 entries of X_t and Y_t.
        
        # Let's define the linear combinations for X_t and Y_t for each t.
        # X_t[i][j] is a linear combo of A[r][c].
        
        def get_X_combo(t):
            """Returns 2x2 list of lists of (r, c, coeff)"""
            # t=0: X = A00 + A11
            if t == 0:
                return [
                    [((0,0,1), (2,2,1))], [((0,1,1), (2,3,1))],
                    [((1,0,1), (3,2,1))], [((1,1,1), (3,3,1))]
                ]
            # t=1: X = A10 + A11
            elif t == 1:
                return [
                    [((2,0,1), (3,0,1))], [((2,1,1), (3,1,1))],
                    [((2,0,1), (3,0,1))], [((2,1,1), (3,1,1))] # Wait, A10 is rows 2-3, cols 0-1. A11 is rows 2-3, cols 2-3.
                    # A10[0][0] = A[2][0]. A11[0][0] = A[2][2].
                    # So X[0][0] = A[2][0] + A[2][2].
                ]
            # Let's do this programmatically.
            
            # Define blocks A00, A01, A10, A11 as 2x2 arrays of (r,c)
            blocks = {
                'A00': [[(0,0), (0,1)], [(1,0), (1,1)]],
                'A01': [[(0,2), (0,3)], [(1,2), (1,3)]],
                'A10': [[(2,0), (2,1)], [(3,0), (3,1)]],
                'A11': [[(2,2), (2,3)], [(3,2), (3,3)]]
            }
            
            # Define which blocks are added/subtracted for X_t
            # t=0: A00+A11
            # t=1: A10+A11
            # t=2: A00
            # t=3: A11
            # t=4: A00+A01
            # t=5: A10-A00
            # t=6: A01-A11
            
            def add_blocks(b1, b2, sign1=1, sign2=1):
                res = [[[] for _ in range(2)] for _ in range(2)]
                for i in range(2):
                    for j in range(2):
                        r, c = blocks[b1][i][j]
                        res[i][j].append((r, c, sign1))
                        r2, c2 = blocks[b2][i][j]
                        res[i][j].append((r2, c2, sign2))
                return res
            
            def single_block(b, sign=1):
                res = [[[] for _ in range(2)] for _ in range(2)]
                for i in range(2):
                    for j in range(2):
                        r, c = blocks[b][i][j]
                        res[i][j].append((r, c, sign))
                return res

            if t == 0: return add_blocks('A00', 'A11')
            if t == 1: return add_blocks('A10', 'A11')
            if t == 2: return single_block('A00')
            if t == 3: return single_block('A11')
            if t == 4: return add_blocks('A00', 'A01')
            if t == 5: return add_blocks('A10', 'A00', 1, -1)
            if t == 6: return add_blocks('A01', 'A11', 1, -1)
            
        def get_Y_combo(t):
            # Same logic for B
            blocks = {
                'B00': [[(0,0), (0,1)], [(1,0), (1,1)]],
                'B01': [[(0,2), (0,3)], [(1,2), (1,3)]],
                'B10': [[(2,0), (2,1)], [(3,0), (3,1)]],
                'B11': [[(2,2), (2,3)], [(3,2), (3,3)]]
            }
            
            def add_blocks(b1, b2, sign1=1, sign2=1):
                res = [[[] for _ in range(2)] for _ in range(2)]
                for i in range(2):
                    for j in range(2):
                        r, c = blocks[b1][i][j]
                        res[i][j].append((r, c, sign1))
                        r2, c2 = blocks[b2][i][j]
                        res[i][j].append((r2, c2, sign2))
                return res
            
            def single_block(b, sign=1):
                res = [[[] for _ in range(2)] for _ in range(2)]
                for i in range(2):
                    for j in range(2):
                        r, c = blocks[b][i][j]
                        res[i][j].append((r, c, sign))
                return res

            if t == 0: return add_blocks('B00', 'B11')
            if t == 1: return single_block('B00')
            if t == 2: return add_blocks('B01', 'B11', 1, -1)
            if t == 3: return add_blocks('B10', 'B00', 1, -1)
            if t == 4: return single_block('B11')
            if t == 5: return add_blocks('B00', 'B01')
            if t == 6: return add_blocks('B10', 'B11')

        X_t = get_X_combo(t)
        Y_t = get_Y_combo(t)
        
        # Now apply inner Strassen to X_t and Y_t
        # Inner Strassen steps for 2x2 matrices X, Y:
        # k=0: (X00+X11)(Y00+Y11)
        # k=1: (X10+X11)Y00
        # k=2: X00(Y01-Y11)
        # k=3: X11(Y10-Y00)
        # k=4: (X00+X01)Y11
        # k=5: (X10-X00)(Y00+Y01)
        # k=6: (X01-X11)(Y10+Y11)
        
        # We need to extract the linear combos for the "inner" 2x2 matrices.
        # For k=0:
        # X_inner = [[X00+X11, 0], [0, 0]]? No.
        # X_inner is a 2x2 matrix.
        # X_inner[0][0] = X[0][0] + X[1][1]
        # X_inner[0][1] = 0
        # X_inner[1][0] = 0
        # X_inner[1][1] = 0
        # Wait, Strassen multiplies 2x2 matrices.
        # The "rank 1" term is a scalar multiplication.
        # The scalar is formed by summing products?
        # No, the rank 1 term is $u_k v_k^T$.
        # $u_k$ is a vector of coefficients for A.
        # $v_k$ is a vector of coefficients for B.
        
        # For the inner step k, we form two vectors:
        # $x_k$ (coeffs for A) and $y_k$ (coeffs for B).
        # $x_k$ is formed by the linear combination of the entries of X_t.
        # $y_k$ is formed by the linear combination of the entries of Y_t.
        
        # Example k=0:
        # $x_0$ corresponds to $X_{00} + X_{11}$.
        # $X_{00}$ is a linear combo of A. $X_{11}$ is a linear combo of A.
        # So $x_0 = X_{00} + X_{11}$.
        # $y_0 = Y_{00} + Y_{11}$.
        
        # Example k=1:
        # $x_1 = X_{10} + X_{11}$.
        # $y_1 = Y_{00}$.
        
        # So we just need to sum the linear combos of the appropriate entries of X_t and Y_t.
        
        def sum_combos(combos):
            """combos: list of lists of (r, c, coeff)"""
            d = {}
            for combo in combos:
                for (r, c, v) in combo:
                    d[(r,c)] = d.get((r,c), 0) + v
            return [(r, c, v) for (r,c), v in d.items() if v != 0]

        # Define which entries of X and Y are used for each inner step k
        # k=0: X[0][0]+X[1][1], Y[0][0]+Y[1][1]
        # k=1: X[1][0]+X[1][1], Y[0][0]
        # k=2: X[0][0], Y[0][1]-Y[1][1]
        # k=3: X[1][1], Y[1][0]-Y[0][0]
        # k=4: X[0][0]+X[0][1], Y[1][1]
        # k=5: X[1][0]-X[0][0], Y[0][0]+Y[0][1]
        # k=6: X[0][1]-X[1][1], Y[1][0]+Y[1][1]
        
        inner_X_indices = [
            [(0,0), (1,1)],
            [(1,0), (1,1)],
            [(0,0)],
            [(1,1)],
            [(0,0), (0,1)],
            [(1,0), (0,0)], # with signs
            [(0,1), (1,1)]  # with signs
        ]
        
        # We need signs too.
        # k=5: X[1][0] - X[0][0]
        # k=6: X[0][1] - X[1][1]
        
        def get_inner_combo(X, Y, k):
            if k == 0:
                x = sum_combos([X[0][0], X[1][1]])
                y = sum_combos([Y[0][0], Y[1][1]])
            elif k == 1:
                x = sum_combos([X[1][0], X[1][1]])
                y = sum_combos([Y[0][0]])
            elif k == 2:
                x = sum_combos([X[0][0]])
                y = sum_combos([Y[0][1], Y[1][1]]) # Y[0][1] - Y[1][1]
                # Need to handle signs
                y = []
                d = {}
                for (r,c,v) in Y[0][1]: d[(r,c)] = d.get((r,c),0) + v
                for (r,c,v) in Y[1][1]: d[(r,c)] = d.get((r,c),0) - v
                y = [(r,c,v) for (r,c),v in d.items() if v!=0]
            elif k == 3:
                x = sum_combos([X[1][1]])
                d = {}
                for (r,c,v) in Y[1][0]: d[(r,c)] = d.get((r,c),0) + v
                for (r,c,v) in Y[0][0]: d[(r,c)] = d.get((r,c),0) - v
                y = [(r,c,v) for (r,c),v in d.items() if v!=0]
            elif k == 4:
                x = sum_combos([X[0][0], X[0][1]])
                y = sum_combos([Y[1][1]])
            elif k == 5:
                d = {}
                for (r,c,v) in X[1][0]: d[(r,c)] = d.get((r,c),0) + v
                for (r,c,v) in X[0][0]: d[(r,c)] = d.get((r,c),0) - v
                x = [(r,c,v) for (r,c),v in d.items() if v!=0]
                y = sum_combos([Y[0][0], Y[0][1]])
            elif k == 6:
                d = {}
                for (r,c,v) in X[0][1]: d[(r,c)] = d.get((r,c),0) + v
                for (r,c,v) in X[1][1]: d[(r,c)] = d.get((r,c),0) - v
                x = [(r,c,v) for (r,c),v in d.items() if v!=0]
                y = sum_combos([Y[1][0], Y[1][1]])
            return x, y

        for k in range(7):
            x_combo, y_combo = get_inner_combo(X_t, Y_t, k)
            
            # Fill U and V
            for (r, c, v) in x_combo:
                U[r*4+c][col] += v
            for (r, c, v) in y_combo:
                V[r*4+c][col] += v
            
            # Fill W
            # Determine which C elements are affected.
            # Outer step t contributes to C blocks.
            # Inner step k contributes to Z elements.
            
            # Outer targets:
            # t=0: C00(+1), C11(+1)
            # t=1: C10(+1), C11(-1)
            # t=2: C01(+1), C11(+1)
            # t=3: C00(+1), C10(+1)
            # t=4: C00(-1), C01(+1)
            # t=5: C11(+1)
            # t=6: C00(+1)
            
            outer_targets = {
                0: [(0,0,1), (1,1,1)],
                1: [(1,0,1), (1,1,-1)],
                2: [(0,1,1), (1,1,1)],
                3: [(0,0,1), (1,0,1)],
                4: [(0,0,-1), (0,1,1)],
                5: [(1,1,1)],
                6: [(0,0,1)]
            }
            
            # Inner targets:
            # k=0: Z00(+1), Z11(+1)
            # k=1: Z10(+1), Z11(-1)
            # k=2: Z01(+1), Z11(+1)
            # k=3: Z00(+1), Z10(+1)
            # k=4: Z00(-1), Z01(+1)
            # k=5: Z11(+1)
            # k=6: Z00(+1)
            
            inner_targets = {
                0: [(0,0,1), (1,1,1)],
                1: [(1,0,1), (1,1,-1)],
                2: [(0,1,1), (1,1,1)],
                3: [(0,0,1), (1,0,1)],
                4: [(0,0,-1), (0,1,1)],
                5: [(1,1,1)],
                6: [(0,0,1)]
            }
            
            for (br, bc, c_coeff) in outer_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    gi = br*2 + li
                    gk = bc*2 + lj
                    W[gk*4+gi][col] += c_coeff * z_coeff
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
