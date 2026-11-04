DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # We'll build the decomposition term by term
    terms = []  # each term is (u, v, w) where u is length n*m=16, v is length m*p=28, w is length p*n=28
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[u_idx] = val
        v[v_idx] = val
        w[w_idx] = val
        terms.append((u, v, w))

    # Part 1: 4x4 by 4x4 multiplication (First 4 columns of B)
    # Using Strassen's algorithm for rank 28
    
    # Indices for A (4x4)
    # A[i, j] -> index i*m + j
    
    # Strassen blocks for A and B (first 4x4 part)
    # A11, A12, A21, A22
    # B11, B12, B21, B22
    
    # We need to represent the linear combinations for the 7 products.
    # Product 1: (A11 + A22) * (B11 + B22)
    # A11 entries: A[0,0], A[0,1], A[1,0], A[1,1]
    # A22 entries: A[2,0], A[2,1], A[3,0], A[3,1]
    # So X1 = A11 + A22 has entries:
    # X1[0,0] = A[0,0] + A[2,0]
    # X1[0,1] = A[0,1] + A[2,1]
    # X1[1,0] = A[1,0] + A[3,0]
    # X1[1,1] = A[1,1] + A[3,1]
    
    # Similarly for Y1 = B11 + B22
    
    # This is getting complex to hardcode all 28 terms manually.
    # I will write a helper to generate terms for a 2x2 multiplication
    # where the inputs are defined by linear combinations.
    
    def add_2x2_mult_terms(x_combos, y_combos, c_offset):
        # x_combos: list of 4 lists, each list of (idx, coeff) for A
        # y_combos: list of 4 lists, each list of (idx, coeff) for B
        # c_offset: offset for C indices (0 for first 4x4 part)
        
        # x_combos[i] corresponds to X[i, j]
        # Actually, let's define X entries as linear forms.
        # X[0,0], X[0,1], X[1,0], X[1,1]
        # Y[0,0], Y[0,1], Y[1,0], Y[1,1]
        
        # 4 terms for Z = X * Y
        # Z[0,0] = X[0,0]Y[0,0] + X[0,1]Y[1,0]
        # Z[0,1] = X[0,0]Y[0,1] + X[0,1]Y[1,1]
        # Z[1,0] = X[1,0]Y[0,0] + X[1,1]Y[1,0]
        # Z[1,1] = X[1,0]Y[0,1] + X[1,1]Y[1,1]
        
        # We add terms for each product of linear forms.
        # Product of (sum a_i u_i) * (sum b_j v_j) = sum a_i b_j (u_i v_j)
        # Each (u_i v_j) is a rank-1 term in the global space.
        
        def expand_product(x_form, y_form, c_idx):
            for (xi, cx) in x_form:
                for (yi, cy) in y_form:
                    val = cx * cy
                    if val != 0:
                        add_term(xi, yi, c_idx, val)
        
        # Z[0,0]
        # X[0,0] * Y[0,0]
        expand_product(x_combos[0], y_combos[0], c_offset + 0)
        # X[0,1] * Y[1,0]
        expand_product(x_combos[1], y_combos[2], c_offset + 0) # Wait, Y[1,0] is index 2 in 2x2 flat? 
        # Let's use explicit indices for 2x2:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # Z[0,0] = X[0,0]Y[0,0] + X[0,1]Y[1,0]
        expand_product(x_combos[0], y_combos[0], c_offset + 0)
        expand_product(x_combos[1], y_combos[2], c_offset + 0)
        
        # Z[0,1] = X[0,0]Y[0,1] + X[0,1]Y[1,1]
        expand_product(x_combos[0], y_combos[1], c_offset + 1)
        expand_product(x_combos[1], y_combos[3], c_offset + 1)
        
        # Z[1,0] = X[1,0]Y[0,0] + X[1,1]Y[1,0]
        expand_product(x_combos[2], y_combos[0], c_offset + 4) # C[1,0] is index 4 in 4x4?
        # C is 4x4. C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3
        # C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
        # C[2,0]=8, ...
        # So C[1,0] is 4.
        
        # Let's re-verify C indices.
        # C[i, j] -> i*4 + j.
        # Z is 2x2.
        # Z[0,0] maps to C[0,0] (offset 0)
        # Z[0,1] maps to C[0,1] (offset 1)
        # Z[1,0] maps to C[1,0] (offset 4)
        # Z[1,1] maps to C[1,1] (offset 5)
        
        # My previous expand calls used c_offset + 0, 1, 4, 5. Correct.
        
        # Z[1,0]
        expand_product(x_combos[2], y_combos[0], c_offset + 4)
        expand_product(x_combos[3], y_combos[2], c_offset + 4)
        
        # Z[1,1]
        expand_product(x_combos[2], y_combos[1], c_offset + 5)
        expand_product(x_combos[3], y_combos[3], c_offset + 5)

    # Define the 7 Strassen products
    # A indices: 0-15
    # B indices: 0-27 (first 16 for 4x4 part)
    
    # Helper to get linear form for a 2x2 block sum/diff
    # Block (br, bc) of A: rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    def get_block_form(mat_type, br, bc, sign=1):
        # mat_type: 'A' or 'B'
        # Returns list of (global_idx, coeff)
        form = []
        for i in range(2):
            for j in range(2):
                r = 2*br + i
                c = 2*bc + j
                if mat_type == 'A':
                    idx = r * 4 + c
                else:
                    idx = r * 4 + c # B first 4x4 part
                form.append((idx, sign))
        return form

    # M1: (A11 + A22) * (B11 + B22)
    x1 = get_block_form('A', 0, 0, 1) + get_block_form('A', 1, 1, 1)
    y1 = get_block_form('B', 0, 0, 1) + get_block_form('B', 1, 1, 1)
    # We need to map the 4 entries of X1 to x_combos[0..3]
    # X1[0,0] = A[0,0] + A[2,0]
    # X1[0,1] = A[0,1] + A[2,1]
    # X1[1,0] = A[1,0] + A[3,0]
    # X1[1,1] = A[1,1] + A[3,1]
    
    # My get_block_form returns a flat list of 8 items.
    # I need to split them into 4 linear forms.
    
    def split_block_form(block_form, br, bc, mat_type):
        # block_form is list of (idx, coeff)
        # We know the structure.
        # idx for A[br*2, bc*2] is br*2*4 + bc*2
        # Let's just reconstruct from indices.
        
        # This is getting too complicated for a single edit.
        # I will hardcode the 28 terms for the 4x4x4 part.
        # Or use a simpler improvement: Just use Schoolbook for 4x4x4 (64) and Schoolbook for 4x4x3 (48). Total 112. No improvement.
        
        # What if I use Strassen for 4x4x4 but just implement it correctly?
        
        # Let's try a different approach: 
        # The rank of M_{4,4,4} is 28.
        # The rank of M_{4,4,3} is 48 (schoolbook).
        # Total 76.
        
        # I will generate the 28 terms programmatically.
        
        # Define the 7 pairs of (X_form, Y_form)
        # X_form is a list of 4 linear forms (for X[0,0], X[0,1], X[1,0], X[1,1])
        # Y_form is a list of 4 linear forms
        
        def make_linear_form(entries):
            # entries: list of (idx, coeff)
            return entries

        # A11: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
        # A22: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
        # B11: 0,1,4,5
        # B22: 8,9,12,13
        
        # M1: X = A11+A22, Y = B11+B22
        # X[0,0] = A[0,0]+A[2,0] = 0 + 8
        # X[0,1] = A[0,1]+A[2,1] = 1 + 9
        # X[1,0] = A[1,0]+A[3,0] = 4 + 12
        # X[1,1] = A[1,1]+A[3,1] = 5 + 13
        
        x1 = [
            [(0,1), (8,1)],
            [(1,1), (9,1)],
            [(4,1), (12,1)],
            [(5,1), (13,1)]
        ]
        y1 = [
            [(0,1), (8,1)],
            [(1,1), (9,1)],
            [(4,1), (12,1)],
            [(5,1), (13,1)]
        ]
        add_2x2_mult_terms(x1, y1, 0)

        # M2: X = A21+A22, Y = B11
        # A21: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
        # A22: 8,9,12,13
        # X[0,0] = A[2,0]+A[2,0] = 2*A[2,0] = 16? No, A[2,0] is idx 8.
        # Wait, A21 is rows 2,3 cols 0,1. A22 is rows 2,3 cols 2,3.
        # A21: 8,9,12,13.
        # A22: 10,11,14,15.
        # My previous indices for A22 were wrong?
        # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
        # A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15
        
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # Correcting M1:
        # X = A11+A22
        # X[0,0] = A[0,0]+A[2,2] = 0 + 10
        # X[0,1] = A[0,1]+A[2,3] = 1 + 11
        # X[1,0] = A[1,0]+A[3,2] = 4 + 14
        # X[1,1] = A[1,1]+A[3,3] = 5 + 15
        
        # I will restart the term generation with correct indices.
        
        # Clear terms
        terms.clear()
        
        # M1: (A11+A22)(B11+B22)
        x1 = [
            [(0,1), (10,1)],
            [(1,1), (11,1)],
            [(4,1), (14,1)],
            [(5,1), (15,1)]
        ]
        y1 = [
            [(0,1), (10,1)],
            [(1,1), (11,1)],
            [(4,1), (14,1)],
            [(5,1), (15,1)]
        ]
        add_2x2_mult_terms(x1, y1, 0)

        # M2: (A21+A22)B11
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        # X[0,0] = A[2,0]+A[2,2] = 8+10
        # X[0,1] = A[2,1]+A[2,3] = 9+11
        # X[1,0] = A[3,0]+A[3,2] = 12+14
        # X[1,1] = A[3,1]+A[3,3] = 13+15
        x2 = [
            [(8,1), (10,1)],
            [(9,1), (11,1)],
            [(12,1), (14,1)],
            [(13,1), (15,1)]
        ]
        # Y = B11
        y2 = [
            [(0,1)],
            [(1,1)],
            [(4,1)],
            [(5,1)]
        ]
        add_2x2_mult_terms(x2, y2, 0)

        # M3: A11(B12-B22)
        x3 = [
            [(0,1)],
            [(1,1)],
            [(4,1)],
            [(5,1)]
        ]
        # Y = B12 - B22
        # B12: 2,3,6,7
        # B22: 10,11,14,15
        y3 = [
            [(2,1), (10,-1)],
            [(3,1), (11,-1)],
            [(6,1), (14,-1)],
            [(7,1), (15,-1)]
        ]
        add_2x2_mult_terms(x3, y3, 0)

        # M4: A22(B21-B11)
        x4 = [
            [(10,1)],
            [(11,1)],
            [(14,1)],
            [(15,1)]
        ]
        # Y = B21 - B11
        # B21: 8,9,12,13
        # B11: 0,1,4,5
        y4 = [
            [(8,1), (0,-1)],
            [(9,1), (1,-1)],
            [(12,1), (4,-1)],
            [(13,1), (5,-1)]
        ]
        add_2x2_mult_terms(x4, y4, 0)

        # M5: (A11+A12)B22
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        x5 = [
            [(0,1), (2,1)],
            [(1,1), (3,1)],
            [(4,1), (6,1)],
            [(5,1), (7,1)]
        ]
        y5 = [
            [(10,1)],
            [(11,1)],
            [(14,1)],
            [(15,1)]
        ]
        add_2x2_mult_terms(x5, y5, 0)

        # M6: (A21-A11)(B11+B12)
        # A21: 8,9,12,13
        # A11: 0,1,4,5
        x6 = [
            [(8,1), (0,-1)],
            [(9,1), (1,-1)],
            [(12,1), (4,-1)],
            [(13,1), (5,-1)]
        ]
        # Y = B11+B12
        # B11: 0,1,4,5
        # B12: 2,3,6,7
        y6 = [
            [(0,1), (2,1)],
            [(1,1), (3,1)],
            [(4,1), (6,1)],
            [(5,1), (7,1)]
        ]
        add_2x2_mult_terms(x6, y6, 0)

        # M7: (A12-A22)(B21+B22)
        # A12: 2,3,6,7
        # A22: 10,11,14,15
        x7 = [
            [(2,1), (10,-1)],
            [(3,1), (11,-1)],
            [(6,1), (14,-1)],
            [(7,1), (15,-1)]
        ]
        # Y = B21+B22
        # B21: 8,9,12,13
        # B22: 10,11,14,15
        y7 = [
            [(8,1), (10,1)],
            [(9,1), (11,1)],
            [(12,1), (14,1)],
            [(13,1), (15,1)]
        ]
        add_2x2_mult_terms(x7, y7, 0)

        # Part 2: 4x4 by 4x3 multiplication (Last 3 columns of B)
        # Schoolbook
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    # A[i, j] * B[j, 4+k]
                    u_idx = i * 4 + j
                    v_idx = j * 7 + (4 + k)
                    w_idx = (4 + k) * 4 + i
                    add_term(u_idx, v_idx, w_idx, 1)

        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(terms):
            for r in range(n * m):
                U[r][c] = u[r]
            for r in range(m * p):
                V[r][c] = v[r]
            for r in range(p * n):
                W[r][c] = w[r]
        
        return {"U": U, "V": V, "W": W}
