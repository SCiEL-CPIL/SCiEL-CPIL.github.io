import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Recursive Strassen: 4x4 = 2x2 blocks of 2x2, each 2x2 uses Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 decomposition:
    # For [[a,b],[c,d]] * [[e,f],[g,h]]:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # p11 = m1 + m4 - m5 + m7
    # p12 = m3 + m5
    # p21 = m2 + m4
    # p22 = m1 - m2 + m3 + m6
    
    # For 4x4, partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Block-level Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, decomposed via Strassen (7 terms)
    # Total: 7 block terms * 7 inner terms = 49 scalar products
    
    # Define the 7 block terms with their X (from A) and Y (from B) definitions
    # X and Y are 2x2 matrices where each element is a linear combination of original elements
    
    # For U: row i*m+j corresponds to A[i,j]
    # For V: row j*p+k corresponds to B[j,k]
    # For W: row k*n+i corresponds to C[i,k]
    
    # Block term definitions:
    # Each block term has X (2x2 from A) and Y (2x2 from B)
    # X[r][c] and Y[r][c] are lists of (element_index, coeff)
    
    def get_block_terms():
        """Return list of 7 block terms, each with X_map, Y_map, W_map."""
        terms = []
        
        # M1: X = A11+A22, Y = B11+B22
        X = [[[(0,1),(10,1)], [(1,1),(11,1)]],
             [[(4,1),(14,1)], [(5,1),(15,1)]]]
        Y = [[[(0,1),(10,1)], [(1,1),(11,1)]],
             [[(4,1),(14,1)], [(5,1),(15,1)]]]
        W = {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,2):1, (2,3):1, (3,2):1, (3,3):1}
        terms.append((X, Y, W))
        
        # M2: X = A21+A22, Y = B11
        X = [[[(8,1),(10,1)], [(9,1),(11,1)]],
             [[(12,1),(14,1)], [(13,1),(15,1)]]]
        Y = [[[(0,1)], [(1,1)]],
             [[(4,1)], [(5,1)]]]
        W = {(2,0):1, (2,1):1, (3,0):1, (3,1):1, (2,2):-1, (2,3):-1, (3,2):-1, (3,3):-1}
        terms.append((X, Y, W))
        
        # M3: X = A11, Y = B12-B22
        X = [[[(0,1)], [(1,1)]],
             [[(4,1)], [(5,1)]]]
        Y = [[[(2,1),(10,-1)], [(3,1),(11,-1)]],
             [[(6,1),(14,-1)], [(7,1),(15,-1)]]]
        W = {(0,2):1, (0,3):1, (1,2):1, (1,3):1, (2,2):1, (2,3):1, (3,2):1, (3,3):1}
        terms.append((X, Y, W))
        
        # M4: X = A22, Y = B21-B11
        X = [[[(10,1)], [(11,1)]],
             [[(14,1)], [(15,1)]]]
        Y = [[[(8,1),(0,-1)], [(9,1),(1,-1)]],
             [[(12,1),(4,-1)], [(13,1),(5,-1)]]]
        W = {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,0):1, (2,1):1, (3,0):1, (3,1):1}
        terms.append((X, Y, W))
        
        # M5: X = A11+A12, Y = B22
        X = [[[(0,1),(2,1)], [(1,1),(3,1)]],
             [[(4,1),(6,1)], [(5,1),(7,1)]]]
        Y = [[[(10,1)], [(11,1)]],
             [[(14,1)], [(15,1)]]]
        W = {(0,0):-1, (0,1):-1, (1,0):-1, (1,1):-1, (0,2):1, (0,3):1, (1,2):1, (1,3):1}
        terms.append((X, Y, W))
        
        # M6: X = A21-A11, Y = B11+B12
        X = [[[(8,1),(0,-1)], [(9,1),(1,-1)]],
             [[(12,1),(4,-1)], [(13,1),(5,-1)]]]
        Y = [[[(0,1),(2,1)], [(1,1),(3,1)]],
             [[(4,1),(6,1)], [(5,1),(7,1)]]]
        W = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
        terms.append((X, Y, W))
        
        # M7: X = A12-A22, Y = B21+B22
        X = [[[(2,1),(10,-1)], [(3,1),(11,-1)]],
             [[(6,1),(14,-1)], [(7,1),(15,-1)]]]
        Y = [[[(8,1),(10,1)], [(9,1),(11,1)]],
             [[(12,1),(14,1)], [(13,1),(15,1)]]]
        W = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
        terms.append((X, Y, W))
        
        return terms
    
    # Strassen 2x2 inner decomposition:
    # For X*Y where X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
    # t1: (x00+x11)(y00+y11) -> contributes to z00(+1), z11(+1)
    # t2: (x10+x11)y00 -> contributes to z10(+1), z11(-1)
    # t3: x00(y01-y11) -> contributes to z01(+1), z11(+1)
    # t4: x11(y10-y00) -> contributes to z00(+1), z10(+1)
    # t5: (x00+x01)y11 -> contributes to z00(-1), z01(+1)
    # t6: (x10-x00)(y00+y01) -> contributes to z11(+1)
    # t7: (x01-x11)(y10+y11) -> contributes to z00(+1)
    
    # For each inner term, define which X elements, Y elements are involved,
    # with coefficients, and which Z elements get contributions.
    
    inner_terms = [
        # (x_indices, x_coeffs, y_indices, y_coeffs, z_targets)
        # x_indices: indices into [x00, x01, x10, x11]
        # y_indices: indices into [y00, y01, y10, y11]
        # z_targets: list of (z_index, coeff) where z_index is into [z00, z01, z10, z11]
        ([0, 3], [1, 1], [0, 3], [1, 1], [(0, 1), (3, 1)]),      # t1
        ([2, 3], [1, 1], [0], [1], [(2, 1), (3, -1)]),            # t2
        ([0], [1], [1, 3], [1, -1], [(1, 1), (3, 1)]),            # t3
        ([3], [1], [2, 0], [1, -1], [(0, 1), (2, 1)]),            # t4
        ([0, 1], [1, 1], [3], [1], [(0, -1), (1, 1)]),            # t5
        ([2, 0], [1, -1], [0, 1], [1, 1], [(3, 1)]),              # t6
        ([1, 3], [1, -1], [2, 3], [1, 1], [(0, 1)]),              # t7
    ]
    
    # For each block term and each inner Strassen term, expand into scalar products
    # and add to U, V, W.
    
    block_terms = get_block_terms()
    
    for b_idx, (X_map, Y_map, W_map) in enumerate(block_terms):
        for s_idx, (x_indices, x_coeffs, y_indices, y_coeffs, z_targets) in enumerate(inner_terms):
            col = b_idx * 7 + s_idx
            
            # For each pair (x_elem, y_elem) in the product, add contributions
            # x_elem is a sum of A elements, y_elem is a sum of B elements
            # The product expands to all pairs of (A_elem, B_elem)
            
            # Get the actual element lists for the x and y elements involved
            x_elem_lists = []
            for x_idx, x_coeff in zip(x_indices, x_coeffs):
                # x_idx: 0=x00, 1=x01, 2=x10, 3=x11
                # Map to 2D index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                r = x_idx // 2
                c = x_idx % 2
                elem_list = X_map[r][c]
                x_elem_lists.append((elem_list, x_coeff))
            
            y_elem_lists = []
            for y_idx, y_coeff in zip(y_indices, y_coeffs):
                r = y_idx // 2
                c = y_idx % 2
                elem_list = Y_map[r][c]
                y_elem_lists.append((elem_list, y_coeff))
            
            # For each x element and y element, expand the product
            for x_elem_list, x_coeff in x_elem_lists:
                for y_elem_list, y_coeff in y_elem_lists:
                    # x_elem_list is list of (A_elem_idx, coeff)
                    # y_elem_list is list of (B_elem_idx, coeff)
                    # The product (sum x_coeff * A_i) * (sum y_coeff * B_j)
                    # expands to sum over i,j of (x_coeff * coeff_i) * (y_coeff * coeff_j) * A_i * B_j
                    # But in tensor decomposition, each rank-1 term is A_i * B_j -> C_k
                    # So we need to add for each (i, j) pair
                    
                    for a_idx, a_coeff in x_elem_list:
                        for b_idx_elem, b_coeff in y_elem_list:
                            # This contributes to the column with coefficient
                            # x_coeff * a_coeff * y_coeff * b_coeff
                            coeff = x_coeff * a_coeff * y_coeff * b_coeff
                            if coeff != 0:
                                U[a_idx, col] += coeff
                                V[b_idx_elem, col] += coeff
            
            # Now add W contributions
            for z_idx, z_coeff in z_targets:
                # z_idx: 0=z00, 1=z01, 2=z10, 3=z11
                # Map to 2D: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                zr = z_idx // 2
                zc = z_idx % 2
                # The block result (zr, zc) maps to C elements
                # Block (0,0) = C11: C[0,0], C[0,1], C[1,0], C[1,1]
                # Block (0,1) = C12: C[0,2], C[0,3], C[1,2], C[1,3]
                # Block (1,0) = C21: C[2,0], C[2,1], C[3,0], C[3,1]
                # Block (1,1) = C22: C[2,2], C[2,3], C[3,2], C[3,3]
                # C[i,k] has W row k*n+i = k*4+i
                for dr in range(2):
                    for dc in range(2):
                        i = zr * 2 + dr
                        k = zc * 2 + dc
                        # Check if this C[i,k] is in W_map
                        if (i, k) in W_map:
                            w_coeff = W_map[(i, k)]
                            w_row = k * n + i
                            W[w_row, col] += z_coeff * w_coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
