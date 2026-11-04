import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U: i*m + j -> 2*2 + 2 = 4 rows. i in 0..1, j in 0..1.
    V: j*p + k -> 2*2 + 2 = 4 rows. j in 0..1, k in 0..1.
    W: k*n + i -> 2*2 + 2 = 4 rows. k in 0..1, i in 0..1.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A11 + A22) * (B11 + B22)
    # A11: U[0,0]=1, A22: U[3,0]=1
    # B11: V[0,0]=1, B22: V[3,0]=1
    # C11: W[0,0]=1, C12: W[1,0]=1, C21: W[2,0]=1, C22: W[3,0]=1
    # Wait, C indices:
    # C11 is k=0, i=0 -> W[0]
    # C12 is k=1, i=0 -> W[1]
    # C21 is k=0, i=1 -> W[2]
    # C22 is k=1, i=1 -> W[3]
    
    # P1 contributes to all C with coeff +1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # P2 = (A21 + A22) * B11
    # A21: U[2,1]=1, A22: U[3,1]=1
    # B11: V[0,1]=1
    # C21: W[2,1]=1, C22: W[3,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # P3 = A11 * (B12 - B22)
    # A11: U[0,2]=1
    # B12: V[1,2]=1, B22: V[3,2]=-1
    # C12: W[1,2]=1, C22: W[3,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A22 * (B21 - B11)
    # A22: U[3,3]=1
    # B21: V[2,3]=1, B11: V[0,3]=-1
    # C11: W[0,3]=1, C22: W[3,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[3, 3] = 1
    
    # P5 = (A11 + A12) * B22
    # A11: U[0,4]=1, A12: U[1,4]=1
    # B22: V[3,4]=1
    # C11: W[0,4]=-1, C21: W[2,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6 = (A21 - A11) * (B11 + B12)
    # A21: U[2,5]=1, A11: U[0,5]=-1
    # B11: V[0,5]=1, B12: V[1,5]=1
    # C21: W[2,5]=1, C22: W[3,5]=-1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = -1
    
    # P7 = (A12 - A22) * (B21 + B22)
    # A12: U[1,6]=1, A22: U[3,6]=-1
    # B21: V[2,6]=1, B22: V[3,6]=1
    # C11: W[0,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Builds 4x4x4 Strassen decomposition using 7 blocks of 2x2x2 Strassen.
    Returns U, V, W of shape (16, 49), (16, 49), (16, 49).
    """
    U_2, V_2, W_2 = _build_strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Block indices for 4x4:
    # A blocks: 11 (0-3), 12 (4-7), 21 (8-11), 22 (12-15)
    # B blocks: 11 (0-3), 12 (4-7), 21 (8-11), 22 (12-15)
    # C blocks: 11 (0-3), 12 (4-7), 21 (8-11), 22 (12-15)
    
    # We have 7 Strassen steps M1..M7.
    # Each step uses a 2x2x2 Strassen decomposition.
    # We need to map the 2x2 blocks of the "virtual" 2x2 matrix multiplication
    # to the 4x4 blocks.
    
    # M1 = (A11 + A22) * (B11 + B22)
    # Virtual A: [[A11, 0], [0, A22]]? No, Strassen sums them.
    # The inputs to the 2x2x2 kernel are linear combinations of the 4x4 blocks.
    
    # Let's define the linear combinations for A and B for each M_i.
    # M1: A' = [[A11, 0], [0, A22]]? No, A11 and A22 are 2x2.
    # The 2x2x2 kernel multiplies two 2x2 matrices.
    # Here, the "2x2 matrices" are themselves 2x2 blocks of the 4x4.
    # So M1 computes (A11+A22) * (B11+B22).
    # This is a 2x2 * 2x2 multiplication resulting in a 2x2 matrix M1.
    # M1 is then added to C blocks.
    
    # We will instantiate 7 copies of the 2x2x2 Strassen kernel.
    # For each copy t (0..6), we map:
    # - The 4 rows of U_2 to specific rows of U (4x4 A).
    # - The 4 rows of V_2 to specific rows of V (4x4 B).
    # - The 4 rows of W_2 to specific rows of W (4x4 C).
    # And apply coefficients.
    
    # M1: (A11 + A22) * (B11 + B22)
    # A inputs: A11 (rows 0-3), A22 (rows 12-15).
    # In 2x2x2 kernel, A11 maps to top-left, A22 maps to bottom-right?
    # No, the kernel multiplies A' * B'.
    # A' = [[A11, 0], [0, A22]]? No, Strassen P1 is (A11+A22)(B11+B22).
    # This is NOT a standard 2x2 matrix multiplication of block matrices.
    # It is a specific tensor contraction.
    
    # Actually, it is easier to just expand the 49 terms manually or use a helper.
    # Each term in the 4x4x4 Strassen decomposition is a linear combination of
    # products of 2x2 blocks.
    
    # Let's stick to the 2x2x2 kernel approach.
    # The 2x2x2 kernel computes C' = A' * B'.
    # For M1, we want to compute M1 = (A11+A22) * (B11+B22).
    # This is not A'*B' for simple block matrices A', B'.
    # Wait, (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # This is exactly a 2x2x2 multiplication.
    # So we can use the 2x2x2 kernel.
    # We just need to feed the correct linear combinations into A' and B'.
    
    # A' for M1:
    # A'_{11} = A11 + A22? No, A' is 2x2.
    # The kernel expects A' to be 2x2.
    # A' = [[A11, 0], [0, A22]]? No.
    # The kernel computes A' * B'.
    # If we set A' = [[A11+A22, 0], [0, 0]]? No.
    
    # Let's look at the structure of 2x2x2 Strassen.
    # It takes 2x2 A and 2x2 B.
    # Here, the "2x2 A" for M1 is actually the sum A11+A22?
    # No, A11 and A22 are 2x2 blocks.
    # The operation (A11+A22) * (B11+B22) is a 2x2 * 2x2 multiplication.
    # So we can treat $A_{virt} = A11+A22$ and $B_{virt} = B11+B22$.
    # But $A_{virt}$ is 2x2.
    # So we can use the 2x2x2 kernel on $A_{virt}$ and $B_{virt}$.
    # The output $M1$ is 2x2.
    
    # How to map $A_{virt}$ to the 4x4 U matrix?
    # $A_{virt}$ is 2x2. It corresponds to 4 scalar values.
    # These values are sums of entries from A11 and A22.
    # A11 is at rows 0-3 of U. A22 is at rows 12-15 of U.
    # So for each entry in $A_{virt}$, we add 1 to the corresponding row in U11 and U22.
    
    # Let's define the mapping for each M_i.
    
    # M1: A_v = A11 + A22, B_v = B11 + B22.
    # U mapping: A11 (0-3) + A22 (12-15).
    # V mapping: B11 (0-3) + B22 (12-15).
    # W mapping: M1 contributes to C11, C12, C21, C22.
    # C11 (0-3), C12 (4-7), C21 (8-11), C22 (12-15).
    # M1 is added to C11 (+1), C12 (+1), C21 (+1), C22 (+1).
    # Wait, C11 = M1 + ...
    # So W for M1 should add M1 to all 4 C blocks?
    # No, M1 is 2x2.
    # C11 is 2x2. C12 is 2x2.
    # M1 is added to C11, C12, C21, C22.
    # So the 2x2 result M1 is added to all 4 2x2 blocks of C.
    
    # This means the W mapping for M1 is:
    # The 4 outputs of the 2x2x2 kernel (which form M1) are added to:
    # C11 (rows 0-3), C12 (rows 4-7), C21 (rows 8-11), C22 (rows 12-15).
    # So for each of the 4 rows in W_2, we add 1 to the corresponding row in all 4 C blocks.
    
    # Let's implement a helper to add a 2x2x2 kernel instance.
    
    def add_kernel(U, V, W, offset, a_map, b_map, c_map, coeff=1.0):
        """
        offset: column offset in U, V, W (0..48).
        a_map: dict mapping 2x2 kernel row (0-3) to 4x4 U row.
        b_map: dict mapping 2x2 kernel row (0-3) to 4x4 V row.
        c_map: dict mapping 2x2 kernel row (0-3) to list of (4x4 W row, coeff).
        """
        for k in range(7):
            col = offset + k
            # U
            for r2 in range(4):
                if U_2[r2, k] != 0:
                    # a_map[r2] is a row index or list of (row, coeff)
                    # For now, assume simple mapping
                    pass # Logic below
            
        # Simpler: just loop over the 7 columns of U_2, V_2, W_2
        for k in range(7):
            col = offset + k
            for r2 in range(4):
                u_val = U_2[r2, k]
                if u_val != 0:
                    # a_map[r2] should give the 4x4 row index
                    # But we need to handle sums like A11+A22
                    # So a_map should be a function or a list of (row, coeff)
                    pass

    # This is getting complex to generalize. Let's hardcode the 7 mappings.
    
    # M1: A_v = A11 + A22, B_v = B11 + B22.
    # A11 rows: 0,1,2,3. A22 rows: 12,13,14,15.
    # Kernel A row 0 (A11) -> 4x4 rows 0,12
    # Kernel A row 1 (A12) -> None (0)
    # Kernel A row 2 (A21) -> None (0)
    # Kernel A row 3 (A22) -> 4x4 rows 3,15? No.
    # A11 is 2x2. Rows 0,1,2,3.
    # A22 is 2x2. Rows 12,13,14,15.
    # A_v = A11 + A22.
    # Kernel A row 0 corresponds to A_v[0,0] = A11[0,0] + A22[0,0].
    # So U[0, col] += 1, U[12, col] += 1.
    # Kernel A row 1 corresponds to A_v[0,1] = A11[0,1] + A22[0,1].
    # So U[1, col] += 1, U[13, col] += 1.
    # Kernel A row 2 corresponds to A_v[1,0] = A11[1,0] + A22[1,0].
    # So U[2, col] += 1, U[14, col] += 1.
    # Kernel A row 3 corresponds to A_v[1,1] = A11[1,1] + A22[1,1].
    # So U[3, col] += 1, U[15, col] += 1.
    
    # Same for V.
    
    # W mapping:
    # Kernel W row 0 (C11) -> C11[0,0], C12[0,0], C21[0,0], C22[0,0]?
    # M1 is 2x2.
    # M1[0,0] is added to C11[0,0], C12[0,0], C21[0,0], C22[0,0].
    # C11[0,0] is W row 0.
    # C12[0,0] is W row 4.
    # C21[0,0] is W row 8.
    # C22[0,0] is W row 12.
    # So W[0, col] += 1, W[4, col] += 1, W[8, col] += 1, W[12, col] += 1.
    
    # This pattern holds for all M_i.
    # We just need to define the A and B mappings for each M_i.
    
    mappings = [
        # M1: A11+A22, B11+B22
        {'a': [(0,1),(12,1), (1,1),(13,1), (2,1),(14,1), (3,1),(15,1)],
         'b': [(0,1),(12,1), (1,1),(13,1), (2,1),(14,1), (3,1),(15,1)],
         'c': [(0,1),(4,1),(8,1),(12,1), (1,1),(5,1),(9,1),(13,1), (2,1),(6,1),(10,1),(14,1), (3,1),(7,1),(11,1),(15,1)]},
        # M2: A21+A22, B11
        {'a': [(8,1),(12,1), (9,1),(13,1), (10,1),(14,1), (11,1),(15,1)],
         'b': [(0,1), (1,1), (2,1), (3,1)],
         'c': [(8,1),(12,1), (9,1),(13,1), (10,1),(14,1), (11,1),(15,1)]}, # C21, C22
        # M3: A11, B12-B22
        {'a': [(0,1), (1,1), (2,1), (3,1)],
         'b': [(4,1),(12,-1), (5,1),(13,-1), (6,1),(14,-1), (7,1),(15,-1)],
         'c': [(4,1),(12,1), (5,1),(13,1), (6,1),(14,1), (7,1),(15,1)]}, # C12, C22
        # M4: A22, B21-B11
        {'a': [(12,1), (13,1), (14,1), (15,1)],
         'b': [(8,1),(0,-1), (9,1),(1,-1), (10,1),(2,-1), (11,1),(3,-1)],
         'c': [(0,1),(12,1), (1,1),(13,1), (2,1),(14,1), (3,1),(15,1)]}, # C11, C22
        # M5: A11+A12, B22
        {'a': [(0,1),(4,1), (1,1),(5,1), (2,1),(6,1), (3,1),(7,1)],
         'b': [(12,1), (13,1), (14,1), (15,1)],
         'c': [(0,-1),(8,1), (1,-1),(9,1), (2,-1),(10,1), (3,-1),(11,1)]}, # C11(-), C21(+)
        # M6: A21-A11, B11+B12
        {'a': [(8,1),(0,-1), (9,1),(1,-1), (10,1),(2,-1), (11,1),(3,-1)],
         'b': [(0,1),(4,1), (1,1),(5,1), (2,1),(6,1), (3,1),(7,1)],
         'c': [(8,1),(12,-1), (9,1),(13,-1), (10,1),(14,-1), (11,1),(15,-1)]}, # C21(+), C22(-)
        # M7: A12-A22, B21+B22
        {'a': [(4,1),(12,-1), (5,1),(13,-1), (6,1),(14,-1), (7,1),(15,-1)],
         'b': [(8,1),(12,1), (9,1),(13,1), (10,1),(14,1), (11,1),(15,1)],
         'c': [(0,1), (1,1), (2,1), (3,1)]}, # C11(+)
    ]
    
    for i, m in enumerate(mappings):
        offset = i * 7
        for k in range(7):
            col = offset + k
            # U
            for r2 in range(4):
                u_val = U_2[r2, k]
                if u_val != 0:
                    # Find mapping for r2
                    # r2=0 -> a_map[0], r2=1 -> a_map[1], etc.
                    # My mappings above are lists of (row, coeff) for the 4 kernel rows.
                    # Wait, the mapping list above has 8 entries for M1.
                    # It should have 4 entries, each being a list of (row, coeff).
                    
                    # Let's restructure mappings to be:
                    # a: list of 4 items, each is list of (row, coeff)
                    # b: list of 4 items, each is list of (row, coeff)
                    # c: list of 4 items, each is list of (row, coeff)
                    
                    pass
    
    # Re-doing mappings structure
    mappings = [
        # M1
        {'a': [[(0,1),(12,1)], [(1,1),(13,1)], [(2,1),(14,1)], [(3,1),(15,1)]],
         'b': [[(0,1),(12,1)], [(1,1),(13,1)], [(2,1),(14,1)], [(3,1),(15,1)]],
         'c': [[(0,1),(4,1),(8,1),(12,1)], [(1,1),(5,1),(9,1),(13,1)], [(2,1),(6,1),(10,1),(14,1)], [(3,1),(7,1),(11,1),(15,1)]]},
        # M2
        {'a': [[(8,1),(12,1)], [(9,1),(13,1)], [(10,1),(14,1)], [(11,1),(15,1)]],
         'b': [[(0,1)], [(1,1)], [(2,1)], [(3,1)]],
         'c': [[(8,1),(12,1)], [(9,1),(13,1)], [(10,1),(14,1)], [(11,1),(15,1)]]},
        # M3
        {'a': [[(0,1)], [(1,1)], [(2,1)], [(3,1)]],
         'b': [[(4,1),(12,-1)], [(5,1),(13,-1)], [(6,1),(14,-1)], [(7,1),(15,-1)]],
         'c': [[(4,1),(12,1)], [(5,1),(13,1)], [(6,1),(14,1)], [(7,1),(15,1)]]},
        # M4
        {'a': [[(12,1)], [(13,1)], [(14,1)], [(15,1)]],
         'b': [[(8,1),(0,-1)], [(9,1),(1,-1)], [(10,1),(2,-1)], [(11,1),(3,-1)]],
         'c': [[(0,1),(12,1)], [(1,1),(13,1)], [(2,1),(14,1)], [(3,1),(15,1)]]},
        # M5
        {'a': [[(0,1),(4,1)], [(1,1),(5,1)], [(2,1),(6,1)], [(3,1),(7,1)]],
         'b': [[(12,1)], [(13,1)], [(14,1)], [(15,1)]],
         'c': [[(0,-1),(8,1)], [(1,-1),(9,1)], [(2,-1),(10,1)], [(3,-1),(11,1)]]},
        # M6
        {'a': [[(8,1),(0,-1)], [(9,1),(1,-1)], [(10,1),(2,-1)], [(11,1),(3,-1)]],
         'b': [[(0,1),(4,1)], [(1,1),(5,1)], [(2,1),(6,1)], [(3,1),(7,1)]],
         'c': [[(8,1),(12,-1)], [(9,1),(13,-1)], [(10,1),(14,-1)], [(11,1),(15,-1)]]},
        # M7
        {'a': [[(4,1),(12,-1)], [(5,1),(13,-1)], [(6,1),(14,-1)], [(7,1),(15,-1)]],
         'b': [[(8,1),(12,1)], [(9,1),(13,1)], [(10,1),(14,1)], [(11,1),(15,1)]],
         'c': [[(0,1)], [(1,1)], [(2,1)], [(3,1)]]},
    ]
    
    for i, m in enumerate(mappings):
        offset = i * 7
        for k in range(7):
            col = offset + k
            for r2 in range(4):
                # U
                u_val = U_2[r2, k]
                if u_val != 0:
                    for row, coeff in m['a'][r2]:
                        U[row, col] += u_val * coeff
                # V
                v_val = V_2[r2, k]
                if v_val != 0:
                    for row, coeff in m['b'][r2]:
                        V[row, col] += v_val * coeff
                # W
                w_val = W_2[r2, k]
                if w_val != 0:
                    for row, coeff in m['c'][r2]:
                        W[row, col] += w_val * coeff
                        
    return U, V, W

def _build_schoolbook_4x4x1():
    """
    Returns U, V, W for 4x4x1 matrix multiplication with rank 16.
    """
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
            
    return U, V, W

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Build components
    U_s4, V_s4, W_s4 = _build_strassen_4x4x4()
    U_s1, V_s1, W_s1 = _build_schoolbook_4x4x1()
    
    # Combine for 4x4x9
    # B is 4x9. Split into B1 (4x4), B2 (4x4), B3 (4x1).
    # C is 4x9. Split into C1 (4x4), C2 (4x4), C3 (4x1).
    
    U_total = np.zeros((16, 49 + 49 + 16))
    V_total = np.zeros((4, 49 + 49 + 16))
    W_total = np.zeros((4, 49 + 49 + 16))
    
    # Part 1: A * B1 (4x4 * 4x4)
    # U: 16x49
    # V: 4x49 (B1 is cols 0-3 of B)
    # W: 4x49 (C1 is cols 0-3 of C? No, C is 4x9. C1 is 4x4. Rows 0-3 of C? No, C is 4x9. C1 is the first 4 columns.)
    # W index is k*n + i. k is col of C (0-3 for C1). i is row of C (0-3).
    # So W rows for C1 are 0..15.
    
    U_total[:, 0:49] = U_s4
    V_total[:, 0:49] = V_s4
    W_total[:, 0:49] = W_s4
    
    # Part 2: A * B2 (4x4 * 4x4)
    # B2 is cols 4-7 of B.
    # V index is j*p + k. j is row of B (0-3). k is col of B (4-7).
    # So V rows for B2 are 4, 8, 12, 16? No.
    # V shape is (m*p, rank) = (4*9, rank) = (36, rank).
    # My V_s4 is 4x49. It assumes B is 4x4.
    # I need to map V_s4 rows (0-3) to V_total rows (4, 8, 12, 16)?
    # No, V row index is j*p + k.
    # For B1 (k=0-3), rows are 0, 4, 8, 12? No.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # Wait, V index is j*p + k.
    # For 4x9, p=9.
    # j=0, k=0 -> 0.
    # j=0, k=1 -> 1.
    # ...
    # j=0, k=8 -> 8.
    # j=1, k=0 -> 9.
    
    # My V_s4 is for 4x4 B. Rows 0-3 correspond to j=0, k=0..3?
    # No, in _build_strassen_2x2x2, V shape is (4, 7).
    # V index j*p + k. p=2.
    # j=0, k=0 -> 0.
    # j=0, k=1 -> 1.
    # j=1, k=0 -> 2.
    # j=1, k=1 -> 3.
    
    # So V_s4 rows 0-3 correspond to B1 cols 0-3 for j=0,1?
    # No, B is 4x4. j=0..3, k=0..3.
    # V rows:
    # 0: j=0, k=0
    # 1: j=0, k=1
    # 2: j=1, k=0
    # 3: j=1, k=1
    # Wait, 4x4 B has 16 entries. V should be 16 rows.
    # My _build_strassen_4x4x4 returns V of shape (16, 49).
    # Yes, V_s4 is 16x49.
    
    # So for Part 1 (B1, cols 0-3 of B_total):
    # B1 is 4x4.
    # V_s4 rows 0-15 correspond to B1 entries.
    # B1 entry (j, k) maps to V_total row j*9 + k.
    # j=0, k=0 -> 0.
    # j=0, k=1 -> 1.
    # j=0, k=2 -> 2.
    # j=0, k=3 -> 3.
    # j=1, k=0 -> 9.
    # ...
    
    # So I need to map V_s4 rows to V_total rows.
    # V_s4 row r corresponds to B1[j, k].
    # How is V_s4 indexed?
    # In _build_strassen_4x4x4, V is built from 2x2x2 kernels.
    # The 2x2x2 kernel V is 4x7.
    # The mapping in _build_strassen_4x4x4 maps kernel rows to 4x4 B rows.
    # The 4x4 B rows are 0-15.
    # Row 0: j=0, k=0.
    # Row 1: j=0, k=1.
    # Row 2: j=1, k=0.
    # Row 3: j=1, k=1.
    # ...
    # Row 15: j=3, k=3.
    
    # So V_s4 row r corresponds to B1[j, k] where r = j*4 + k?
    # No, standard row-major.
    # r = j*4 + k.
    # V_total row = j*9 + k.
    
    # Let's create a mapping array.
    v_map_1 = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            r = j*4 + k
            v_map_1[r] = j*9 + k
            
    V_total[:, 0:49] = V_s4[v_map_1, :]
    
    # W for Part 1:
    # W_s4 is 16x49.
    # W index k*n + i.
    # C1 is 4x4. k=0..3, i=0..3.
    # W_s4 row r corresponds to C1[k, i] where r = k*4 + i.
    # W_total row = k*4 + i. (Since C is 4x9, n=4. W row = k*4 + i).
    # So W_total rows 0-15 are the same as W_s4.
    
    W_total[:, 0:49] = W_s4
    
    # Part 2: A * B2 (4x4 * 4x4)
    # B2 is cols 4-7 of B.
    # V_s4 rows correspond to B2[j, k'] where k'=0..3.
    # V_total row = j*9 + (4 + k').
    
    v_map_2 = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            r = j*4 + k
            v_map_2[r] = j*9 + (4 + k)
            
    V_total[:, 49:98] = V_s4[v_map_2, :]
    
    # W for Part 2:
    # C2 is 4x4. k=4..7.
    # W_s4 row r corresponds to C2[k', i] where k'=0..3.
    # W_total row = (4 + k')*4 + i.
    
    w_map_2 = np.zeros(16, dtype=int)
    for k in range(4):
        for i in range(4):
            r = k*4 + i
            w_map_2[r] = (4 + k)*4 + i
            
    W_total[:, 49:98] = W_s4[w_map_2, :]
    
    # Part 3: A * B3 (4x4 * 4x1)
    # B3 is col 8 of B.
    # U_s1 is 16x16.
    # V_s1 is 4x16.
    # W_s1 is 4x16.
    
    U_total[:, 98:114] = U_s1
    
    # V_s1 row j corresponds to B3[j, 0].
    # V_total row = j*9 + 8.
    
    v_map_3 = np.zeros(4, dtype=int)
    for j in range(4):
        v_map_3[j] = j*9 + 8
        
    V_total[:, 98:114] = V_s1[v_map_3, :]
    
    # W_s1 row i corresponds to C3[i, 0].
    # C3 is col 8 of C.
    # W_total row = 8*4 + i.
    
    w_map_3 = np.zeros(4, dtype=int)
    for i in range(4):
        w_map_3[i] = 8*4 + i
        
    W_total[:, 98:114] = W_s1[w_map_3, :]
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
