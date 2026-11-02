"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns U, V, W as lists of (index, value) for each rank term.
    A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    B indices: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7
    C indices: (0,0)=8, (0,1)=9, (1,0)=10, (1,1)=11
    """
    # Strassen's 7 products
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    U = []  # each element: list of (a_index, coeff)
    V = []  # each element: list of (b_index, coeff)
    W = []  # each element: list of (c_index, coeff)
    
    # P1 = (a00+a11)(b00+b11)
    U.append([(0,1),(2,1)])
    V.append([(4,1),(7,1)])
    W.append([(8,1),(11,-1)])  # contributes to C00 and C11
    
    # P2 = (a10+a11)b00
    U.append([(2,1),(3,1)])
    V.append([(4,1)])
    W.append([(10,1),(11,-1)])  # C10 and C11
    
    # P3 = a00(b01-b11)
    U.append([(0,1)])
    V.append([(5,1),(7,-1)])
    W.append([(9,1),(11,1)])  # C01 and C11
    
    # P4 = a11(b10-b00)
    U.append([(3,1)])
    V.append([(6,1),(4,-1)])
    W.append([(8,1),(10,1)])  # C00 and C10
    
    # P5 = (a00+a01)b11
    U.append([(0,1),(1,1)])
    V.append([(7,1)])
    W.append([(8,-1),(9,1)])  # C00 and C01
    
    # P6 = (a10-a00)(b00+b01)
    U.append([(2,1),(0,-1)])
    V.append([(4,1),(5,1)])
    W.append([(11,1)])  # C11
    
    # P7 = (a01-a11)(b10+b11)
    U.append([(1,1),(3,-1)])
    V.append([(6,1),(7,1)])
    W.append([(8,1)])  # C00
    
    return U, V, W

def _schoolbook_2x2x2x3():
    """Schoolbook for 2x2 * 2x3 -> 2x3. Rank 6.
    A: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3
    B: (0,0)=4,(0,1)=5,(0,2)=6,(1,0)=7,(1,1)=8,(1,2)=9
    C: (0,0)=10,(0,1)=11,(0,2)=12,(1,0)=13,(1,1)=14,(1,2)=15
    """
    U = []
    V = []
    W = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U.append([(i*2+j, 1)])
                V.append([(j*3+k, 1)])
                W.append([(i*3+k, 1)])
    return U, V, W

def _schoolbook_2x3x3x2():
    """Schoolbook for 2x3 * 3x2 -> 2x2. Rank 6.
    A: (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5
    B: (0,0)=6,(0,1)=7,(1,0)=8,(1,1)=9,(2,0)=10,(2,1)=11
    C: (0,0)=12,(0,1)=13,(1,0)=14,(1,1)=15
    """
    U = []
    V = []
    W = []
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U.append([(i*3+j, 1)])
                V.append([(j*2+k, 1)])
                W.append([(i*2+k, 1)])
    return U, V, W

def _schoolbook_2x3x3x3():
    """Schoolbook for 2x3 * 3x3 -> 2x3. Rank 9.
    A: (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5
    B: (0,0)=6,(0,1)=7,(0,2)=8,(1,0)=9,(1,1)=10,(1,2)=11,(2,0)=12,(2,1)=13,(2,2)=14
    C: (0,0)=15,(0,1)=16,(0,2)=17,(1,0)=18,(1,1)=19,(1,2)=20
    """
    U = []
    V = []
    W = []
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U.append([(i*3+j, 1)])
                V.append([(j*3+k, 1)])
                W.append([(i*3+k, 1)])
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Partition:
    # A (4x5): rows split 2+2, cols split 2+3
    #   A11: rows 0-1, cols 0-1 (2x2)
    #   A12: rows 0-1, cols 2-4 (2x3)
    #   A21: rows 2-3, cols 0-1 (2x2)
    #   A22: rows 2-3, cols 2-4 (2x3)
    # B (5x5): rows split 2+3, cols split 2+3
    #   B11: rows 0-1, cols 0-1 (2x2)
    #   B12: rows 0-1, cols 2-4 (2x3)
    #   B21: rows 2-4, cols 0-1 (3x2)
    #   B22: rows 2-4, cols 2-4 (3x3)
    # C (4x5): rows split 2+2, cols split 2+3
    #   C11: rows 0-1, cols 0-1 (2x2)
    #   C12: rows 0-1, cols 2-4 (2x3)
    #   C21: rows 2-3, cols 0-1 (2x2)
    #   C22: rows 2-3, cols 2-4 (2x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We'll build the full U, V, W matrices by concatenating sub-decompositions
    # with appropriate index mappings.
    
    # Global index mappings:
    # A global indices: row*5 + col, so 0..19
    # B global indices: row*5 + col, so 0..24
    # C global indices: row*5 + col, so 0..19 (but W uses k*n+i convention: col*4+row)
    
    # Wait, the output factor W follows W[k*n+i, r] where k is column of C, i is row of C.
    # So W has p*n = 5*4 = 20 rows.
    
    # Let me define local-to-global index mappings for each sub-block.
    
    # For A11 (2x2): local (i,j) -> global (i, j) -> index i*5+j
    # For A12 (2x3): local (i,j) -> global (i, j+2) -> index i*5+(j+2)
    # For A21 (2x2): local (i,j) -> global (i+2, j) -> index (i+2)*5+j
    # For A22 (2x3): local (i,j) -> global (i+2, j+2) -> index (i+2)*5+(j+2)
    
    # For B11 (2x2): local (i,j) -> global (i, j) -> index i*5+j
    # For B12 (2x3): local (i,j) -> global (i, j+2) -> index i*5+(j+2)
    # For B21 (3x2): local (i,j) -> global (i+2, j) -> index (i+2)*5+j
    # For B22 (3x3): local (i,j) -> global (i+2, j+2) -> index (i+2)*5+(j+2)
    
    # For C11 (2x2): local (i,j) -> global (i, j) -> W index j*4+i
    # For C12 (2x3): local (i,j) -> global (i, j+2) -> W index (j+2)*4+i
    # For C21 (2x2): local (i,j) -> global (i+2, j) -> W index j*4+(i+2)
    # For C22 (2x3): local (i,j) -> global (i+2, j+2) -> W index (j+2)*4+(i+2)
    
    all_U = []  # list of dicts: {global_a_index: coeff}
    all_V = []  # list of dicts: {global_b_index: coeff}
    all_W = []  # list of dicts: {global_w_index: coeff}
    
    def add_decomposition(U_local, V_local, W_local, a_map, b_map, w_map):
        """Add a local decomposition to the global one using index mappings."""
        for u_terms, v_terms, w_terms in zip(U_local, V_local, W_local):
            u_dict = {}
            for local_idx, coeff in u_terms:
                global_idx = a_map(local_idx)
                u_dict[global_idx] = u_dict.get(global_idx, 0) + coeff
            v_dict = {}
            for local_idx, coeff in v_terms:
                global_idx = b_map(local_idx)
                v_dict[global_idx] = v_dict.get(global_idx, 0) + coeff
            w_dict = {}
            for local_idx, coeff in w_terms:
                global_idx = w_map(local_idx)
                w_dict[global_idx] = w_dict.get(global_idx, 0) + coeff
            all_U.append(u_dict)
            all_V.append(v_dict)
            all_W.append(w_dict)
    
    # Define index mappings
    def a11_map(idx):  # 2x2, local idx = i*2+j
        i, j = divmod(idx, 2)
        return i*5 + j
    
    def a12_map(idx):  # 2x3, local idx = i*3+j
        i, j = divmod(idx, 3)
        return i*5 + (j+2)
    
    def a21_map(idx):  # 2x2, local idx = i*2+j
        i, j = divmod(idx, 2)
        return (i+2)*5 + j
    
    def a22_map(idx):  # 2x3, local idx = i*3+j
        i, j = divmod(idx, 3)
        return (i+2)*5 + (j+2)
    
    def b11_map(idx):  # 2x2, local idx = i*2+j
        i, j = divmod(idx, 2)
        return i*5 + j
    
    def b12_map(idx):  # 2x3, local idx = i*3+j
        i, j = divmod(idx, 3)
        return i*5 + (j+2)
    
    def b21_map(idx):  # 3x2, local idx = i*2+j
        i, j = divmod(idx, 2)
        return (i+2)*5 + j
    
    def b22_map(idx):  # 3x3, local idx = i*3+j
        i, j = divmod(idx, 3)
        return (i+2)*5 + (j+2)
    
    def c11_map(idx):  # 2x2, local idx = i*2+j (i=row, j=col), W index = j*4+i
        i, j = divmod(idx, 2)
        return j*4 + i
    
    def c12_map(idx):  # 2x3, local idx = i*3+j, W index = (j+2)*4+i
        i, j = divmod(idx, 3)
        return (j+2)*4 + i
    
    def c21_map(idx):  # 2x2, local idx = i*2+j, W index = j*4+(i+2)
        i, j = divmod(idx, 2)
        return j*4 + (i+2)
    
    def c22_map(idx):  # 2x3, local idx = i*3+j, W index = (j+2)*4+(i+2)
        i, j = divmod(idx, 3)
        return (j+2)*4 + (i+2)
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2 * 2x2 -> 2x2, use Strassen
    U1, V1, W1 = _strassen_2x2()
    add_decomposition(U1, V1, W1, a11_map, b11_map, c11_map)
    
    # A12*B21: 2x3 * 3x2 -> 2x2, use schoolbook
    U2, V2, W2 = _schoolbook_2x3x3x2()
    add_decomposition(U2, V2, W2, a12_map, b21_map, c11_map)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2 * 2x3 -> 2x3, use schoolbook
    U3, V3, W3 = _schoolbook_2x2x2x3()
    add_decomposition(U3, V3, W3, a11_map, b12_map, c12_map)
    
    # A12*B22: 2x3 * 3x3 -> 2x3, use schoolbook
    U4, V4, W4 = _schoolbook_2x3x3x3()
    add_decomposition(U4, V4, W4, a12_map, b22_map, c12_map)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 2x2 * 2x2 -> 2x2, use Strassen
    U5, V5, W5 = _strassen_2x2()
    add_decomposition(U5, V5, W5, a21_map, b11_map, c21_map)
    
    # A22*B21: 2x3 * 3x2 -> 2x2, use schoolbook
    U6, V6, W6 = _schoolbook_2x3x3x2()
    add_decomposition(U6, V6, W6, a22_map, b21_map, c21_map)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 2x2 * 2x3 -> 2x3, use schoolbook
    U7, V7, W7 = _schoolbook_2x2x2x3()
    add_decomposition(U7, V7, W7, a21_map, b12_map, c22_map)
    
    # A22*B22: 2x3 * 3x3 -> 2x3, use schoolbook
    U8, V8, W8 = _schoolbook_2x3x3x3()
    add_decomposition(U8, V8, W8, a22_map, b22_map, c22_map)
    
    rank = len(all_U)
    
    # Convert to matrix form
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for idx, coeff in all_U[col].items():
            U_mat[idx][col] = coeff
        for idx, coeff in all_V[col].items():
            V_mat[idx][col] = coeff
        for idx, coeff in all_W[col].items():
            W_mat[idx][col] = coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
