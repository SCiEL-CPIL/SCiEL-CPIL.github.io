"""Matrix multiplication 3x5x6 via Strassen decomposition for 2x2x2 sub-blocks."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _build_strassen_2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where W uses trace convention W[k*2+i, r].
    
    Strassen's 7 products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U: rows indexed by (i,j) for A, i in {0,1}, j in {0,1}; U[i*2+j, r]
    # V: rows indexed by (j,k) for B, j in {0,1}, k in {0,1}; V[j*2+k, r]
    # W: rows indexed by (k,i) for C, k in {0,1}, i in {0,1}; W[k*2+i, r]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00 (k=0,i=0)
    W[3, 0] = 1  # C11 (k=1,i=1)
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 (k=1,i=0)
    W[3, 1] = -1 # C11 (k=1,i=1)
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01 (k=0,i=1)
    W[3, 2] = 1  # C11 (k=1,i=1)
    
    # M4 = A11(B01 - B00)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00 (k=0,i=0)
    W[2, 3] = 1  # C10 (k=1,i=0)
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00 (k=0,i=0)
    W[1, 4] = 1  # C01 (k=0,i=1)
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11 (k=1,i=1)
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00 (k=0,i=0)
    
    return U, V, W

def construct():
    """
    Construct a rank-84 tensor decomposition for 3x5x6 matrix multiplication
    using Strassen's algorithm on 2x2x2 sub-blocks.
    
    n=3, m=5, p=6. We decompose into 2x2x2 blocks where possible.
    The 3x5x6 tensor can be seen as having 2x2x2 blocks in various positions.
    
    Strategy: Use Strassen for 2x2x2 subproblems.
    For 3x5x6:
    - The first dimension (n=3) can be split as 2+1
    - The middle dimension (m=5) can be split as 2+2+1 or 2+1+2 etc.
    - The last dimension (p=6) can be split as 2+2+2
    
    We'll use a systematic approach:
    For each pair of 2x2 blocks in A and B that produce a 2x2 block in C,
    apply Strassen. For remaining 1x1 or mixed blocks, use schoolbook.
    
    Let's think about this more carefully.
    A is 3x5, B is 5x6, C is 3x6.
    
    We can partition:
    A rows: [0,1] and [2]  (2+1)
    A cols / B rows: [0,1], [2,3], [4]  (2+2+1)
    B cols / C cols: [0,1], [2,3], [4,5]  (2+2+2)
    
    For each combination of (A row block, A col block, B col block):
    - If A row block is 2, A col block is 2, B col block is 2: use Strassen (rank 7)
    - Otherwise: use schoolbook (rank = product of sizes)
    
    Number of 2x2x2 blocks: 
    A row blocks of size 2: 1 choice ([0,1])
    A col blocks of size 2: 2 choices ([0,1], [2,3])
    B col blocks of size 2: 3 choices ([0,1], [2,3], [4,5])
    So 1*2*3 = 6 Strassen blocks, each rank 7 = 42
    
    Remaining schoolbook terms:
    - A row [2] (size 1) with any A col block and B col block:
      A col blocks: 3 choices, B col blocks: 3 choices
      Each is 1 x size(A col) x size(B col)
      [0,1]x[0,1]: 1*2*2=4
      [0,1]x[2,3]: 1*2*2=4
      [0,1]x[4,5]: 1*2*2=4
      [2,3]x[0,1]: 1*2*2=4
      [2,3]x[2,3]: 1*2*2=4
      [2,3]x[4,5]: 1*2*2=4
      [4]x[0,1]: 1*1*2=2
      [4]x[2,3]: 1*1*2=2
      [4]x[4,5]: 1*1*2=2
      Total for A row [2]: 4+4+4+4+4+4+2+2+2 = 30
    
    - A row [0,1] (size 2) with A col [4] (size 1) and any B col block:
      B col blocks: 3 choices
      Each is 2 x 1 x size(B col)
      [0,1]: 2*1*2=4
      [2,3]: 2*1*2=4
      [4,5]: 2*1*2=4
      Total: 12
    
    - A row [0,1] (size 2) with A col [0,1] or [2,3] (size 2) and B col [4] (size 1):
      A col blocks: 2 choices
      B col [4,5] is size 2, not 1. Wait, B col blocks are [0,1], [2,3], [4,5], all size 2.
      So there's no B col block of size 1. Let me re-examine.
    
    Actually, I need to be more careful. The B col blocks are all size 2.
    So the only "mixed" cases are:
    - A row size 1 with any A col and B col
    - A row size 2, A col size 1, B col size 2
    - A row size 2, A col size 2, B col size 2 (Strassen)
    
    Let me recount:
    A row blocks: [0,1] (size 2), [2] (size 1)
    A col / B row blocks: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    B col / C col blocks: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2)
    
    For each (a_row, a_col, b_col):
    - (2,2,2): Strassen, rank 7. Count: 1*2*3=6, total 42
    - (1,*,*): schoolbook. a_row=[2]
      - (1,2,2): 1*2*2=4. Count: 1*2*3=6, total 24
      - (1,1,2): 1*1*2=2. Count: 1*1*3=3, total 6
    - (2,1,2): schoolbook. a_row=[0,1], a_col=[4]
      - Count: 1*1*3=3, each 2*1*2=4, total 12
    
    Total rank = 42 + 24 + 6 + 12 = 84
    """
    n, m, p = DIMENSIONS
    
    # Define partitions
    # A rows: [0,1], [2]
    a_row_blocks = [(0, 2), (2, 3)]  # (start, end)
    # A cols / B rows: [0,1], [2,3], [4]
    a_col_blocks = [(0, 2), (2, 4), (4, 5)]
    # B cols / C cols: [0,1], [2,3], [4,5]
    b_col_blocks = [(0, 2), (2, 4), (4, 6)]
    
    U_rows = n * m  # 15
    V_rows = m * p  # 30
    W_rows = p * n  # 18
    
    # We'll build the decomposition by accumulating contributions
    # Use lists of contributions to each row of U, V, W
    U_contribs = [dict() for _ in range(U_rows)]
    V_contribs = [dict() for _ in range(V_rows)]
    W_contribs = [dict() for _ in range(W_rows)]
    
    col_offset = 0
    
    for a_r_start, a_r_end in a_row_blocks:
        a_r_size = a_r_end - a_r_start
        for a_c_start, a_c_end in a_col_blocks:
            a_c_size = a_c_end - a_c_start
            for b_c_start, b_c_end in b_col_blocks:
                b_c_size = b_c_end - b_c_start
                
                if a_r_size == 2 and a_c_size == 2 and b_c_size == 2:
                    # Strassen block
                    S_U, S_V, S_W = _build_strassen_2x2()
                    for r in range(7):
                        # U: for each row in A block
                        for i in range(2):
                            for j in range(2):
                                val = S_U[i * 2 + j, r]
                                if val != 0:
                                    u_idx = (a_r_start + i) * m + (a_c_start + j)
                                    U_contribs[u_idx][col_offset + r] = U_contribs[u_idx].get(col_offset + r, 0) + val
                        # V: for each row in B block
                        for j in range(2):
                            for k in range(2):
                                val = S_V[j * 2 + k, r]
                                if val != 0:
                                    v_idx = (a_c_start + j) * p + (b_c_start + k)
                                    V_contribs[v_idx][col_offset + r] = V_contribs[v_idx].get(col_offset + r, 0) + val
                        # W: for each row in C block
                        for k in range(2):
                            for i in range(2):
                                val = S_W[k * 2 + i, r]
                                if val != 0:
                                    w_idx = (b_c_start + k) * n + (a_r_start + i)
                                    W_contribs[w_idx][col_offset + r] = W_contribs[w_idx].get(col_offset + r, 0) + val
                    col_offset += 7
                else:
                    # Schoolbook block
                    for i in range(a_r_size):
                        for j in range(a_c_size):
                            for k in range(b_c_size):
                                u_idx = (a_r_start + i) * m + (a_c_start + j)
                                v_idx = (a_c_start + j) * p + (b_c_start + k)
                                w_idx = (b_c_start + k) * n + (a_r_start + i)
                                U_contribs[u_idx][col_offset] = 1
                                V_contribs[v_idx][col_offset] = 1
                                W_contribs[w_idx][col_offset] = 1
                                col_offset += 1
    
    rank = col_offset
    
    # Build final matrices
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for i in range(U_rows):
        for col, val in U_contribs[i].items():
            U[i][col] = val
    for i in range(V_rows):
        for col, val in V_contribs[i].items():
            V[i][col] = val
    for i in range(W_rows):
        for col, val in W_contribs[i].items():
            W[i][col] = val
    
    return {"U": U, "V": V, "W": W}
