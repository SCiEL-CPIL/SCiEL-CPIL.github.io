"""Construct a rank-31 tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 31
    
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(9):
            U[i][col] = u_vec[i]
        for j in range(12):
            V[j][col] = v_vec[j]
        for k in range(12):
            W[k][col] = w_vec[k]
        col += 1
        
    # --- Part 1: 3x3x3 Decomposition (Rank 22) ---
    
    # Indices:
    # A: i,j in 0..2
    # B: j,k in 0..2
    # C: i,k in 0..2
    
    # Block 11: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Note: A indices are (i,j). B indices are (j,k).
    # A00 is A[0,0], A11 is A[1,1], etc.
    # But in Strassen, A is a 2x2 matrix.
    # A_sub = [[A00, A01], [A10, A11]]
    
    # Term 1: M1
    # U: A00, A11
    # V: B00, B11
    # W: C00, C01, C10, C11 (with signs)
    # C00 += M1, C01 += M1, C10 += M1, C11 -= M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We add terms one by one.
    
    # M1: U has 1 at (0,0) and (1,1). V has 1 at (0,0) and (1,1).
    # W contributions:
    # C00 (0,0): +1
    # C01 (0,1): +1
    # C10 (1,0): +1
    # C11 (1,1): -1
    u = [0]*9; u[0]=1; u[3]=1
    v = [0]*12; v[0]=1; v[4]=1 # B[0,0]=v[0], B[1,1]=v[4] (j*3+k)
    w = [0]*12; w[0]=1; w[1]=1; w[3]=1; w[4]=-1 # C[0,0]=w[0], C[0,1]=w[1], C[1,0]=w[3], C[1,1]=w[4] (k*3+i)
    add_term(u, v, w)
    
    # M2: (A10+A11)B00
    # U: A[1,0], A[1,1] -> indices 3, 4
    # V: B[0,0] -> index 0
    # W:
    # C10 (1,0): +1
    # C11 (1,1): -1
    u = [0]*9; u[3]=1; u[4]=1
    v = [0]*12; v[0]=1
    w = [0]*12; w[3]=1; w[4]=-1
    add_term(u, v, w)
    
    # M3: A00(B01-B11)
    # U: A[0,0] -> index 0
    # V: B[0,1], B[1,1] -> indices 1, 4
    # W:
    # C01 (0,1): +1
    # C11 (1,1): +1
    u = [0]*9; u[0]=1
    v = [0]*12; v[1]=1; v[4]=-1
    w = [0]*12; w[1]=1; w[4]=1
    add_term(u, v, w)
    
    # M4: A11(B10-B00)
    # U: A[1,1] -> index 4
    # V: B[1,0], B[0,0] -> indices 3, 0
    # W:
    # C00 (0,0): +1
    # C10 (1,0): +1
    u = [0]*9; u[4]=1
    v = [0]*12; v[3]=1; v[0]=-1
    w = [0]*12; w[0]=1; w[3]=1
    add_term(u, v, w)
    
    # M5: (A00+A01)B11
    # U: A[0,0], A[0,1] -> indices 0, 1
    # V: B[1,1] -> index 4
    # W:
    # C00 (0,0): -1
    # C01 (0,1): +1
    u = [0]*9; u[0]=1; u[1]=1
    v = [0]*12; v[4]=1
    w = [0]*12; w[0]=-1; w[1]=1
    add_term(u, v, w)
    
    # M6: (A10-A00)(B00+B01)
    # U: A[1,0], A[0,0] -> indices 3, 0
    # V: B[0,0], B[0,1] -> indices 0, 1
    # W:
    # C11 (1,1): +1
    u = [0]*9; u[3]=1; u[0]=-1
    v = [0]*12; v[0]=1; v[1]=1
    w = [0]*12; w[4]=1
    add_term(u, v, w)
    
    # M7: (A01-A11)(B10+B11)
    # U: A[0,1], A[1,1] -> indices 1, 4
    # V: B[1,0], B[1,1] -> indices 3, 4
    # W:
    # C00 (0,0): +1
    u = [0]*9; u[1]=1; u[4]=-1
    v = [0]*12; v[3]=1; v[4]=1
    w = [0]*12; w[0]=1
    add_term(u, v, w)
    
    # Block 12: A[0:2, 0:2] * B[0:2, 2] + A[0:2, 2] * B[2, 2]
    # A11_sub * B12_sub + A12_sub * B22_sub
    # A11_sub is 2x2, B12_sub is 2x1. Rank 4.
    # Terms:
    # 1. A[0,0] * B[0,2] -> C[0,2]
    # 2. A[0,1] * B[1,2] -> C[0,2]
    # 3. A[1,0] * B[0,2] -> C[1,2]
    # 4. A[1,1] * B[1,2] -> C[1,2]
    
    u = [0]*9; u[0]=1
    v = [0]*12; v[2]=1 # B[0,2]
    w = [0]*12; w[6]=1 # C[0,2] (k=2, i=0)
    add_term(u, v, w)
    
    u = [0]*9; u[1]=1
    v = [0]*12; v[5]=1 # B[1,2]
    w = [0]*12; w[6]=1
    add_term(u, v, w)
    
    u = [0]*9; u[3]=1
    v = [0]*12; v[2]=1
    w = [0]*12; w[9]=1 # C[1,2] (k=2, i=1)
    add_term(u, v, w)
    
    u = [0]*9; u[4]=1
    v = [0]*12; v[5]=1
    w = [0]*12; w[9]=1
    add_term(u, v, w)
    
    # A12_sub * B22_sub
    # A[0,2] * B[2,2] -> C[0,2]
    # A[1,2] * B[2,2] -> C[1,2]
    
    u = [0]*9; u[2]=1
    v = [0]*12; v[8]=1 # B[2,2]
    w = [0]*12; w[6]=1
    add_term(u, v, w)
    
    u = [0]*9; u[5]=1
    v = [0]*12; v[8]=1
    w = [0]*12; w[9]=1
    add_term(u, v, w)
    
    # Block 21: A[2, 0:2] * B[0:2, 0:2] + A[2, 2] * B[2, 0:2]
    # A21_sub * B11_sub + A22_sub * B21_sub
    # A21_sub is 1x2, B11_sub is 2x2. Rank 4.
    # 1. A[2,0] * B[0,0] -> C[2,0]
    # 2. A[2,0] * B[0,1] -> C[2,1]
    # 3. A[2,1] * B[1,0] -> C[2,0]
    # 4. A[2,1] * B[1,1] -> C[2,1]
    
    u = [0]*9; u[6]=1
    v = [0]*12; v[0]=1
    w = [0]*12; w[12]=1 # C[2,0] (k=0, i=2) -> index 0*3+2=2? No. W[k*n+i]. k=0, i=2 -> 2.
    # Wait, W index is k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=0, i=2 -> 2
    # k=1, i=0 -> 3
    # k=1, i=1 -> 4
    # k=1, i=2 -> 5
    # k=2, i=0 -> 6
    # k=2, i=1 -> 7
    # k=2, i=2 -> 8
    
    # My previous W indices were wrong?
    # C[0,0] -> k=0, i=0 -> 0. Correct.
    # C[0,1] -> k=1, i=0 -> 3. I used 1. WRONG.
    # C[1,0] -> k=0, i=1 -> 1. I used 3. WRONG.
    # C[1,1] -> k=1, i=1 -> 4. I used 4. Correct.
    
    # I need to fix the W indices in the Strassen part.
    # Let's restart the W index mapping.
    # W[k*n + i]
    
    # I will rewrite the terms with correct indices.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(9):
            U[i][col] = u_vec[i]
        for j in range(12):
            V[j][col] = v_vec[j]
        for k in range(12):
            W[k][col] = w_vec[k]
        col += 1
        
    # Strassen M1
    u = [0]*9; u[0]=1; u[3]=1
    v = [0]*12; v[0]=1; v[4]=1
    w = [0]*12
    w[0] = 1   # C[0,0]
    w[3] = 1   # C[0,1] (k=1, i=0)
    w[1] = 1   # C[1,0] (k=0, i=1)
    w[4] = -1  # C[1,1] (k=1, i=1)
    add_term(u, v, w)
    
    # Strassen M2
    u = [0]*9; u[3]=1; u[4]=1
    v = [0]*12; v[0]=1
    w = [0]*12
    w[1] = 1   # C[1,0]
    w[4] = -1  # C[1,1]
    add_term(u, v, w)
    
    # Strassen M3
    u = [0]*9; u[0]=1
    v = [0]*12; v[1]=1; v[4]=-1
    w = [0]*12
    w[3] = 1   # C[0,1]
    w[4] = 1   # C[1,1]
    add_term(u, v, w)
    
    # Strassen M4
    u = [0]*9; u[4]=1
    v = [0]*12; v[3]=1; v[0]=-1
    w = [0]*12
    w[0] = 1   # C[0,0]
    w[1] = 1   # C[1,0]
    add_term(u, v, w)
    
    # Strassen M5
    u = [0]*9; u[0]=1; u[1]=1
    v = [0]*12; v[4]=1
    w = [0]*12
    w[0] = -1  # C[0,0]
    w[3] = 1   # C[0,1]
    add_term(u, v, w)
    
    # Strassen M6
    u = [0]*9; u[3]=1; u[0]=-1
    v = [0]*12; v[0]=1; v[1]=1
    w = [0]*12
    w[4] = 1   # C[1,1]
    add_term(u, v, w)
    
    # Strassen M7
    u = [0]*9; u[1]=1; u[4]=-1
    v = [0]*12; v[3]=1; v[4]=1
    w = [0]*12
    w[0] = 1   # C[0,0]
    add_term(u, v, w)
    
    # Block 12 (C[0,2] and C[1,2])
    # A[0,0]B[0,2] -> C[0,2] (k=2, i=0 -> 6)
    u = [0]*9; u[0]=1
    v = [0]*12; v[2]=1
    w = [0]*12; w[6]=1
    add_term(u, v, w)
    
    # A[0,1]B[1,2] -> C[0,2]
    u = [0]*9; u[1]=1
    v = [0]*12; v[5]=1
    w = [0]*12; w[6]=1
    add_term(u, v, w)
    
    # A[1,0]B[0,2] -> C[1,2] (k=2, i=1 -> 7)
    u = [0]*9; u[3]=1
    v = [0]*12; v[2]=1
    w = [0]*12; w[7]=1
    add_term(u, v, w)
    
    # A[1,1]B[1,2] -> C[1,2]
    u = [0]*9; u[4]=1
    v = [0]*12; v[5]=1
    w = [0]*12; w[7]=1
    add_term(u, v, w)
    
    # A[0,2]B[2,2] -> C[0,2]
    u = [0]*9; u[2]=1
    v = [0]*12; v[8]=1
    w = [0]*12; w[6]=1
    add_term(u, v, w)
    
    # A[1,2]B[2,2] -> C[1,2]
    u = [0]*9; u[5]=1
    v = [0]*12; v[8]=1
    w = [0]*12; w[7]=1
    add_term(u, v, w)
    
    # Block 21 (C[2,0] and C[2,1])
    # C[2,0] (k=0, i=2 -> 2)
    # C[2,1] (k=1, i=2 -> 5)
    
    # A[2,0]B[0,0] -> C[2,0]
    u = [0]*9; u[6]=1
    v = [0]*12; v[0]=1
    w = [0]*12; w[2]=1
    add_term(u, v, w)
    
    # A[2,0]B[0,1] -> C[2,1]
    u = [0]*9; u[6]=1
    v = [0]*12; v[1]=1
    w = [0]*12; w[5]=1
    add_term(u, v, w)
    
    # A[2,1]B[1,0] -> C[2,0]
    u = [0]*9; u[7]=1
    v = [0]*12; v[3]=1
    w = [0]*12; w[2]=1
    add_term(u, v, w)
    
    # A[2,1]B[1,1] -> C[2,1]
    u = [0]*9; u[7]=1
    v = [0]*12; v[4]=1
    w = [0]*12; w[5]=1
    add_term(u, v, w)
    
    # A[2,2]B[2,0] -> C[2,0]
    u = [0]*9; u[8]=1
    v = [0]*12; v[6]=1
    w = [0]*12; w[2]=1
    add_term(u, v, w)
    
    # A[2,2]B[2,1] -> C[2,1]
    u = [0]*9; u[8]=1
    v = [0]*12; v[7]=1
    w = [0]*12; w[5]=1
    add_term(u, v, w)
    
    # Block 22 (C[2,2])
    # C[2,2] (k=2, i=2 -> 8)
    
    # A[2,0]B[0,2] -> C[2,2]
    u = [0]*9; u[6]=1
    v = [0]*12; v[2]=1
    w = [0]*12; w[8]=1
    add_term(u, v, w)
    
    # A[2,1]B[1,2] -> C[2,2]
    u = [0]*9; u[7]=1
    v = [0]*12; v[5]=1
    w = [0]*12; w[8]=1
    add_term(u, v, w)
    
    # A[2,2]B[2,2] -> C[2,2]
    u = [0]*9; u[8]=1
    v = [0]*12; v[8]=1
    w = [0]*12; w[8]=1
    add_term(u, v, w)
    
    # --- Part 2: 4th Column (k=3) Schoolbook (Rank 9) ---
    # C[i,3] = sum_j A[i,j] B[j,3]
    # W index for k=3: 3*3+i = 9+i.
    
    for i in range(3):
        for j in range(3):
            u = [0]*9; u[i*3+j] = 1
            v = [0]*12; v[j*3+3] = 1
            w = [0]*12; w[3*3+i] = 1
            add_term(u, v, w)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
