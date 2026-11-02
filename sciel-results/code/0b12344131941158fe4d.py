import time

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    U = [[0]*rank for _ in range(n*m)]  # 6 x 39
    V = [[0]*rank for _ in range(m*p)]  # 21 x 39
    W = [[0]*rank for _ in range(n*p)]  # 14 x 39
    
    col = 0
    
    # Helper to add a Strassen block for A (2x2) * B (2x2) contributing to C columns c0, c1
    def add_strassen(A_rows, B_col_offsets, C_col_offsets):
        nonlocal col
        # A_rows: list of 2 U-row indices for the 2x2 A block
        # B_col_offsets: list of 2 V-row base offsets (for the 2 rows of B)
        # C_col_offsets: the 2 output column indices
        a00, a10 = A_rows  # U row indices for A[0,*] and A[1,*]... 
        # Actually, let me be more explicit.
        # A = [[a00, a01], [a10, a11]]
        # U row for a00 = A[0,0], a01 = A[0,1], a10 = A[1,0], a11 = A[1,1]
        # For our case, A1 = A[:,0:2], so:
        # a00 = A[0,0] → U row 0
        # a01 = A[0,1] → U row 1
        # a10 = A[1,0] → U row 2
        # a11 = A[1,1] → U row 3
        pass
    
    # Let me just hardcode the three Strassen blocks and the schoolbook parts.
    
    # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # U rows: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # B1 rows: B[0,*] → V rows 0-6, B[1,*] → V rows 7-13
    # B2 row: B[2,*] → V rows 14-20
    # A2 = [A[0,2], A[1,2]] → U rows 4, 5
    # C: C[0,k] → W row k*2+0, C[1,k] → W row k*2+1
    
    # Strassen block 1: A1 * B1[:,0:2] → C[:,0:2]
    # B1[:,0] = [B[0,0], B[1,0]] → V rows 0, 7
    # B1[:,1] = [B[0,1], B[1,1]] → V rows 1, 8
    # C[:,0] = [C[0,0], C[1,0]] → W rows 0, 2
    # C[:,1] = [C[0,1], C[1,1]] → W rows 1, 3
    
    def strassen_block(col, a_rows, b_row0, b_row1, c_row0, c_row1):
        # a_rows = [a00_row, a01_row, a10_row, a11_row]
        # b_row0 = [b00_row, b01_row] (V rows for B[0,0] and B[0,1]... wait)
        # Let me redefine: b_rows = [[b00_row, b01_row], [b10_row, b11_row]]
        # c_rows = [c00_row, c01_row, c10_row, c11_row]
        pass
    
    # This is getting complicated. Let me just write it out explicitly.
    
    # Strassen for 2x2 * 2x2:
    # m1 = (a00+a11)*(b00+b11) → W: c00+=1, c11+=1
    # m2 = (a10+a11)*b00 → W: c10+=1
    # m3 = a00*(b01-b11) → W: c01+=1, c11+=1
    # m4 = a11*(b10-b00) → W: c00+=1, c10+=1
    # m5 = (a00+a01)*b11 → W: c00-=1, c01+=1
    # m6 = (a10-a00)*(b00+b10) → W: c11+=1
    # m7 = (a01-a11)*(b01+b11) → W: c00+=1, c11-=1
    
    A_rows = [0, 1, 2, 3]  # a00, a01, a10, a11
    
    # Block 1: B columns 0,1 → V rows for B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    # C columns 0,1 → W rows: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    b00, b01, b10, b11 = 0, 1, 7, 8
    c00, c01, c10, c11 = 0, 1, 2, 3
    
    # m1
    U[A_rows[0]][col] += 1  # a00
    U[A_rows[3]][col] += 1  # a11
    V[b00][col] += 1
    V[b11][col] += 1
    W[c00][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m2
    U[A_rows[2]][col] += 1  # a10
    U[A_rows[3]][col] += 1  # a11
    V[b00][col] += 1
    W[c10][col] += 1
    col += 1
    
    # m3
    U[A_rows[0]][col] += 1  # a00
    V[b01][col] += 1
    V[b11][col] -= 1
    W[c01][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m4
    U[A_rows[3]][col] += 1  # a11
    V[b10][col] += 1
    V[b00][col] -= 1
    W[c00][col] += 1
    W[c10][col] += 1
    col += 1
    
    # m5
    U[A_rows[0]][col] += 1  # a00
    U[A_rows[1]][col] += 1  # a01
    V[b11][col] += 1
    W[c00][col] -= 1
    W[c01][col] += 1
    col += 1
    
    # m6
    U[A_rows[2]][col] += 1  # a10
    U[A_rows[0]][col] -= 1  # a00
    V[b00][col] += 1
    V[b10][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m7
    U[A_rows[1]][col] += 1  # a01
    U[A_rows[3]][col] -= 1  # a11
    V[b01][col] += 1
    V[b11][col] += 1
    W[c00][col] += 1
    W[c11][col] -= 1
    col += 1
    
    # Block 2: B columns 2,3 → V rows: B[0,2]=2, B[0,3]=3, B[1,2]=9, B[1,3]=10
    # C columns 2,3 → W rows: C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7
    
    b00, b01, b10, b11 = 2, 3, 9, 10
    c00, c01, c10, c11 = 4, 5, 6, 7
    
    # m1
    U[A_rows[0]][col] += 1
    U[A_rows[3]][col] += 1
    V[b00][col] += 1
    V[b11][col] += 1
    W[c00][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m2
    U[A_rows[2]][col] += 1
    U[A_rows[3]][col] += 1
    V[b00][col] += 1
    W[c10][col] += 1
    col += 1
    
    # m3
    U[A_rows[0]][col] += 1
    V[b01][col] += 1
    V[b11][col] -= 1
    W[c01][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m4
    U[A_rows[3]][col] += 1
    V[b10][col] += 1
    V[b00][col] -= 1
    W[c00][col] += 1
    W[c10][col] += 1
    col += 1
    
    # m5
    U[A_rows[0]][col] += 1
    U[A_rows[1]][col] += 1
    V[b11][col] += 1
    W[c00][col] -= 1
    W[c01][col] += 1
    col += 1
    
    # m6
    U[A_rows[2]][col] += 1
    U[A_rows[0]][col] -= 1
    V[b00][col] += 1
    V[b10][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m7
    U[A_rows[1]][col] += 1
    U[A_rows[3]][col] -= 1
    V[b01][col] += 1
    V[b11][col] += 1
    W[c00][col] += 1
    W[c11][col] -= 1
    col += 1
    
    # Block 3: B columns 4,5 → V rows: B[0,4]=4, B[0,5]=5, B[1,4]=11, B[1,5]=12
    # C columns 4,5 → W rows: C[0,4]=8, C[0,5]=9, C[1,4]=10, C[1,5]=11
    
    b00, b01, b10, b11 = 4, 5, 11, 12
    c00, c01, c10, c11 = 8, 9, 10, 11
    
    # m1
    U[A_rows[0]][col] += 1
    U[A_rows[3]][col] += 1
    V[b00][col] += 1
    V[b11][col] += 1
    W[c00][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m2
    U[A_rows[2]][col] += 1
    U[A_rows[3]][col] += 1
    V[b00][col] += 1
    W[c10][col] += 1
    col += 1
    
    # m3
    U[A_rows[0]][col] += 1
    V[b01][col] += 1
    V[b11][col] -= 1
    W[c01][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m4
    U[A_rows[3]][col] += 1
    V[b10][col] += 1
    V[b00][col] -= 1
    W[c00][col] += 1
    W[c10][col] += 1
    col += 1
    
    # m5
    U[A_rows[0]][col] += 1
    U[A_rows[1]][col] += 1
    V[b11][col] += 1
    W[c00][col] -= 1
    W[c01][col] += 1
    col += 1
    
    # m6
    U[A_rows[2]][col] += 1
    U[A_rows[0]][col] -= 1
    V[b00][col] += 1
    V[b10][col] += 1
    W[c11][col] += 1
    col += 1
    
    # m7
    U[A_rows[1]][col] += 1
    U[A_rows[3]][col] -= 1
    V[b01][col] += 1
    V[b11][col] += 1
    W[c00][col] += 1
    W[c11][col] -= 1
    col += 1
    
    # Block 4: B column 6 (2x1) → schoolbook
    # A1 * [B[0,6], B[1,6]] → C[:,6]
    # V rows: B[0,6]=6, B[1,6]=13
    # C rows: C[0,6]=12, C[1,6]=13
    
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    U[0][col] += 1  # A[0,0]
    V[6][col] += 1  # B[0,6]
    W[12][col] += 1  # C[0,6]
    col += 1
    
    U[1][col] += 1  # A[0,1]
    V[13][col] += 1  # B[1,6]
    W[12][col] += 1  # C[0,6]
    col += 1
    
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    U[2][col] += 1  # A[1,0]
    V[6][col] += 1  # B[0,6]
    W[13][col] += 1  # C[1,6]
    col += 1
    
    U[3][col] += 1  # A[1,1]
    V[13][col] += 1  # B[1,6]
    W[13][col] += 1  # C[1,6]
    col += 1
    
    # A2 * B2: A[0,2]*B[2,k] → C[0,k], A[1,2]*B[2,k] → C[1,k] for k=0,...,6
    # U rows: A[0,2]=4, A[1,2]=5
    # V rows: B[2,k]=14+k for k=0,...,6
    # W rows: C[0,k]=2k, C[1,k]=2k+1
    
    for k in range(7):
        U[4][col] += 1  # A[0,2]
        V[14+k][col] += 1  # B[2,k]
        W[2*k][col] += 1  # C[0,k]
        col += 1
        
        U[5][col] += 1  # A[1,2]
        V[14+k][col] += 1  # B[2,k]
        W[2*k+1][col] += 1  # C[1,k]
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
