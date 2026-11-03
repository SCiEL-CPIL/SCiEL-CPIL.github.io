import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 4, 4
    rank = 32
    U = [[0]*rank for _ in range(n*m)]  # 12 x 32
    V = [[0]*rank for _ in range(m*p)]  # 16 x 32
    W = [[0]*rank for _ in range(p*n)]  # 12 x 32
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        """Add Strassen decomposition for 2x2x2 block.
        A rows a_r, a_r+1; cols a_c, a_c+1
        B rows b_r, b_r+1; cols b_c, b_c+1
        C rows c_r, c_r+1; cols c_c, c_c+1
        """
        c = start_col
        # m1 = (a00 + a11)(b00 + b11)
        U[a_r*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m2 = (a10 + a11)b00
        U[(a_r+1)*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] -= 1
        c += 1
        
        # m3 = a00(b01 - b11)
        U[a_r*m + a_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m4 = a11(b10 - b00)
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + b_c][c] += 1
        V[b_r*p + b_c][c] -= 1
        W[c_c*n + c_r][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        c += 1
        
        # m5 = (a00 + a01)b11
        U[a_r*m + a_c][c] += 1
        U[a_r*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        c += 1
        
        # m6 = (a10 - a00)(b00 + b01)
        U[(a_r+1)*m + a_c][c] += 1
        U[a_r*m + a_c][c] -= 1
        V[b_r*p + b_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m7 = (a01 - a11)(b10 + b11)
        U[a_r*m + (a_c+1)][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] -= 1
        V[(b_r+1)*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        c += 1
        
        return c
    
    def add_trivial_1x4x4(a_r, b_c_start, c_c_start, start_col):
        """Trivial 1x4x4 multiplication for one output row."""
        c = start_col
        for i in range(4):
            for j in range(4):
                U[a_r*m + i][c] += 1
                V[i*p + j][c] += 1
                W[j*n + a_r][c] += 1
                c += 1
        return c
    
    # Top 2 rows: 2x4x4 using 4 Strassen blocks
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    col = add_strassen(0, 0, 0, 0, 0, 0, col)  # cols 0-6
    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    col = add_strassen(0, 2, 2, 0, 0, 0, col)  # cols 7-13
    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    col = add_strassen(0, 0, 0, 2, 0, 2, col)  # cols 14-20
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    col = add_strassen(0, 2, 2, 2, 0, 2, col)  # cols 21-27
    
    # Bottom row: 1x4x4 trivial
    col = add_trivial_1x4x4(2, 0, 0, col)  # cols 28-31
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
