import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen 2x2x2 terms: (U_pos:coeff, V_pos:coeff, W_pos:coeff)
    # Local 2x2 positions: 0=TL, 1=TR, 2=BL, 3=BR
    S = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),       # P1=(x00+x11)(y00+y11)
        ({2:1, 3:1}, {0:1},       {2:1, 3:-1}),      # P2=(x10+x11)y00
        ({0:1},       {1:1, 3:-1}, {1:1, 3:1}),      # P3=x00(y01-y11)
        ({3:1},       {2:1, 0:-1}, {0:1, 2:1}),      # P4=x11(y10-y00)
        ({0:1, 1:1}, {3:1},       {0:-1, 1:1}),      # P5=(x00+x01)y11
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),            # P6=(x10-x00)(y00+y01)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),            # P7=(x01-x11)(y10+y11)
    ]
    
    def add_block(col_off, a_rows, b_rows, c_rows):
        """Add one Strassen 2x2x2 block.
        a_rows: [a00,a01,a10,a11] global U row indices
        b_rows: [b00,b01,b10,b11] global V row indices  
        c_rows: [c00,c01,c10,c11] global W row indices (W[k*n+i])
        """
        for t in range(7):
            uc, vc, wc = S[t]
            for pos, coef in uc.items():
                U[a_rows[pos]][col_off + t] += coef
            for pos, coef in vc.items():
                V[b_rows[pos]][col_off + t] += coef
            for pos, coef in wc.items():
                W[c_rows[pos]][col_off + t] += coef
    
    # Block 1: A_left(2x2) * B_tl(2x2) -> C_left(2x2)
    # A_left: A[0,0],A[0,1],A[1,0],A[1,1] -> U[0],U[1],U[4],U[5]
    # B_tl:   B[0,0],B[0,1],B[1,0],B[1,1] -> V[0],V[1],V[4],V[5]
    # C_left: C[0,0],C[0,1],C[1,0],C[1,1] -> W[0*2+0],W[0*2+1],W[1*2+0],W[1*2+1] = W[0],W[1],W[2],W[3]
    add_block(0, [0,1,4,5], [0,1,4,5], [0,1,2,3])
    
    # Block 2: A_right(2x2) * B_bl(2x2) -> C_left(2x2)
    # A_right: A[0,2],A[0,3],A[1,2],A[1,3] -> U[2],U[3],U[6],U[7]
    # B_bl:   B[2,0],B[2,1],B[3,0],B[3,1] -> V[8],V[9],V[12],V[13]
    # C_left: same W rows
    add_block(7, [2,3,6,7], [8,9,12,13], [0,1,2,3])
    
    # Block 3: A_left(2x2) * B_tr(2x2) -> C_right(2x2)
    # A_left: U[0],U[1],U[4],U[5]
    # B_tr:   B[0,2],B[0,3],B[1,2],B[1,3] -> V[2],V[3],V[6],V[7]
    # C_right: C[0,2],C[0,3],C[1,2],C[1,3] -> W[2*2+0],W[2*2+1],W[3*2+0],W[3*2+1] = W[4],W[5],W[6],W[7]
    add_block(14, [0,1,4,5], [2,3,6,7], [4,5,6,7])
    
    # Block 4: A_right(2x2) * B_br(2x2) -> C_right(2x2)
    # A_right: U[2],U[3],U[6],U[7]
    # B_br:   B[2,2],B[2,3],B[3,2],B[3,3] -> V[10],V[11],V[14],V[15]
    # C_right: W[4],W[5],W[6],W[7]
    add_block(21, [2,3,6,7], [10,11,14,15], [4,5,6,7])
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(999)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_comp = np.zeros((n, p))
    for r in range(rank):
        u_r = float(U_arr @ A_test.ravel())  # sum over all (i,j)
        v_r = float(V_arr @ B_test.ravel())  # sum over all (j,k)
        for i in range(n):
            for k in range(p):
                C_comp[i, k] += W_arr[k * n + i, r] * u_r * v_r
    
    if not np.allclose(C_comp, C_expected, atol=1e-9):
        # Fallback: schoolbook rank 32 (should never trigger)
        rank32 = n * m * p
        U32 = [[0]*rank32 for _ in range(n*m)]
        V32 = [[0]*rank32 for _ in range(m*p)]
        W32 = [[0]*rank32 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U32[i*m+j][col] = 1
                    V32[j*p+k][col] = 1
                    W32[k*n+i][col] = 1
                    col += 1
        return {"U": U32, "V": V32, "W": W32}
    
    return {"U": U, "V": V, "W": W}
