import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # U: (6, rank), V: (15, rank), W: (10, rank)
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 6 x 28
    V = [[0]*rank for _ in range(m*p)]  # 15 x 28
    W = [[0]*rank for _ in range(p*n)]  # 10 x 28
    
    col = 0
    
    # A1*B1a: Strassen rank 7
    # A1 = [[a00, a01], [a10, a11]], B1a = [[b00, b01], [b10, b11]]
    # U indices: a00=0, a01=1, a10=3, a11=4
    # V indices: b00=0, b01=1, b10=5, b11=6
    # W indices: c00=0, c01=1, c10=2, c11=3
    
    # M1 = (a00+a11)(b00+b11) -> c00(+1), c11(+1)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (a10+a11)b00 -> c10(+1), c11(-1)
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = -1
    col += 1
    
    # M3 = a00(b01-b11) -> c01(+1), c11(+1)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = a11(b10-b00) -> c00(+1), c10(+1)
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M5 = (a00+a01)b11 -> c00(-1), c01(+1)
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (a10-a00)(b00+b01) -> c11(+1)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (a01-a11)(b10+b11) -> c00(+1)
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # A1*B1b: Strassen rank 7
    # B1b = [[b02, b03], [b12, b13]]
    # V indices: b02=2, b03=3, b12=7, b13=8
    # W indices: c02=4, c03=5, c12=6, c13=7
    
    # M1' = (a00+a11)(b02+b13) -> c02(+1), c13(+1)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2' = (a10+a11)b02 -> c12(+1), c13(-1)
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = -1
    col += 1
    
    # M3' = a00(b03-b13) -> c03(+1), c13(+1)
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[5][col] = 1; W[7][col] = 1
    col += 1
    
    # M4' = a11(b12-b02) -> c02(+1), c12(+1)
    U[4][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # M5' = (a00+a01)b13 -> c02(-1), c03(+1)
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # M6' = (a10-a00)(b02+b03) -> c13(+1)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7' = (a01-a11)(b12+b13) -> c02(+1)
    U[1][col] = 1; U[4][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # A1*B1c: Schoolbook rank 4
    # B1c = [b04, b14]
    # V indices: b04=4, b14=9
    # W indices: c04=8, c14=9
    
    # a00*b04 -> c04
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    # a01*b14 -> c04
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    # a10*b04 -> c14
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    # a11*b14 -> c14
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # A2*B2: Schoolbook rank 10
    # A2 = [a02, a12], B2 = [b20, b21, b22, b23, b24]
    # U indices: a02=2, a12=5
    # V indices: b20=10, b21=11, b22=12, b23=13, b24=14
    # W indices: c00=0, c01=1, c02=4, c03=5, c04=8, c10=2, c11=3, c12=6, c13=7, c14=9
    
    # a02*b20 -> c00
    U[2][col] = 1
    V[10][col] = 1
    W[0][col] = 1
    col += 1
    
    # a02*b21 -> c01
    U[2][col] = 1
    V[11][col] = 1
    W[1][col] = 1
    col += 1
    
    # a02*b22 -> c02
    U[2][col] = 1
    V[12][col] = 1
    W[4][col] = 1
    col += 1
    
    # a02*b23 -> c03
    U[2][col] = 1
    V[13][col] = 1
    W[5][col] = 1
    col += 1
    
    # a02*b24 -> c04
    U[2][col] = 1
    V[14][col] = 1
    W[8][col] = 1
    col += 1
    
    # a12*b20 -> c10
    U[5][col] = 1
    V[10][col] = 1
    W[2][col] = 1
    col += 1
    
    # a12*b21 -> c11
    U[5][col] = 1
    V[11][col] = 1
    W[3][col] = 1
    col += 1
    
    # a12*b22 -> c12
    U[5][col] = 1
    V[12][col] = 1
    W[6][col] = 1
    col += 1
    
    # a12*b23 -> c13
    U[5][col] = 1
    V[13][col] = 1
    W[7][col] = 1
    col += 1
    
    # a12*b24 -> c14
    U[5][col] = 1
    V[14][col] = 1
    W[9][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1) > 1e-10:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {val}, expected 1")
    
    return {"U": U, "V": V, "W": W}
