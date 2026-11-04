"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    # Block structure: split m=4 into two blocks of 2, split p=6 into three blocks of 2
    # For each (ba, bb) pair, apply Strassen (rank 7) to 2x2 x 2x2
    # Total rank = 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    # Strassen 7 products: u (2x2), v (2x2), w (2x2)
    # M_r = (sum u_r[a,b] A[a,b]) * (sum v_r[b,c] B[b,c])
    # C[a,c] += w_r[a,c] * M_r
    strassen = [
        # r=0: M1=(A00+A11)(B00+B11)
        ([[1,0],[1,0]], [[1,1],[0,0]], [[1,0],[0,1]]),
        # r=1: M2=(A10+A11)B00
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        # r=2: M3=A00(B10-B11)
        ([[1,0],[0,0]], [[0,0],[1,-1]], [[0,1],[0,1]]),
        # r=3: M4=A11(B01-B00)
        ([[0,0],[0,1]], [[-1,1],[0,0]], [[1,0],[0,1]]),
        # r=4: M5=(A00+A01)B11
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),
        # r=5: M6=(A10-A00)(B00+B01)
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # r=6: M7=(A01-A11)(B10+B11)
        ([[0,1],[-1,0]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    for ba in range(2):      # block index in m dimension (2 blocks of size 2)
        for bb in range(3):  # block index in p dimension (3 blocks of size 2)
            for r in range(7):
                col = (ba * 3 + bb) * 7 + r
                u, v, w = strassen[r]
                for a in range(2):
                    for b in range(2):
                        i = a
                        j = 2 * ba + b
                        U[i * m + j][col] += u[a][b]
                    for c in range(2):
                        k = 2 * bb + c
                        W[k * n + a][col] += w[a][c]
                for b in range(2):
                    for c in range(2):
                        j = 2 * ba + b
                        k = 2 * bb + c
                        V[j * p + k][col] += v[b][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
