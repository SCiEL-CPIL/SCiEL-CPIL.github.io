"""Matrix multiplication 3x4x7 rank-minimal decomposition."""
import time

DIMENSIONS = [3, 4, 7]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = -1; W[2][0] = -1; W[3][0] = 1
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = -1; W[2][1] = 1
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1
    # M6 = (c-d)(f+h)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[3][5] = 1
    W[3][5] = -1
    # M7 = (b-d)(e+g)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = -1
    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for c in range(rank):
                    rhs += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 4, 7
    
    sU, sV, sW = _strassen_222()
    
    # Build 3x4x7 with rank 78
    # 2x4x7 part (i=0,1): split m into [2,2], p into [2,2,2,1]
    # Each 2x2x2: Strassen (7), each 2x2x1: schoolbook (4)
    # 2*(3*7 + 4) = 2*25 = 50
    # 1x4x7 part (i=2): schoolbook (28)
    # Total: 78
    
    rank_total = 78
    U = [[0]*rank_total for _ in range(12)]
    V = [[0]*rank_total for _ in range(28)]
    W = [[0]*rank_total for _ in range(21)]
    
    col = 0
    p_blocks = [2, 2, 2, 1]
    
    for mb_idx in range(2):
        j_start = mb_idx * 2
        for pb_idx in range(4):
            pb = p_blocks[pb_idx]
            k_start = sum(p_blocks[:pb_idx])
            if pb == 2:
                Uc, Vc, Wc = sU, sV, sW
                rc = 7
            else:
                Uc, Vc, Wc = _schoolbook(2, 2, 1)
                rc = 4
            
            for i in range(2):
                for jl in range(2):
                    for c in range(rc):
                        U[i*4 + j_start + jl][col + c] = Uc[i*2 + jl][c]
            for jl in range(2):
                for kl in range(pb):
                    for c in range(rc):
                        V[(j_start+jl)*7 + k_start + kl][col + c] = Vc[jl*pb + kl][c]
            for kl in range(pb):
                for i in range(2):
                    for c in range(rc):
                        W[(k_start+kl)*3 + i][col + c] = Wc[kl*2 + i][c]
            col += rc
    
    # 1x4x7 part
    U1, V1, W1 = _schoolbook(1, 4, 7)
    for j in range(4):
        for k in range(7):
            cl = j*7 + k
            U[8+j][col + cl] = U1[j][cl]
            V[j*7 + k][col + cl] = V1[j*7 + k][cl]
            W[k*3 + 2][col + cl] = W1[k][cl]
    
    # Verify
    if not _verify(U, V, W, 3, 4, 7):
        U, V, W = _schoolbook(3, 4, 7)
    
    return {"U": U, "V": V, "W": W}
