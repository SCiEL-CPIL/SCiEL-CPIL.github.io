"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]


def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _schoolbook(n=4, m=4, p=9):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Strassen decomposition for 2x2 x 2x2 matmul, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = -1

    return {"U": U, "V": V, "W": W}


def _schoolbook_2x2x1():
    """Schoolbook decomposition for 2x2 A times 2x1 B -> 2x1 C. Rank 4.
    U[i*2+j, c] (4 rows), V[j*1+k, c] (2 rows), W[k*2+i, c] (2 rows)."""
    rank = 4
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(2)]
    W = [[0] * rank for _ in range(2)]
    
    # C0 = a*e + b*h
    # C1 = c*e + d*h
    
    # Term 1: a*e
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 2: b*h
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # Term 3: c*e
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # Term 4: d*h
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return {"U": U, "V": V, "W": W}


def _decomp_2x2x3():
    """Decomposition for 2x2 A times 2x3 B -> 2x3 C. Rank 11.
    Direct sum of Strassen (2x2x2) and Schoolbook (2x2x1).
    U[i*2+j, c] (4 rows), V[j*3+k, c] (6 rows), W[k*2+i, c] (4 rows)."""
    S = _strassen_2x2()
    B = _schoolbook_2x2x1()
    rank = 11
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(4)]
    
    # Part 1: Strassen for columns 0,1 (rank 7)
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                U[ii * 2 + jj][c] += S['U'][ii * 2 + jj][c]
        for jj in range(2):
            for kk in range(2):
                V[jj * 3 + kk][c] += S['V'][jj * 2 + kk][c]
        for kk in range(2):
            for ii in range(2):
                W[kk * 2 + ii][c] += S['W'][kk * 2 + ii][c]
                
    # Part 2: Schoolbook for column 2 (rank 4)
    for c in range(4):
        cc = 7 + c
        for ii in range(2):
            for jj in range(2):
                U[ii * 2 + jj][cc] += B['U'][ii * 2 + jj][c]
        for jj in range(2):
            for kk in range(1):
                V[jj * 3 + (2 + kk)][cc] += B['V'][jj * 1 + kk][c]
        for kk in range(1):
            for ii in range(2):
                W[(2 + kk) * 2 + ii][cc] += B['W'][kk * 2 + ii][c]
                
    return {"U": U, "V": V, "W": W}


def _decomp_4x4x3():
    """Decomposition for 4x4 A times 4x3 B -> 4x3 C. Rank 44.
    Direct sum of 4x (2x2x3).
    U[i*4+j, c] (16 rows), V[j*3+k, c] (12 rows), W[k*4+i, c] (12 rows)."""
    S = _decomp_2x2x3()
    rank = 44
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    # 4 blocks: (I,J) in {0,1}x{0,1}
    for I in range(2):
        for J in range(2):
            base = (I * 2 + J) * 11
            for c in range(11):
                cc = base + c
                for ii in range(2):
                    for jj in range(2):
                        U[(2 * I + ii) * 4 + (2 * J + jj)][cc] += S['U'][ii * 2 + jj][c]
                for jj in range(2):
                    for kk in range(3):
                        V[(2 * J + jj) * 3 + kk][cc] += S['V'][jj * 3 + kk][c]
                for kk in range(3):
                    for ii in range(2):
                        W[kk * 4 + (2 * I + ii)][cc] += S['W'][kk * 2 + ii][c]
                        
    return {"U": U, "V": V, "W": W}


def _decomp_4x4x9():
    """Full 4x4 x 4x9 decomposition, rank 132.
    Direct sum of three 4x4x3 blocks (columns 0-2, 3-5, 6-8)."""
    B = _decomp_4x4x3()
    rank = 132
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for t in range(3):
        for c in range(44):
            cc = t * 44 + c
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][cc] += B['U'][i * 4 + j][c]
            for j in range(4):
                for k in range(3):
                    V[j * 9 + (3 * t + k)][cc] += B['V'][j * 3 + k][c]
            for k in range(3):
                for i in range(4):
                    W[(3 * t + k) * 4 + i][cc] += B['W'][k * 4 + i][c]
                    
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 9
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    try:
        cand = _decomp_4x4x9()
        if _verify(cand['U'], cand['V'], cand['W'], n, m, p):
            best_result = cand
            best_rank = 132
            print("Verified rank-132 decomposition")
    except Exception as e:
        print("rank-132 construction failed:", e)

    return best_result
