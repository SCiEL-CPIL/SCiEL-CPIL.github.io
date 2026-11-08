"""Rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
DIMENSIONS = [3, 3, 3]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _known_rank26():
    """Return a known valid rank-26 decomposition."""
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen for 2x2 block
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    # Cross terms
    cross = [(2,6,0),(2,7,3),(5,6,1),(5,7,4),(0,2,6),(1,5,6),(3,2,7),(4,5,7),
             (2,8,6),(5,8,7),(6,0,2),(7,3,2),(6,1,5),(7,4,5),(8,6,2),(8,7,5),
             (6,2,8),(7,5,8),(8,8,8)]
    for (u,v,w) in cross:
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    """Check if decomposition is exact."""
    rank = len(U[0])
    # Build target tensor T[i,j,k] = delta[i,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # In flattened form: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                expected = 1.0 if i == k else 0.0
                if abs(val - expected) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    best_result = _known_rank26()
    best_rank = 26
    
    # Try to find lower rank decompositions
    # Start with rank 25 and try random search
    for target_rank in [25, 24, 23]:
        if time.time() > deadline:
            break
        max_iters = 5000
        for attempt in range(max_iters):
            if time.time() > deadline:
                break
            # Random initialization with integer/half-integer values
            U = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-2,-1,-0.5,0,0.5,1,2]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _validate(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank < 26:
            break
    
    return best_result

    # U: rows indexed by (i, j) in row-major order: i*3 + j
    # V: rows indexed by (j, k) in row-major order: j*3 + k
    # W: rows indexed by (k, i) following the official trace convention W[k*n+i]
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1

    # Term 2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = -1

    # Term 3: A00(B01-B11) -> C01 += 1, C11 += 1
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[3][2] = 1; W[4][2] = 1

    # Term 4: A11(B10-B00) -> C00 += 1, C10 += 1
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # Term 5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[3][4] = 1

    # Term 6: (A10-A00)(B00+B01) -> C11 += 1
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # Term 7: (A01-A11)(B10+B11) -> C00 += 1
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1

    # Term 8: A02*B20 -> C00 += 1
    U[2][7] = 1
    V[6][7] = 1
    W[0][7] = 1

    # Term 9: A02*B21 -> C01 += 1
    U[2][8] = 1
    V[7][8] = 1
    W[3][8] = 1

    # Term 10: A12*B20 -> C10 += 1
    U[5][9] = 1
    V[6][9] = 1
    W[1][9] = 1

    # Term 11: A12*B21 -> C11 += 1
    U[5][10] = 1
    V[7][10] = 1
    W[4][10] = 1

    # Term 12: A00*B02 -> C02 += 1
    U[0][11] = 1
    V[2][11] = 1
    W[6][11] = 1

    # Term 13: A01*B12 -> C02 += 1
    U[1][12] = 1
    V[5][12] = 1
    W[6][12] = 1

    # Term 14: A10*B02 -> C12 += 1
    U[3][13] = 1
    V[2][13] = 1
    W[7][13] = 1

    # Term 15: A11*B12 -> C12 += 1
    U[4][14] = 1
    V[5][14] = 1
    W[7][14] = 1

    # Term 16: A02*B22 -> C02 += 1
    U[2][15] = 1
    V[8][15] = 1
    W[6][15] = 1

    # Term 17: A12*B22 -> C12 += 1
    U[5][16] = 1
    V[8][16] = 1
    W[7][16] = 1

    # Term 18: A20*B00 -> C20 += 1
    U[6][17] = 1
    V[0][17] = 1
    W[2][17] = 1

    # Term 19: A21*B10 -> C20 += 1
    U[7][18] = 1
    V[3][18] = 1
    W[2][18] = 1

    # Term 20: A20*B01 -> C21 += 1
    U[6][19] = 1
    V[1][19] = 1
    W[5][19] = 1

    # Term 21: A21*B11 -> C21 += 1
    U[7][20] = 1
    V[4][20] = 1
    W[5][20] = 1

    # Term 22: A22*B20 -> C20 += 1
    U[8][21] = 1
    V[6][21] = 1
    W[2][21] = 1

    # Term 23: A22*B21 -> C21 += 1
    U[8][22] = 1
    V[7][22] = 1
    W[5][22] = 1

    # Term 24: A20*B02 -> C22 += 1
    U[6][23] = 1
    V[2][23] = 1
    W[8][23] = 1

    # Term 25: A21*B12 -> C22 += 1
    U[7][24] = 1
    V[5][24] = 1
    W[8][24] = 1

# EVOLVE-BLOCK-END
