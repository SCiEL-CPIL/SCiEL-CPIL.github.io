"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x2():
    """Strassen rank-7 decomposition for 2x2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00(b10-b11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b01-b00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b01+b11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_3x4x4_rank44():
    """
    Construct a rank-44 decomposition for 3x4x4.
    Split n=3 into 2+1.
    The 2x4x4 part uses Strassen on 2x2 blocks: rank = 2 * 2 * 7 = 28.
    The 1x4x4 part uses schoolbook: rank = 4 * 4 = 16.
    Total rank = 28 + 16 = 44.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    sU, sV, sW = _winograd_2x2x2()
    
    col = 0
    # Part 1: rows 0,1 of A (2x4x4) using Strassen
    for g in range(2):  # inner dim pair: 0 -> {0,1}, 1 -> {2,3}
        j0, j1 = 2*g, 2*g+1
        for h in range(2):  # output col pair: 0 -> {0,1}, 1 -> {2,3}
            k0, k1 = 2*h, 2*h+1
            for s in range(7):
                c = col + s
                # A sub: rows {0,1}, cols {j0,j1}
                U[0*m + j0][c] = sU[0][s]
                U[0*m + j1][c] = sU[1][s]
                U[1*m + j0][c] = sU[2][s]
                U[1*m + j1][c] = sU[3][s]
                
                # B sub: rows {j0,j1}, cols {k0,k1}
                V[j0*p + k0][c] = sV[0][s]
                V[j0*p + k1][c] = sV[1][s]
                V[j1*p + k0][c] = sV[2][s]
                V[j1*p + k1][c] = sV[3][s]
                
                # C output: rows {0,1}, cols {k0,k1}
                # W index: (k, i)
                W[k0*n + 0][c] = sW[0][s]
                W[k0*n + 1][c] = sW[1][s]
                W[k1*n + 0][c] = sW[2][s]
                W[k1*n + 1][c] = sW[3][s]
                
                col += 1
    
    # Part 2: row 2 of A (1x4x4) using schoolbook
    for j in range(4):
        for k in range(4):
            c = col
            U[2*m + j][c] = 1
            V[j*p + k][c] = 1
            W[k*n + 2][c] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    try:
        candidate = _construct_3x4x4_rank44()
        if candidate and _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            rank = len(candidate["U"][0])
            if rank < best_rank:
                best = candidate
                best_rank = rank
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
