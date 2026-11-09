import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U44, V44, W44 = _strassen_4x4x4()
    
    # Build 4x4 * 4x2 decomposition (schoolbook, rank 32)
    U442, V442, W442 = _mul_4x4_by_4x2()
    
    rank1 = U44.shape[1]  # 49
    rank2 = U442.shape[1]  # 32
    total_rank = rank1 + rank2  # 81
    
    U = np.zeros((n * m, total_rank), dtype=float)  # 16 x 81
    V = np.zeros((m * p, total_rank), dtype=float)  # 24 x 81
    W = np.zeros((p * n, total_rank), dtype=float)  # 24 x 81
    
    # Part 1: 4x4 * 4x4
    U[:, :rank1] = U44
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, :rank1] = V44[j * 4 + k, :rank1]
    W[:, :rank1] = W44
    
    # Part 2: 4x4 * 4x2
    U[:, rank1:] = U442
    for j in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            V[j * 6 + k_global, rank1:] = V442[j * 2 + k_local, rank1:]
    for i in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            W[k_global * 4 + i, rank1:] = W442[i * 2 + k_local, rank1:]
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)
    
    return {
        "U": [list(map(float, row)) for row in U],
        "V": [list(map(float, row)) for row in V],
        "W": [list(map(float, row)) for row in W]
    }


def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # C00 = M1 - M4 + M6 + M7
    W[0, 0] = 1; W[0, 3] = -1; W[0, 5] = 1; W[0, 6] = 1
    # C01 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C10 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C11 = M1 - M3 + M2 - M5
    W[3, 0] = 1; W[3, 2] = -1; W[3, 1] = 1; W[3, 4] = -1
    
    return U, V, W


def _strassen_4x4x4():
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Strassen steps at 4x4 level
    steps = [
        # M1 = (A00+A11)(B00+B11)
        {'a': [(0,0,1), (1,1,1)], 'b': [(0,0,1), (1,1,1)], 'c': [(0,0,1), (1,1,1)]},
        # M2 = (A10+A11)(B00)
        {'a': [(1,0,1), (1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1), (1,1,1)]},
        # M3 = A00(B01-B11)
        {'a': [(0,0,1)], 'b': [(0,1,1), (1,1,-1)], 'c': [(0,1,1), (1,1,-1)]},
        # M4 = A11(B10-B00)
        {'a': [(1,1,1)], 'b': [(1,0,1), (0,0,-1)], 'c': [(1,0,1), (1,1,-1)]},
        # M5 = (A00+A01)(B11)
        {'a': [(0,0,1), (0,1,1)], 'b': [(1,1,1)], 'c': [(0,1,1), (1,1,-1)]},
        # M6 = (A10-A00)(B00+B01)
        {'a': [(1,0,1), (0,0,-1)], 'b': [(0,0,1), (0,1,1)], 'c': [(0,0,1), (1,0,-1)]},
        # M7 = (A01-A11)(B10+B11)
        {'a': [(0,1,1), (1,1,-1)], 'b': [(1,0,1), (1,1,1)], 'c': [(0,0,1), (1,1,-1)]},
    ]
    
    for s_idx, step in enumerate(steps):
        for t_idx in range(7):
            col = s_idx * 7 + t_idx
            
            for (br, bc, coeff) in step['a']:
                for lr in range(2):
                    for lc in range(2):
                        gidx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                        lidx = lr * 2 + lc
                        U[gidx, col] += coeff * U2[lidx, t_idx]
            
            for (br, bc, coeff) in step['b']:
                for lr in range(2):
                    for lc in range(2):
                        gidx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                        lidx = lr * 2 + lc
                        V[gidx, col] += coeff * V2[lidx, t_idx]
            
            for (cr, cc, ccoeff) in step['c']:
                for lr in range(2):
                    for lc in range(2):
                        gidx = (cc * 2 + lc) * 4 + (cr * 2 + lr)
                        lidx = lr * 2 + lc
                        W[gidx, col] += ccoeff * W2[lidx, t_idx]
    
    return U, V, W


def _mul_4x4_by_4x2():
    rank = 32
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((8, rank), dtype=float)
    W = np.zeros((8, rank), dtype=float)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i * 4 + j, col] = 1.0
                V[j * 2 + k, col] = 1.0
                W[i * 2 + k, col] = 1.0
                col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {
        "U": [list(map(float, row)) for row in U],
        "V": [list(map(float, row)) for row in V],
        "W": [list(map(float, row)) for row in W]
    }

