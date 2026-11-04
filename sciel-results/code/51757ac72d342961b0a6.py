"""Matrix multiplication 2x3x5 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return 7 (u, v, w) terms for 2x2x2 matrix multiplication.
    Each term is (u_dict, v_dict, w_dict) mapping local (i,j), (j,k), (k,i) -> coefficient."""
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    terms = []
    # M0
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}))
    # M1
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):1, (1,0):1}))
    # M2
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}))
    # M3
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}))
    # M4
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}))
    # M5
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1, (0,1):1}))
    # M6
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,0):1, (1,1):1}))
    return terms

def _schoolbook_block(n, m, p):
    """Return schoolbook terms for n x m x p block."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n, m, p = 2, 3, 5  # A is n x m, B is m x p, C is p x n (trace convention: W[k*n+i, r])
    
    # Build rank-28 decomposition:
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): 2x2x2 -> 7 Strassen terms
    # Block (i in {0,1}, j in {0,1}, k in {2,3}): 2x2x2 -> 7 Strassen terms
    # Block (i in {0,1}, j in {0,1}, k in {4}):   2x2x1 -> 4 schoolbook terms
    # Block (i in {0,1}, j in {2},    k in {0..4}): 2x1x5 -> 10 schoolbook terms
    # Total: 7 + 7 + 4 + 10 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    strassen_terms = _strassen_2x2x2()
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    for (u, v, w) in strassen_terms:
        for (li, lj), coef in u.items():
            gi, gj = li, lj
            U[gi * m + gj][col] += coef
        for (lj, lk), coef in v.items():
            gj, gk = lj, lk
            V[gj * p + gk][col] += coef
        for (lk, li), coef in w.items():
            gk, gi = lk, li
            W[gk * n + gi][col] += coef
        col += 1
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    for (u, v, w) in strassen_terms:
        for (li, lj), coef in u.items():
            gi, gj = li, lj
            U[gi * m + gj][col] += coef
        for (lj, lk), coef in v.items():
            gj, gk = lj, lk + 2
            V[gj * p + gk][col] += coef
        for (lk, li), coef in w.items():
            gk, gi = lk + 2, li
            W[gk * n + gi][col] += coef
        col += 1
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4} (2x2x1 schoolbook)
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Block 4: i in {0,1}, j in {2}, k in {0,1,2,3,4} (2x1x5 schoolbook)
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
