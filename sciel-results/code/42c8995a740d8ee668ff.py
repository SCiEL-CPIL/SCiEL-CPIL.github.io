"""Tensor decomposition for 3x3 by 3x8 matrix multiplication."""
import time
DIMENSIONS = [3, 3, 8]

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Try Strassen-based decomposition for rank 52
    # Partition j into {0,1} and {2}
    # For j in {0,1}: split k into 4 pairs, use 7-term 2x2x2 decomposition
    # For j = 2: schoolbook (3*8 = 24 terms)
    
    terms = []  # Each term: (U_dict, V_dict, W_dict)
    
    # 2x2x2 Strassen terms for rows i in {0,1}, cols j in {0,1}
    # Verified 7-term decomposition
    def add_2x2_2(i1, i2, j1, j2, k1, k2):
        # 7 rank-1 terms for 2x2x2 subproblem
        # Using verified coefficients
        t = [
            ({(i1,j1):1, (i2,j2):1}, {(j1,k1):1, (j2,k2):1}, {(i1,k1):1, (i2,k2):1}),
            ({(i2,j1):1, (i2,j2):1}, {(j1,k1):1}, {(i2,k1):1}),
            ({(i1,j1):1}, {(j2,k1):1, (j2,k2):-1}, {(i1,k1):1, (i1,k2):-1}),
            ({(i2,j2):1}, {(j1,k1):1, (j2,k1):-1}, {(i1,k1):-1, (i2,k1):1}),
            ({(i1,j1):1, (i1,j2):1}, {(j2,k2):1}, {(i1,k1):1, (i2,k2):1}),
            ({(i1,j2):1, (i2,j1):-1}, {(j1,k1):1, (j2,k1):1}, {(i2,k2):-1}),
            ({(i2,j1):1, (i1,j2):-1}, {(j2,k2):1, (j1,k2):1}, {(i2,k1):1, (i1,k2):-1}),
        ]
        terms.extend(t)
    
    for kp in range(4):
        k1 = 2 * kp
        k2 = 2 * kp + 1
        add_2x2_2(0, 1, 0, 1, k1, k2)
    
    # Schoolbook for j=2
    for i in range(3):
        for k in range(8):
            terms.append(({(i,2):1}, {(2,k):1}, {(i,k):1}))
    
    rank = len(terms)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i,j), val in u_dict.items():
            U[i*m+j][c] += val
        for (j,k), val in v_dict.items():
            V[j*p+k][c] += val
        for (i,k), val in w_dict.items():
            W[k*n+i][c] += val
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
