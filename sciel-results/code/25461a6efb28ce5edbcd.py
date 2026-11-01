"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_block(U, V, W, rank_offset, r1, r2, c1, c2, k1, k2, n, m, p):
    """
    Apply Strassen's rank-7 algorithm to a 2x2x2 sub-block.
    r1,r2: row indices of A/C
    c1,c2: col indices of A / row indices of B
    k1,k2: row indices of B / col indices of C
    """
    strassen_factors = [
        # M1 = (A00+A11)(B00+B11)
        ([((r1,c1),1),((r2,c2),1)], [((c1,k1),1),((c2,k2),1)], [((k1,r1),1),((k2,r2),1)]),
        # M2 = (A10+A11)B00
        ([((r2,c1),1),((r2,c2),1)], [((c1,k1),1)], [((k1,r2),1),((k2,r2),-1)]),
        # M3 = A00(B01-B11)
        ([((r1,c1),1)], [((c1,k2),1),((c2,k2),-1)], [((k2,r1),1),((k2,r2),1)]),
        # M4 = A11(B10-B00)
        ([((r2,c2),1)], [((c2,k1),1),((c1,k1),-1)], [((k1,r1),1),((k1,r2),1)]),
        # M5 = (A00+A01)B11
        ([((r1,c1),1),((r1,c2),1)], [((c2,k2),1)], [((k1,r1),-1),((k2,r2),1)]),
        # M6 = (A10-A00)(B00+B01)
        ([((r2,c1),1),((r1,c1),-1)], [((c1,k1),1),((c1,k2),1)], [((k2,r2),1)]),
        # M7 = (A01-A11)(B10+B11)
        ([((r1,c2),1),((r2,c2),-1)], [((c2,k1),1),((c2,k2),1)], [((k1,r1),1)]),
    ]
    
    for t, (a_factors, b_factors, c_factors) in enumerate(strassen_factors):
        r = rank_offset + t
        for (idx, coeff) in a_factors:
            U[idx[0] * m + idx[1]][r] += coeff
        for (idx, coeff) in b_factors:
            V[idx[0] * p + idx[1]][r] += coeff
        for (idx, coeff) in c_factors:
            W[idx[0] * n + idx[1]][r] += coeff

def _schoolbook_block(U, V, W, rank_offset, i_rows, j_cols, k_rows, n, m, p):
    """Apply schoolbook to a sub-block, return new rank offset."""
    r = rank_offset
    for i in i_rows:
        for j in j_cols:
            for k in k_rows:
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return r

def _verify(U, V, W, n, m, p, tol=1e-9):
    U_a = np.array(U, dtype=float)
    V_a = np.array(V, dtype=float)
    W_a = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_a[i*m+j] * V_a[j*p+k] * W_a[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 5, 6
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Allocate max possible rank
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(U_dim)]
    V = [[0] * max_rank for _ in range(V_dim)]
    W = [[0] * max_rank for _ in range(W_dim)]
    
    # j-pairs: (0,1), (2,3); singleton: (4)
    # k-pairs: (0,1), (2,3), (4,5)
    j_pairs = [(0,1), (2,3)]
    k_pairs = [(0,1), (2,3), (4,5)]
    i_rows = (0,1)
    
    rank = 0
    # Strassen blocks: 2 j-pairs x 3 k-pairs = 6 blocks x rank 7 = 42
    for (c1, c2) in j_pairs:
        for (k1, k2) in k_pairs:
            _strassen_block(U, V, W, rank, 0, 1, c1, c2, k1, k2, n, m, p)
            rank += 7
    
    # Schoolbook blocks: j=4 singleton x 3 k-pairs = 3 blocks x rank 4 = 12
    for (k1, k2) in k_pairs:
        rank = _schoolbook_block(U, V, W, rank, i_rows, (4,), (k1, k2), n, m, p)
    
    # Trim to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        U_fb = [[0] * (n*m*p) for _ in range(U_dim)]
        V_fb = [[0] * (n*m*p) for _ in range(V_dim)]
        W_fb = [[0] * (n*m*p) for _ in range(W_dim)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
