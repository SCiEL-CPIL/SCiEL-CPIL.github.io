import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# Strassen 2x2x2: 7 rank-1 terms
# Each term is (u, v, w) where u,v,w are length 4 vectors (flattened 2x2)
# Indexing: u[i*m+j], v[j*p+k], w[k*n+i]
# For 2x2x2: m=p=n=2
# u indices: 00, 01, 10, 11 -> 0, 1, 2, 3
# v indices: 00, 01, 10, 11 -> 0, 1, 2, 3
# w indices: 00, 01, 10, 11 -> 0, 1, 2, 3

_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
]

def _generate_strassen_444():
    """Generate rank-49 decomposition for 4x4x4 using Kronecker product of Strassen 2x2x2."""
    terms = []
    for u1, v1, w1 in _STRASSEN_222:
        for u2, v2, w2 in _STRASSEN_222:
            # Kronecker product
            # u1 is 2x2 flattened, u2 is 2x2 flattened
            # Result is 4x4 flattened
            # Index mapping:
            # 2x2: (r,c) -> r*2+c
            # 4x4: (R,C) -> R*4+C
            # Kronecker: A(2x2) * B(2x2) -> C(4x4)
            # C[r1*2+r2, c1*2+c2] = A[r1,c1] * B[r2,c2]
            # Flattened index for C: (r1*2+r2)*4 + (c1*2+c2)
            # u1 index i1 = r1*2+c1
            # u2 index i2 = r2*2+c2
            # We need to map (i1, i2) to i4.
            # r1 = i1//2, c1 = i1%2
            # r2 = i2//2, c2 = i2%2
            # i4 = (r1*2+r2)*4 + (c1*2+c2)
            
            u = [0]*16
            v = [0]*16
            w = [0]*16
            for i1 in range(4):
                r1 = i1 // 2
                c1 = i1 % 2
                for i2 in range(4):
                    r2 = i2 // 2
                    c2 = i2 % 2
                    i4 = (r1*2 + r2)*4 + (c1*2 + c2)
                    u[i4] = u1[i1] * u2[i2]
                    v[i4] = v1[i1] * v2[i2]
                    w[i4] = w1[i1] * w2[i2]
            terms.append((u, v, w))
    return terms

_STRASSEN_444 = _generate_strassen_444()

def _add_444(U, V, W, col, a_base, b_base, c_base):
    """Add rank-49 Strassen terms for a 4x4x4 block."""
    for u, v, w in _STRASSEN_444:
        for t in range(16):
            if u[t]:
                U[a_base + t][col] += u[t]
            if v[t]:
                V[b_base + t][col] += v[t]
            if w[t]:
                W[c_base + t][col] += w[t]
        col += 1
    return col

def _add_matrix_vector_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    """Matrix-vector: A(ar x ac) * B(ac x bc) where bc=1. Rank ar*ac."""
    for i in range(ar):
        for j in range(ac):
            U[a_base + i * ac + j][col] += 1
            V[b_base + j * bc][col] += 1
            W[c_base + i * bc][col] += 1
            col += 1
    return col

def _add_449(U, V, W, col, a_base, b_base, c_base):
    """Rank-114 decomposition for 4x4x9: two 4x4x4 Strassen blocks + 4x4x1 MV."""
    # B is 4x9. Split into 4x4 and 4x5.
    # C is 4x9. Split into 4x4 and 4x5.
    # Part 1: A(4x4) * B1(4x4) = C1(4x4). Rank 49.
    col = _add_444(U, V, W, col, a_base, b_base, c_base)
    
    # Part 2: A(4x4) * B2(4x5) = C2(4x5).
    # Split B2 into B2a(4x4) and B2b(4x1).
    # Split C2 into C2a(4x4) and C2b(4x1).
    # Part 2a: A * B2a(4x4) = C2a(4x4). Rank 49.
    # b_base offset for B2a: b_base + 4 (cols 4-7)
    # c_base offset for C2a: c_base + 4 (cols 4-7)
    col = _add_444(U, V, W, col, a_base, b_base + 4, c_base + 4)
    
    # Part 2b: A * B2b(4x1) = C2b(4x1). Rank 16.
    # b_base offset for B2b: b_base + 8 (col 8)
    # c_base offset for C2b: c_base + 8 (col 8)
    col = _add_matrix_vector_block(U, V, W, col, a_base, b_base + 8, c_base + 8, 4, 4, 1)
    
    return col

def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _build_rank114(n, m, p):
    """Rank-114: 4x4x4 Strassen (49) + 4x4x4 Strassen (49) + 4x4x1 MV (16)."""
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = _add_449(U, V, W, 0, 0, 0, 0)
    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    res = _build_rank114(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res

    return _build_schoolbook(n, m, p)
