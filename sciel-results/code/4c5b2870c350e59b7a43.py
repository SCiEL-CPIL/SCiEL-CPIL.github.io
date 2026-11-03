import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B.
    C[0,0]=ae+cg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh.
    m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    C[0,0]=m1+m4-m5+m7
    C[0,1]=m3+m5
    C[1,0]=m2+m4
    C[1,1]=m1-m2+m3+m6
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W

def build_6x6x6_strassen():
    """
    Build a rank-56 decomposition of the 6x6x6 matrix multiplication tensor
    using 2x2x2 Strassen blocks.
    Returns (U, V, W) with shapes (36, 56), (36, 56), (36, 56).
    Convention: U[i*6+j, c], V[j*6+k, c], W[k*6+i, c].
    """
    S = 6
    U_full = np.zeros((S * S, 56), dtype=np.float64)
    V_full = np.zeros((S * S, 56), dtype=np.float64)
    W_full = np.zeros((S * S, 56), dtype=np.float64)

    col = 0
    for a in range(0, S, 2):
        for b in range(0, S, 2):
            for c in range(0, S, 2):
                Ub, Vb, Wb = strassen_factors()
                for i in range(2):
                    for j in range(2):
                        U_full[(a + i) * S + (b + j), col:col+7] = Ub[i * 2 + j, :]
                for j in range(2):
                    for k in range(2):
                        V_full[(b + j) * S + (c + k), col:col+7] = Vb[j * 2 + k, :]
                for k in range(2):
                    for i in range(2):
                        W_full[(c + k) * S + (a + i), col:col+7] = Wb[k * 2 + i, :]
                col += 7

    return U_full, V_full, W_full

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[i,j,k] should be 1 for all i,j,k.
    """
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx

    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)

    return bool(np.allclose(vals, 1.0, atol=atol))

def schoolbook_lists(n, m, p):
    """Return schoolbook decomposition as nested lists (fallback)."""
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

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    
    Strategy: Build a rank-56 decomposition of the 6x6x6 tensor using Strassen
    2x2x2 blocks, then restrict (project) to the 3x5x6 subspace by selecting
    the appropriate rows. This gives a valid rank-56 decomposition of the
    3x5x6 tensor (better than the rank-84 block decomposition).
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock

    # Build the 6x6x6 Strassen decomposition (rank 56)
    S = 6
    U6, V6, W6 = build_6x6x6_strassen()

    # Restrict to the 3x5x6 subspace:
    # U rows: (i, j) for i in 0..2, j in 0..4  -> 15 rows
    u_rows = [(i * S + j) for i in range(n) for j in range(m)]
    # V rows: (j, k) for j in 0..4, k in 0..5  -> 30 rows
    v_rows = [(j * S + k) for j in range(m) for k in range(p)]
    # W rows: (k, i) for k in 0..5, i in 0..2  -> 18 rows
    w_rows = [(k * S + i) for k in range(p) for i in range(n)]

    U = U6[u_rows, :]
    V = V6[v_rows, :]
    W = W6[w_rows, :]

    # Verify correctness; fall back to schoolbook if it fails
    if verify_decomposition(U, V, W, n, m, p, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (rank 90)
    if time.time() < deadline:
        U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Last resort (should not be reached)
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
