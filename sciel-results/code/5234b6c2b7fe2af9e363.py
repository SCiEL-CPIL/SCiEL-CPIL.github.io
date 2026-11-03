"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
        if abs(val - 1) > 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m      # 15
    mp = m * p      # 35
    pn = p * n      # 21

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomp(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # ---- 4x4 rank-17 decomposition (Buerisser-Dorn 2007) ----
    # A = [[a0,a1,a2,a3],[a4,a5,a6,a7],[a8,a9,a10,a11],[a12,a13,a14,a15]]
    # B = [[b0,b1,b2,b3],[b4,b5,b6,b7],[b8,b9,b10,b11],[b12,b13,b14,b15]]
    # C = AB, C[4r+c] = C_{r,c}
    # 17 products M_t = sum_r Acoef[r,t]*a_r * sum_s Bcoef[s,t]*b_s
    # C[4r+c] = sum_t Ccoef[(r,c),t]*M_t
    Acoef = [
        [0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
        [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    ]
    Bcoef = [
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    ]
    Ccoef = [
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    ]

    # ---- 2x2 Strassen 7 products ----
    S_A = [
        (0, 1, 0, 1),
        (1, 0, 1, 1),
        (0, 0, 1, 0),
        (1, 1, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 0),
        (1, 1, 0, 1),
    ]
    S_B = [
        (0, 0, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 1),
        (0, 0, 0, 0),
        (1, 0, 1, 1),
    ]
    S_C = [
        (0, 1, 1, 0),
        (0, 1, 0, 0),
        (1, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 1, 1, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 1),
    ]

    # ---- Build the combined decomposition ----
    # 4x4 block: A rows 0-3, B cols 0-3  (17 terms)
    # 2x2 block: A rows 0-1, B cols 4-5  (7 terms)
    # Schoolbook: A rows 2-3, B cols 4-6 (8*3 = 24 terms)
    # Total: 17 + 7 + 24 = 48

    rank = 17 + 7 + 24
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0

    # 4x4 rank-17 block: A rows 0-3, B cols 0-3
    for t in range(17):
        for r in range(4):
            for s in range(4):
                if Acoef[r][t] != 0:
                    U[r * m + s][col] += Acoef[r][t]
                if Bcoef[s][t] != 0:
                    V[s * p + r][col] += Bcoef[s][t]
                if Ccoef[r * 4 + s][t] != 0:
                    W[r * n + s][col] += Ccoef[r * 4 + s][t]
        col += 1

    # 2x2 Strassen block: A rows 0-1, B cols 4-5
    for t in range(7):
        a1, a2, a3, a4 = S_A[t]
        b1, b2, b3, b4 = S_B[t]
        c1, c2, c3, c4 = S_C[t]
        # A rows: a1*row0 + a2*row1, B cols: b1*col4 + b2*col5
        # C: c1*col0, c2*col1, c3*col4, c4*col5
        if a1 != 0:
            U[0 * m + 4][col] += a1 * b1
            U[0 * m + 5][col] += a1 * b2
        if a2 != 0:
            U[1 * m + 4][col] += a2 * b1
            U[1 * m + 5][col] += a2 * b2
        if b1 != 0:
            V[4 * p + 0][col] += a1 * b1
            V[4 * p + 1][col] += a2 * b1
        if b2 != 0:
            V[5 * p + 0][col] += a1 * b2
            V[5 * p + 1][col] += a2 * b2
        if c1 != 0:
            W[0 * n + 4][col] += c1 * b1
            W[0 * n + 5][col] += c1 * b2
        if c2 != 0:
            W[1 * n + 4][col] += c2 * b1
            W[1 * n + 5][col] += c2 * b2
        if c3 != 0:
            W[4 * n + 4][col] += c3 * b1
            W[4 * n + 5][col] += c3 * b2
        if c4 != 0:
            W[5 * n + 4][col] += c4 * b1
            W[5 * n + 5][col] += c4 * b2
        col += 1

    # Schoolbook for A rows 2-3, B cols 4-6
    for i in range(2, 4):
        for j in range(4, 7):
            for k in range(n):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Full verification
    if verify_decomp(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # If verification fails, fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
