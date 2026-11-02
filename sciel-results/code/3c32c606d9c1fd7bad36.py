import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 140

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Build rank-26 decomposition
    def build_rank_26():
        rank = 26
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen_block(j_off, k_off):
            nonlocal col
            # 7 Strassen terms for 2x2x2 (i=0,1; j=j_off,j_off+1; k=k_off,k_off+1)
            # M1: (A00+A11)(B00+B11)
            U[0 * m + j_off][col] = 1; U[1 * m + j_off + 1][col] = 1
            V[j_off * p + k_off][col] = 1; V[j_off + 1 * p + k_off + 1][col] = 1
            W[k_off * n + 0][col] = 1; W[k_off + 1 * n + 1][col] = 1
            col += 1
            # M2: (A10+A11)(B00)
            U[1 * m + j_off][col] = 1; U[1 * m + j_off + 1][col] = 1
            V[j_off * p + k_off][col] = 1
            W[k_off * n + 1][col] = 1; W[k_off + 1 * n + 1][col] = -1
            col += 1
            # M3: (A00)(B11 - B01)
            U[0 * m + j_off][col] = 1
            V[j_off * p + k_off + 1][col] = 1; V[j_off + 1 * p + k_off + 1][col] = -1
            W[k_off + 1 * n + 0][col] = 1; W[k_off + 1 * n + 1][col] = 1
            col += 1
            # M4: (A11)(B00 - B10)
            U[1 * m + j_off + 1][col] = 1
            V[j_off + 1 * p + k_off][col] = 1; V[j_off * p + k_off][col] = -1
            W[k_off * n + 0][col] = 1; W[k_off * n + 1][col] = 1
            col += 1
            # M5: (A00+A01)(B11)
            U[0 * m + j_off][col] = 1; U[0 * m + j_off + 1][col] = 1
            V[j_off + 1 * p + k_off + 1][col] = 1
            W[k_off * n + 0][col] = -1; W[k_off + 1 * n + 0][col] = 1
            col += 1
            # M6: (A10-A00)(B00+B01)
            U[1 * m + j_off][col] = 1; U[0 * m + j_off][col] = -1
            V[j_off * p + k_off][col] = 1; V[j_off * p + k_off + 1][col] = 1
            W[k_off + 1 * n + 1][col] = 1
            col += 1
            # M7: (A01-A11)(B10+B11)
            U[0 * m + j_off + 1][col] = 1; U[1 * m + j_off + 1][col] = -1
            V[j_off + 1 * p + k_off][col] = 1; V[j_off + 1 * p + k_off + 1][col] = 1
            W[k_off * n + 0][col] = 1
            col += 1

        # Strassen block 1: j=1,2 k=0,1
        add_strassen_block(1, 0)
        # Strassen block 2: j=1,2 k=3,4
        add_strassen_block(1, 3)
        # col = 14

        # Schoolbook for j=0: 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        # col = 24

        # 2 columns for j=1,2 k=2
        # Column 24: handles (i,1,2) for i=0,1
        U[0 * m + 1][col] = 1  # i=0, j=1
        U[1 * m + 1][col] = 1  # i=1, j=1
        V[1 * p + 2][col] = 1  # j=1, k=2
        W[2 * n + 0][col] = 1  # k=2, i=0
        W[2 * n + 1][col] = 1  # k=2, i=1
        col += 1
        # Column 25: handles (i,2,2) for i=0,1
        U[0 * m + 2][col] = 1  # i=0, j=2
        U[1 * m + 2][col] = 1  # i=1, j=2
        V[2 * p + 2][col] = 1  # j=2, k=2
        W[2 * n + 0][col] = 1  # k=2, i=0
        W[2 * n + 1][col] = 1  # k=2, i=1
        col += 1

        assert col == 26, f"Expected 26 cols, got {col}"
        
        # Verify
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try rank-26 first
    if time_left() > 30:
        res_26 = build_rank_26()
        if res_26:
            if verify(res_26["U"], res_26["V"], res_26["W"]):
                return res_26

    # Fallback to schoolbook rank 30
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
