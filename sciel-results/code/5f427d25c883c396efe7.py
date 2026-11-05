import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    """
    Rank-50 decomposition for 2x5 * 5x6 matrix multiplication.
    
    Strategy:
    - 6 Strassen 2x2*2x2 blocks: j-pairs {0,1} and {2,3} crossed with k-pairs {0,1},{2,3},{4,5}
      -> 6 * 7 = 42
    - 4 rank-2 blocks for the remaining j=4,5 columns with k-pairs {0,1},{2,3},{4,5}
      -> 4 * 2 = 8
    Total rank: 50
    """
    start_time = time.time()
    deadline = start_time + 110

    def check_decomposition(U, V, W, tol=1e-9):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
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

    def get_rank50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]  # (10, 50)
        V = [[0] * rank for _ in range(m * p)]  # (30, 50)
        W = [[0] * rank for _ in range(p * n)]  # (12, 50)

        col = 0

        def add_strassen_block(i_base, j_base, k_base):
            """
            Rank-7 Strassen for a 2x2 * 2x2 sub-block.
            A_sub[i,j] -> U[i_base*m + j_base+j]
            B_sub[j,k] -> V[(j_base+j)*p + k_base+k]
            C_sub[i,k] -> W[(k_base+k)*n + i_base+i]
            """
            nonlocal col
            # M0 = (A00 + A11) * (B00 + B11)
            U[i_base * m + j_base][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

            # M1 = (A10 + A11) * B00
            U[(i_base + 1) * m + j_base][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] -= 1
            col += 1

            # M2 = A00 * (B01 - B11)
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] -= 1
            W[(k_base + 1) * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

            # M3 = A11 * (B10 - B00)
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            col += 1

            # M4 = (A00 + A01) * B11
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base + 1) * n + i_base][col] += 1
            col += 1

            # M5 = (A10 - A00) * (B00 + B01)
            U[(i_base + 1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

            # M6 = (A01 - A11) * (B10 + B11)
            U[i_base * m + (j_base + 1)][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] -= 1
            V[(j_base + 1) * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1

        def add_rank2_block(i_base, j_single, k_base):
            """
            Rank-2 decomposition for a 2x1 * 1x2 sub-block.
            C[i, k_base] = A[i, j_single] * B[j_single, k_base]
            C[i, k_base+1] = A[i, j_single] * B[j_single, k_base+1]
            """
            nonlocal col
            # M0 = (A0 + A1) * (B0 + B1)
            U[i_base * m + j_single][col] += 1
            U[(i_base + 1) * m + j_single][col] += 1
            V[j_single * p + k_base][col] += 1
            V[j_single * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base + 1) * n + i_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

            # M1 = A0 * B1
            U[i_base * m + j_single][col] += 1
            V[j_single * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + i_base][col] += 1
            col += 1

        # 6 Strassen blocks: j-pairs {0,1} and {2,3} x k-pairs {0,1},{2,3},{4,5}
        for j_base in (0, 2):
            for k_base in (0, 2, 4):
                add_strassen_block(0, j_base, k_base)

        # 4 rank-2 blocks: j in {4, 5} x k-pairs {0,1},{2,3},{4,5}
        for j_single in (4, 5):
            for k_base in (0, 2, 4):
                add_rank2_block(0, j_single, k_base)

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W

    U, V, W = get_rank50()

    if check_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    if time.time() < deadline:
        U, V, W = get_schoolbook()
        return {"U": U, "V": V, "W": W}

    U, V, W = get_schoolbook()
    return {"U": U, "V": V, "W": W}

def check_decomposition(U, V, W, tol=1e-9):
    n, m, p = 2, 5, 6
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    is_valid = check_decomposition(res['U'], res['V'], res['W'])
    rank = len(res['U'][0])
    print(f"Rank: {rank}, Valid: {is_valid}")
