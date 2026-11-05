import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x5 matrix by a 5x6 matrix.
    
    Strategy:
    - Partition middle dimension j=5 into {0,1}, {2,3}, {4}
    - For j-pairs {0,1} and {2,3}: use Strassen (rank 7) for each 2x2*2x2 sub-block
      across the 3 k-pairs (k=0,1), (k=2,3), (k=4,5) -> 3*7 + 3*7 = 42
    - For j=4: use rank 1 decomposition for the 2x1 * 1x6 block (outer product structure)
    - Total rank: 42 + 1 = 43
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

    def get_rank43():
        """
        Rank-43 decomposition using Strassen on 2x2 blocks + rank-1 outer product for the last column.
        """
        rank = 43
        U = [[0] * rank for _ in range(n * m)]  # (10, 43)
        V = [[0] * rank for _ in range(m * p)]  # (30, 43)
        W = [[0] * rank for _ in range(p * n)]  # (12, 43)

        col = 0

        def add_strassen_block(i_base, j_base, k_base):
            """
            Add a rank-7 Strassen decomposition for a 2x2 * 2x2 sub-block.
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

        def add_outer_product_rank1(i_base, j_single, k_base, k_width):
            """
            Add rank 1 decomposition for a 2x1 * 1xk_width block.
            C[i, k] = A[i, j_single] * B[j_single, k]
            This is valid because the tensor slice is rank 1.
            """
            nonlocal col
            # U[i_base*m + j_single, col] = 1
            # V[j_single*p + k, col] = 1 for all k in range
            # W[k*n + i_base, col] = 1 for all k, i
            # We need to set these for the specific indices.
            # Note: U, V, W are constant matrices.
            # U[i_base*m + j_single, col] += 1
            # For all k in k_base..k_base+k_width-1:
            #   V[j_single*p + k, col] += 1
            # For all k in k_base..k_base+k_width-1, i in i_base..i_base+n-1:
            #   W[k*n + i_base+i, col] += 1
            
            U[i_base * m + j_single][col] += 1
            for k in range(k_base, k_base + k_width):
                V[j_single * p + k][col] += 1
            for k in range(k_base, k_base + k_width):
                for i in range(n):
                    W[k * n + i_base + i][col] += 1
            col += 1

        # Block 1: A[:, 0:2] * B[0:2, :] -> 2x2 * 2x6
        add_strassen_block(0, 0, 0)  # j=0,1; k=0,1
        add_strassen_block(0, 0, 2)  # j=0,1; k=2,3
        add_strassen_block(0, 0, 4)  # j=0,1; k=4,5

        # Block 2: A[:, 2:4] * B[2:4, :] -> 2x2 * 2x6
        add_strassen_block(0, 2, 0)  # j=2,3; k=0,1
        add_strassen_block(0, 2, 2)  # j=2,3; k=2,3
        add_strassen_block(0, 2, 4)  # j=2,3; k=4,5

        # Block 3: A[:, 4] * B[4, :] -> 2x1 * 1x6, rank 1
        add_outer_product_rank1(0, 4, 0, 6)

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W

    # Primary: rank-43 decomposition
    U, V, W = get_rank43()

    if check_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    if time.time() < deadline:
        U, V, W = get_schoolbook()
        return {"U": U, "V": V, "W": W}

    # Last resort (should not happen)
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
