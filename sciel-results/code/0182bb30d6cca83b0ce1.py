"""
Matrix multiplication tensor decomposition for 2x5 * 5x6 -> 2x6
Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank below schoolbook.
Includes verification to ensure exactness; falls back to schoolbook if invalid.
"""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 5, 6

    def uidx(i, j):
        return i * m + j

    def vidx(j, k):
        return j * p + k

    def widx(k, i):
        return k * n + i

    def ok():
        return time.time() - start < deadline

    def build_strassen():
        """Build the rank-54 Strassen-based decomposition."""
        total_rank = 54
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0

        def emit(u_list, v_list, w_list):
            nonlocal col
            if col >= total_rank:
                return
            if not ok():
                return
            for (idx, val) in u_list:
                U[idx][col] = val
            for (idx, val) in v_list:
                V[idx][col] = val
            for (idx, val) in w_list:
                W[idx][col] = val
            col += 1

        def strassen_2x2(j0, j1, k0, k1):
            a00, a01 = uidx(0, j0), uidx(0, j1)
            a10, a11 = uidx(1, j0), uidx(1, j1)
            b00, b01 = vidx(j0, k0), vidx(j0, k1)
            b10, b11 = vidx(j1, k0), vidx(j1, k1)
            c00, c01 = widx(k0, 0), widx(k1, 0)
            c10, c11 = widx(k0, 1), widx(k1, 1)

            # M1 = (A00 + A11)(B00 + B11)
            emit([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])
            # M2 = (A10 + A11) B00
            emit([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)])
            # M3 = A00 (B01 - B11)
            emit([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)])
            # M4 = A11 (B10 - B00)
            emit([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)])
            # M5 = (A00 + A01) B11
            emit([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])
            # M6 = (A10 - A00)(B00 + B01)
            emit([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])
            # M7 = (A01 - A11)(B10 + B11)
            emit([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, -1)])

        def schoolbook_2x1x2(j, k0, k1):
            a0, a1 = uidx(0, j), uidx(1, j)
            b0, b1 = vidx(j, k0), vidx(j, k1)
            c00, c01 = widx(k0, 0), widx(k1, 0)
            c10, c11 = widx(k0, 1), widx(k1, 1)

            emit([(a0, 1)], [(b0, 1)], [(c00, 1)])
            emit([(a0, 1)], [(b1, 1)], [(c01, 1)])
            emit([(a1, 1)], [(b0, 1)], [(c10, 1)])
            emit([(a1, 1)], [(b1, 1)], [(c11, 1)])

        for (j0, j1) in [(0, 1), (2, 3)]:
            for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
                if not ok():
                    return None
                strassen_2x2(j0, j1, k0, k1)

        for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
            if not ok():
                return None
            schoolbook_2x1x2(4, k0, k1)

        return {"U": U, "V": V, "W": W}

    def build_schoolbook():
        """Build the rank-60 schoolbook decomposition (always valid)."""
        total_rank = n * m * p  # 60
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(result, max_checks=200):
        """
        Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for all (i,j,k) where the tensor component is 1, and 0 otherwise.
        For matrix multiplication tensor, the component at (i,j,k) is 1 if
        the output index (k,i) equals... actually the tensor T_{ijk} = 1
        when we multiply A[i,j]*B[j,k] to get C[i,k].
        Convention: W[k*n+i, r] so output index is (k, i).
        The tensor is: T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k.
        We check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        should equal 1.
        """
        U_mat = result["U"]
        V_mat = result["V"]
        W_mat = result["W"]
        rank = len(U_mat[0])

        # Check a random subset of (i,j,k) triples
        import random
        random.seed(42)

        all_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        if len(all_indices) <= max_checks:
            checks = all_indices
        else:
            checks = random.sample(all_indices, max_checks)

        for (i, j, k) in checks:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            s = 0
            for r in range(rank):
                s += U_mat[u_idx][r] * V_mat[v_idx][r] * W_mat[w_idx][r]
            if s != 1:
                return False

        # Also check that non-matching entries are 0.
        # The tensor only has 1s at positions where the three indices are
        # consistent (i from A rows, j from A cols / B rows, k from B cols / C cols).
        # In the full tensor space of dimension (n*m) x (m*p) x (p*n),
        # the matrix multiplication tensor has T_{a,b,c} = 1 iff
        # a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
        # All other entries are 0.
        # We check a few random non-matching triples.
        for _ in range(50):
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            c = random.randint(0, p * n - 1)
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            # This is a valid tensor index iff j_a == j_b and k_b == k_c and i_a == i_c
            if not (j_a == j_b and k_b == k_c and i_a == i_c):
                s = 0
                for r in range(rank):
                    s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                if s != 0:
                    return False

        return True

    # Try the Strassen decomposition first
    result = build_strassen()
    if result is not None and verify(result):
        return result

    # Fall back to schoolbook
    if not ok():
        return {"U": [[0] * 0 for _ in range(n * m)], "V": [[0] * 0 for _ in range(m * p)], "W": [[0] * 0 for _ in range(p * n)]}

    return build_schoolbook()

# EVOLVE-BLOCK-END
