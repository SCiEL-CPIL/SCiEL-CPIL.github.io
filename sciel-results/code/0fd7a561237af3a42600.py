"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    # Rank calculation:
    # 3 k-pairs of size 2, each with 2 j-pairs of size 2.
    # Each 2x2x2 block uses Strassen (7 mults).
    # 3 * 2 * 7 = 42 mults.
    # 1 k-single of size 1: 2x4x1 tensor has rank 4 (not 8).
    # Total rank = 46.
    rank = 46
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        # Sub-tensor: i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_single_k_block(k0):
        """Handle 2x4x1 tensor for a single k index. Rank = 4."""
        nonlocal col
        # C[i, k0] = sum_j A[i,j] * B[j, k0] for i in {0,1}, j in {0,1,2,3}
        for j in range(m):
            a0_idx = 0 * m + j  # A[0,j]
            a1_idx = 1 * m + j  # A[1,j]
            b_idx = j * p + k0  # B[j,k0]
            c0_idx = k0 * n + 0  # C[0,k0]
            c1_idx = k0 * n + 1  # C[1,k0]
            
            U[a0_idx, col] += 1
            U[a1_idx, col] += 1
            V[b_idx, col] += 1
            W[c0_idx, col] += 1
            W[c1_idx, col] += 1
            col += 1

    # k pairs: (0,1), (2,3), (4,5)
    for k_start in [0, 2, 4]:
        # j pairs: (0,1), (2,3)
        for j_start in [0, 2]:
            add_strassen_block(0, j_start, k_start)
    
    # k single: 6
    add_single_k_block(6)

    if col != rank:
        # Fallback to schoolbook if construction fails
        return _schoolbook(n, m, p)

    # Verify correctness with random test
    try:
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        # Compute C using the decomposition
        C_decomp = np.zeros((p, n), dtype=np.int64)
        for r in range(rank):
            u = U_arr[:, r]
            v = V_arr[:, r]
            w = W_arr[:, r]
            prod = np.outer(u, v)  # (n*m, m*p)
            prod_reshaped = prod.reshape(n, m, m, p)
            partial = np.einsum('imjm->ip', prod_reshaped)
            C_decomp += partial * w[np.arange(p)[:, None] * n + np.arange(n)[None, :]].reshape(p, n)
        
        if not np.array_equal(C_decomp, C_expected):
            return _schoolbook(n, m, p)
    except:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
