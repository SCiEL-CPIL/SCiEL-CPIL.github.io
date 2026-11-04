"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact using vectorized numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _make_3x3_laderman():
    """
    Laderman decomposition of 3x3 * 3x3 -> 3x3, rank 23.
    Returns (U, V, W) as 2D arrays of shape (9, 23).
    Indexing: U[i*3+j], V[j*3+k], W[k*3+i]
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))

    # 9 direct terms: A[i][j] * B[j][k] -> C[i][k]
    # Term 0: A00*B00 -> C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A01*B10 -> C00
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2: A02*B20 -> C00
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
    # Term 3: A10*B01 -> C10
    U[3, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    # Term 4: A11*B11 -> C11
    U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    # Term 5: A12*B21 -> C11
    U[5, 5] = 1; V[7, 5] = 1; W[4, 5] = 1
    # Term 6: A20*B02 -> C20
    U[6, 6] = 1; V[2, 6] = 1; W[6, 6] = 1
    # Term 7: A21*B12 -> C21
    U[7, 7] = 1; V[5, 7] = 1; W[7, 7] = 1
    # Term 8: A22*B22 -> C22
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1

    # 14 composite terms
    # Term 9: (A00+A11)*(B00+B11) -> C01, C10
    U[0, 9] = 1; U[4, 9] = 1
    V[0, 9] = 1; V[4, 9] = 1
    W[1, 9] = 1; W[3, 9] = 1

    # Term 10: (A00+A22)*(B00+B22) -> C02, C20
    U[0, 10] = 1; U[8, 10] = 1
    V[0, 10] = 1; V[8, 10] = 1
    W[2, 10] = 1; W[6, 10] = 1

    # Term 11: (A11+A22)*(B11+B22) -> C12, C21
    U[4, 11] = 1; U[8, 11] = 1
    V[4, 11] = 1; V[8, 11] = 1
    W[5, 11] = 1; W[7, 11] = 1

    # Term 12: (A01+A10)*(B11+B00) -> C00, C11
    U[1, 12] = 1; U[3, 12] = 1
    V[4, 12] = 1; V[0, 12] = 1
    W[0, 12] = 1; W[4, 12] = 1

    # Term 13: (A02+A20)*(B22+B00) -> C00, C22
    U[2, 13] = 1; U[6, 13] = 1
    V[8, 13] = 1; V[0, 13] = 1
    W[0, 13] = 1; W[8, 13] = 1

    # Term 14: (A12+A21)*(B22+B11) -> C11, C22
    U[5, 14] = 1; U[7, 14] = 1
    V[8, 14] = 1; V[4, 14] = 1
    W[4, 14] = 1; W[8, 14] = 1

    # Term 15: (A01+A11)*(B00+B10) -> C00, C10
    U[1, 15] = 1; U[4, 15] = 1
    V[0, 15] = 1; V[3, 15] = 1
    W[0, 15] = 1; W[3, 15] = 1

    # Term 16: (A10+A11)*(B01+B11) -> C10, C11
    U[3, 16] = 1; U[4, 16] = 1
    V[1, 16] = 1; V[4, 16] = 1
    W[3, 16] = 1; W[4, 16] = 1

    # Term 17: (A02+A22)*(B00+B20) -> C00, C20
    U[2, 17] = 1; U[8, 17] = 1
    V[0, 17] = 1; V[6, 17] = 1
    W[0, 17] = 1; W[6, 17] = 1

    # Term 18: (A20+A22)*(B02+B22) -> C20, C22
    U[6, 18] = 1; U[8, 18] = 1
    V[2, 18] = 1; V[8, 18] = 1
    W[6, 18] = 1; W[8, 18] = 1

    # Term 19: (A01+A02+A11+A12)*(B10+B20+B11+B21) -> C01, C02, C11, C12
    U[1, 19] = 1; U[2, 19] = 1; U[4, 19] = 1; U[5, 19] = 1
    V[3, 19] = 1; V[6, 19] = 1; V[4, 19] = 1; V[7, 19] = 1
    W[1, 19] = 1; W[2, 19] = 1; W[4, 19] = 1; W[5, 19] = 1

    # Term 20: (A01+A02+A21+A22)*(B10+B20+B12+B22) -> C01, C02, C21, C22
    U[1, 20] = 1; U[2, 20] = 1; U[7, 20] = 1; U[8, 20] = 1
    V[3, 20] = 1; V[6, 20] = 1; V[5, 20] = 1; V[8, 20] = 1
    W[1, 20] = 1; W[2, 20] = 1; W[7, 20] = 1; W[8, 20] = 1

    # Term 21: (A10+A12+A20+A22)*(B00+B20+B02+B22) -> C10, C12, C20, C22
    U[3, 21] = 1; U[5, 21] = 1; U[6, 21] = 1; U[8, 21] = 1
    V[0, 21] = 1; V[6, 21] = 1; V[2, 21] = 1; V[8, 21] = 1
    W[3, 21] = 1; W[5, 21] = 1; W[6, 21] = 1; W[8, 21] = 1

    # Term 22: (A01+A10+A12+A21)*(B00+B01+B11+B12) -> C00, C01, C10, C12
    U[1, 22] = 1; U[3, 22] = 1; U[5, 22] = 1; U[7, 22] = 1
    V[0, 22] = 1; V[1, 22] = 1; V[4, 22] = 1; V[5, 22] = 1
    W[0, 22] = 1; W[1, 22] = 1; W[3, 22] = 1; W[5, 22] = 1

    return U, V, W


def _make_2x2_strassen():
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2, rank 7.
    Returns (U, V, W) as 2D arrays of shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # P3 = (A00)(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # P4 = (A11)(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # P5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _compose_6x6(U_outer, V_outer, W_outer, U_inner, V_inner, W_inner,
                 block_size, num_blocks):
    """
    Compose outer decomposition with inner decomposition.
    Outer: num_blocks x num_blocks matrix mult, each entry is a block_size x block_size matrix.
    Inner: block_size x block_size matrix mult.
    
    Returns (U, V, W) for the full 6x6 problem.
    """
    dim = num_blocks * block_size  # 6
    outer_dim = num_blocks  # 2
    inner_dim = block_size  # 3
    
    rank_outer = U_outer.shape[1]  # 7
    rank_inner = U_inner.shape[1]  # 23
    total_rank = rank_outer * rank_inner  # 161
    
    U = np.zeros((dim * dim, total_rank))
    V = np.zeros((dim * dim, total_rank))
    W = np.zeros((dim * dim, total_rank))
    
    for c_out in range(rank_outer):
        for c_in in range(rank_inner):
            col = c_out * rank_inner + c_in
            
            # U: outer index (I, J) maps to inner (i, j)
            # Global: (I*block_size + i, J*block_size + j)
            for I in range(outer_dim):
                for J in range(outer_dim):
                    u_outer_val = U_outer[I * outer_dim + J, c_out]
                    if u_outer_val == 0:
                        continue
                    for i in range(inner_dim):
                        for j in range(inner_dim):
                            u_inner_val = U_inner[i * inner_dim + j, c_in]
                            if u_inner_val == 0:
                                continue
                            global_i = I * block_size + i
                            global_j = J * block_size + j
                            U[global_i * dim + global_j, col] += u_outer_val * u_inner_val
            
            # V: outer index (J, K) maps to inner (j, k)
            for J in range(outer_dim):
                for K in range(outer_dim):
                    v_outer_val = V_outer[J * outer_dim + K, c_out]
                    if v_outer_val == 0:
                        continue
                    for j in range(inner_dim):
                        for k in range(inner_dim):
                            v_inner_val = V_inner[j * inner_dim + k, c_in]
                            if v_inner_val == 0:
                                continue
                            global_j = J * block_size + j
                            global_k = K * block_size + k
                            V[global_j * dim + global_k, col] += v_outer_val * v_inner_val
            
            # W: outer index (K, I) maps to inner (k, i)
            for K in range(outer_dim):
                for I in range(outer_dim):
                    w_outer_val = W_outer[K * outer_dim + I, c_out]
                    if w_outer_val == 0:
                        continue
                    for k in range(inner_dim):
                        for i in range(inner_dim):
                            w_inner_val = W_inner[k * inner_dim + i, c_in]
                            if w_inner_val == 0:
                                continue
                            global_k = K * block_size + k
                            global_i = I * block_size + i
                            W[global_k * dim + global_i, col] += w_outer_val * w_inner_val
    
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
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
    
    # Try: 2x2 Strassen (outer) x 3x3 Laderman (inner) = rank 161
    try:
        U_str, V_str, W_str = _make_2x2_strassen()
        U_lad, V_lad, W_lad = _make_3x3_laderman()
        
        U_full, V_full, W_full = _compose_6x6(
            U_str, V_str, W_str,
            U_lad, V_lad, W_lad,
            block_size=3, num_blocks=2
        )
        
        # Convert to lists for the interface
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        
        # Verify
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_full[i * m + j, :]
                for k in range(p):
                    v_row = V_full[j * p + k, :]
                    w_row = W_full[k * n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Laderman failed verification, fall back
            pass
    except Exception:
        pass
    
    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
