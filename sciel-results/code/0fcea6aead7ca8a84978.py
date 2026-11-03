"""
Rank-98 Strassen-based decomposition for 4x4 * 4x8.

Uses the rank-7 Strassen decomposition for 2x2*2x2 as a building block,
composes it into a rank-49 decomposition for 4x4*4x4, and then uses
two copies for the two 4x4 halves of the 4x8 B matrix, giving rank 98.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    
    U: (4, 7) - rows indexed by i*2+j for A[i,j]
    V: (4, 7) - rows indexed by j*2+k for B[j,k]
    W: (4, 7) - rows indexed by k*2+i for C[i,k] (output convention W[k*n+i])
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M0: (a00+a11)(b00+b11) -> c00 + c11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M1: (a10+a11)(b00) -> c10 - c11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M2: (a00)(b01-b11) -> c01 + c11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M3: (a11)(b10-b00) -> c00 + c10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M4: (a00+a01)(b11) -> -c00 + c01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M5: (a10-a00)(b00+b01) -> c11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M6: (a01-a11)(b10+b11) -> c00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4_4x4():
    """
    Construct the rank-49 Strassen-based decomposition for 4x4 * 4x4.
    """
    n = 4
    U2, V2, W2 = _strassen_2x2_2x2()
    
    rank = 49
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    # Block-level Strassen specifications for 4x4x4
    # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1), each 2x2
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    # C_specs: how each P_k contributes to C blocks
    # C00 = P0 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    C_specs = [
        [(1, 0), (1, 3)],   # P0: +C00, +C11
        [(1, 2), (-1, 3)],  # P1: +C10, -C11
        [(1, 1), (1, 3)],   # P2: +C01, +C11
        [(1, 2)],           # P3: +C10
        [(-1, 0), (1, 1)],  # P4: -C00, +C01
        [(1, 3)],           # P5: +C11
        [(1, 0)],           # P6: +C00
    ]
    
    for k_p in range(7):
        for m_s in range(7):
            col = k_p * 7 + m_s
            
            # U: A input
            for coeff_a, block_idx in A_specs[k_p]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r
                        global_j = bj * 2 + c
                        global_a_idx = global_i * 4 + global_j
                        a_eff_idx = r * 2 + c
                        U[global_a_idx, col] += coeff_a * U2[a_eff_idx, m_s]
            
            # V: B input
            for coeff_b, block_idx in B_specs[k_p]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_j = bi * 2 + r
                        global_k = bj * 2 + c
                        global_b_idx = global_j * 4 + global_k
                        b_eff_idx = r * 2 + c
                        V[global_b_idx, col] += coeff_b * V2[b_eff_idx, m_s]
            
            # W: C output
            for coeff_c, c_block_idx in C_specs[k_p]:
                bi, bj = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r
                        global_k = bj * 2 + c
                        global_w_idx = global_k * 4 + global_i
                        p_eff_idx = r * 2 + c
                        W[global_w_idx, col] += coeff_c * W2[p_eff_idx, m_s]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify the decomposition."""
    U_arr = np.array(U, dtype=float) if not isinstance(U, np.ndarray) else U
    V_arr = np.array(V, dtype=float) if not isinstance(V, np.ndarray) else V
    W_arr = np.array(W, dtype=float) if not isinstance(W, np.ndarray) else W
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the rank-49 Strassen decomposition for 4x4*4x4
    U49, V49, W49 = _strassen_4x4_4x4()
    
    # Construct the rank-98 decomposition for 4x4*4x8
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First half: B columns 0-3
    U[:, 0:49] = U49
    V[0:16, 0:49] = V49
    W[0:16, 0:49] = W49
    
    # Second half: B columns 4-7
    U[:, 49:98] = U49
    V[16:32, 49:98] = V49
    W[16:32, 49:98] = W49
    
    # Verify with time check
    if time.time() < deadline - 10:
        valid, err = _verify_decomposition(U, V, W, n, m, p, rank)
        if not valid:
            return _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
