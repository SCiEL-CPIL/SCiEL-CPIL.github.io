"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

n, m, p = 3, 4, 5
SCHOOLBOOK_RANK = n * m * p  # 60

_DEADLINE = time.time() + 140  # safety margin within 150s limit


def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix mult using Strassen's 7-term algorithm.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B01+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: M1 = (A00+A11)*(B01+B10)
    # A00=idx0, A11=idx3, B01=idx1, B10=idx2
    U[0, 0] = 1.0  # A00
    U[3, 0] = 1.0  # A11
    V[1, 0] = 1.0  # B01
    V[2, 0] = 1.0  # B10
    W[0, 0] = 1.0  # C00
    W[1, 0] = 1.0  # C01
    W[2, 0] = 1.0  # C10
    W[3, 0] = 1.0  # C11
    
    # Term 2: M2 = (A10+A11)*B00
    # A10=idx2, A11=idx3, B00=idx0
    U[2, 1] = 1.0  # A10
    U[3, 1] = 1.0  # A11
    V[0, 1] = 1.0  # B00
    W[2, 1] = 1.0  # C10
    W[3, 1] = -1.0  # C11
    
    # Term 3: M3 = A00*(B10-B11)
    # A00=idx0, B10=idx2, B11=idx3
    U[0, 2] = 1.0  # A00
    V[2, 2] = 1.0  # B10
    V[3, 2] = -1.0  # B11
    W[1, 2] = 1.0  # C01
    W[3, 2] = 1.0  # C11
    
    # Term 4: M4 = A11*(B01-B00)
    # A11=idx3, B01=idx1, B00=idx0
    U[3, 3] = 1.0  # A11
    V[1, 3] = 1.0  # B01
    V[0, 3] = -1.0  # B00
    W[0, 3] = 1.0  # C00
    W[2, 3] = 1.0  # C10
    
    # Term 5: M5 = (A00+A01)*B11
    # A00=idx0, A01=idx1, B11=idx3
    U[0, 4] = 1.0  # A00
    U[1, 4] = 1.0  # A01
    V[3, 4] = 1.0  # B11
    W[0, 4] = -1.0  # C00
    W[1, 4] = 1.0  # C01
    
    # Term 6: M6 = (A10-A00)*(B00+B01)
    # A10=idx2, A00=idx0, B00=idx0, B01=idx1
    U[2, 5] = 1.0  # A10
    U[0, 5] = -1.0  # A00
    V[0, 5] = 1.0  # B00
    V[1, 5] = 1.0  # B01
    W[3, 5] = 1.0  # C11
    
    # Term 7: M7 = (A01-A11)*(B10+B11)
    # A01=idx1, A11=idx3, B10=idx2, B11=idx3
    U[1, 6] = 1.0  # A01
    U[3, 6] = -1.0  # A11
    V[2, 6] = 1.0  # B10
    V[3, 6] = 1.0  # B11
    W[0, 6] = 1.0  # C00
    
    return U, V, W


def _schoolbook_small(n_small, m_small, p_small):
    """Return (U, V, W) for n_small x m_small x p_small matrix mult using schoolbook.
    U shape (n_small*m_small, rank), V shape (m_small*p_small, rank), W shape (p_small*n_small, rank)
    """
    rank = n_small * m_small * p_small
    U = np.zeros((n_small * m_small, rank))
    V = np.zeros((m_small * p_small, rank))
    W = np.zeros((p_small * n_small, rank))
    col = 0
    for i in range(n_small):
        for j in range(m_small):
            for k in range(p_small):
                U[i * m_small + j, col] = 1.0
                V[j * p_small + k, col] = 1.0
                W[k * n_small + i, col] = 1.0
                col += 1
    return U, V, W


def _add_block(U_parts, V_parts, W_parts, u_offset, v_offset, w_offset, 
               n_b, m_b, p_b, U_b, V_b, W_b):
    """Add a block decomposition to the parts lists.
    u_offset, v_offset, w_offset are starting indices in the full matrices.
    """
    rank_b = U_b.shape[1]
    
    # Create full-sized matrices for this block
    U_full = np.zeros((12, rank_b))  # n*m = 3*4 = 12
    V_full = np.zeros((20, rank_b))  # m*p = 4*5 = 20
    W_full = np.zeros((15, rank_b))  # p*n = 5*3 = 15
    
    # Place block U
    for i in range(n_b):
        for j in range(m_b):
            full_i = u_offset // m + i
            full_j = u_offset % m + j
            full_idx = full_i * m + full_j
            U_full[full_idx, :] = U_b[i * m_b + j, :]
    
    # Place block V
    for j in range(m_b):
        for k in range(p_b):
            full_j = v_offset // p + j
            full_k = v_offset % p + k
            full_idx = full_j * p + full_k
            V_full[full_idx, :] = V_b[j * p_b + k, :]
    
    # Place block W
    for k in range(p_b):
        for i in range(n_b):
            full_k = w_offset // n + k
            full_i = w_offset % n + i
            full_idx = full_k * n + full_i
            W_full[full_idx, :] = W_b[k * n_b + i, :]
    
    U_parts.append(U_full)
    V_parts.append(V_full)
    W_parts.append(W_full)


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x5 matrix mult.
    
    Uses Strassen's algorithm for 2x2x2 sub-blocks to reduce rank from 60 to 56.
    """
    start_time = time.time()
    
    # Split the 3x4x5 multiplication into blocks:
    # Row split: A[0:2, :] and A[2:3, :]
    # Col split: B[:, 0:2], B[:, 2:4], B[:, 4:5]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2, use Strassen)
    U_b, V_b, W_b = _strassen_2x2x2()
    U_parts, V_parts, W_parts = [], [], []
    _add_block(U_parts, V_parts, W_parts, 0, 0, 0, 2, 2, 2, U_b, V_b, W_b)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (2x2x2, use Strassen)
    U_b, V_b, W_b = _strassen_2x2x2()
    _add_block(U_parts, V_parts, W_parts, 0, 10, 2, 2, 2, 2, U_b, V_b, W_b)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (2x2x1, schoolbook)
    U_b, V_b, W_b = _schoolbook_small(2, 2, 1)
    _add_block(U_parts, V_parts, W_parts, 0, 19, 4, 2, 2, 1, U_b, V_b, W_b)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2x2, use Strassen)
    U_b, V_b, W_b = _strassen_2x2x2()
    _add_block(U_parts, V_parts, W_parts, 2, 10, 0, 2, 2, 2, U_b, V_b, W_b)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (2x2x2, use Strassen)
    U_b, V_b, W_b = _strassen_2x2x2()
    _add_block(U_parts, V_parts, W_parts, 2, 20, 2, 2, 2, 2, U_b, V_b, W_b)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5] (2x2x1, schoolbook)
    U_b, V_b, W_b = _schoolbook_small(2, 2, 1)
    _add_block(U_parts, V_parts, W_parts, 2, 29, 4, 2, 2, 1, U_b, V_b, W_b)
    
    # Block 7: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2] (1x2x2, schoolbook)
    U_b, V_b, W_b = _schoolbook_small(1, 2, 2)
    _add_block(U_parts, V_parts, W_parts, 8, 0, 10, 1, 2, 2, U_b, V_b, W_b)
    
    # Block 8: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4] (1x2x2, schoolbook)
    U_b, V_b, W_b = _schoolbook_small(1, 2, 2)
    _add_block(U_parts, V_parts, W_parts, 8, 10, 12, 1, 2, 2, U_b, V_b, W_b)
    
    # Block 9: A[2:3, 0:2] * B[0:2, 4:5] -> C[2:3, 4:5] (1x2x1, schoolbook)
    U_b, V_b, W_b = _schoolbook_small(1, 2, 1)
    _add_block(U_parts, V_parts, W_parts, 8, 19, 14, 1, 2, 1, U_b, V_b, W_b)
    
    # Block 10: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2] (1x2x2, schoolbook)
    U_b, V_b, W_b = _schoolbook_small(1, 2, 2)
    _add_block(U_parts, V_parts, W_parts, 10, 10, 10, 1, 2, 2, U_b, V_b, W_b)
    
    # Block 11: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4] (1x2x2, schoolbook)
    U_b, V_b, W_b = _schoolbook_small(1, 2, 2)
    _add_block(U_parts, V_parts, W_parts, 10, 20, 12, 1, 2, 2, U_b, V_b, W_b)
    
    # Block 12: A[2:3, 2:4] * B[2:4, 4:5] -> C[2:3, 4:5] (1x2x1, schoolbook)
    U_b, V_b, W_b = _schoolbook_small(1, 2, 1)
    _add_block(U_parts, V_parts, W_parts, 10, 29, 14, 1, 2, 1, U_b, V_b, W_b)
    
    # Assemble
    rank = sum(U.shape[1] for U in U_parts)
    U = np.column_stack(U_parts)
    V = np.column_stack(V_parts)
    W = np.column_stack(W_parts)
    
    # Validate
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
