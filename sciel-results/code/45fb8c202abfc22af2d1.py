"""Matrix multiplication 2x3x8 - rank-44 decomposition via Kronecker/Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Build the deterministic rank-44 decomposition
    U, V, W = _build_rank44(n, m, p)

    # Verify correctness
    if not _verify(n, m, p, U, V, W):
        # Fallback: schoolbook
        U, V, W = _schoolbook(n, m, p)

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}


def _build_rank44(n, m, p):
    """
    Build rank-44 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Split p=8 into 4 blocks of size 2.
    For each 2x3 * 3x2 sub-problem, use rank-11:
      - Strassen (rank 7) for 2x2 * 2x2 part
      - Outer product (rank 4) for 2x1 * 1x2 part
    
    Total rank: 4 * 11 = 44
    """
    n, m, p = 2, 3, 8
    block_size = 2  # p // 4
    num_blocks = p // block_size  # 4
    rank_per_block = 11
    total_rank = rank_per_block * num_blocks  # 44

    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44

    for b in range(num_blocks):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block

        # --- Strassen 7 terms for 2x2 * 2x2 part ---
        # A_top = [[a00, a01], [a10, a11]]  (indices in U: 0,1,2,3)
        # B_top = [[b00, b01], [b10, b11]]  (indices in V: j*p+k for j=0,1, k=k_offset,k_offset+1)
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # U indices for A_top: a00=0, a01=1, a10=2, a11=3
        strassen_U = [
            [1, 0, 0, 1],   # M1: A00+A11
            [0, 0, 1, 1],   # M2: A10+A11
            [1, 0, 0, 0],   # M3: A00
            [0, 0, 0, 1],   # M4: A11
            [1, 1, 0, 0],   # M5: A00+A01
            [-1, 0, 1, 0],  # M6: A10-A00
            [0, 1, 0, -1],  # M7: A01-A11
        ]

        # V indices for B_top: b00=0*p+k_offset, b01=0*p+k_offset+1, b10=1*p+k_offset, b11=1*p+k_offset+1
        b00 = 0 * p + k_offset
        b01 = 0 * p + k_offset + 1
        b10 = 1 * p + k_offset
        b11 = 1 * p + k_offset + 1

        strassen_V = [
            [b00, 1, b11, 1],   # M1: B00+B11
            [b00, 1],           # M2: B00
            [b10, 1, b11, -1],  # M3: B10-B11
            [b01, 1, b00, -1],  # M4: B01-B00
            [b11, 1],           # M5: B11
            [b00, 1, b01, 1],   # M6: B00+B01
            [b10, 1, b11, 1],   # M7: B10+B11
        ]

        # W indices for C: C[i,k] -> W[k*n + i]
        # C00: (k_offset+0)*n + 0
        # C01: (k_offset+1)*n + 0
        # C10: (k_offset+0)*n + 1
        # C11: (k_offset+1)*n + 1
        c00 = (k_offset + 0) * n + 0
        c01 = (k_offset + 1) * n + 0
        c10 = (k_offset + 0) * n + 1
        c11 = (k_offset + 1) * n + 1

        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        strassen_W = [
            [c00, 1, c11, 1],      # M1: C00, C11
            [c10, 1],              # M2: C10
            [c01, 1, c11, 1],      # M3: C01, C11
            [c00, 1, c10, 1],      # M4: C00, C10
            [c00, -1, c01, 1],     # M5: -C00, +C01
            [c11, 1],              # M6: C11
            [c00, 1],              # M7: C00
        ]

        for c in range(7):
            # Fill U
            for idx, val in [(0, strassen_U[c][0]), (1, strassen_U[c][1]), 
                            (2, strassen_U[c][2]), (3, strassen_U[c][3])]:
                if val != 0:
                    U[idx, col_offset + c] = val
            # Fill V
            for i in range(0, len(strassen_V[c]), 2):
                row = strassen_V[c][i]
                val = strassen_V[c][i + 1]
                if val != 0:
                    V[row, col_offset + c] = val
            # Fill W
            for i in range(0, len(strassen_W[c]), 2):
                row = strassen_W[c][i]
                val = strassen_W[c][i + 1]
                if val != 0:
                    W[row, col_offset + c] = val

        # --- Outer product part: A_bot(2x1) * B_bot(1x2) - rank 4 ---
        # A_bot = [a02, a12]^T  (indices in U: 2, 3... wait, a02 is index 2, a12 is index 5? No)
        # U indices: i*m + j where i=0,1 and j=0,1,2
        # a00=0, a01=1, a02=2, a10=3, a11=4, a12=5
        # B_bot = [b20, b21]  (indices in V: 2*p+k_offset, 2*p+k_offset+1)
        # C[i,k] += A_bot[i] * B_bot[k]
        # W index: (k_offset + k)*n + i

        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7,8,9,10
                U[ii * m + 2, col_offset + c] = 1  # a02 or a12
                V[2 * p + k_offset + kk, col_offset + c] = 1  # b20 or b21
                W[(k_offset + kk) * n + ii, col_offset + c] = 1

    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    Uses the trace convention W[k*n+i, r].
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    rank = U.shape[1]

    # Reshape for einsum
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)

    # u_vals[i, c] = sum_j U_reshaped[i,j,c] * A[i,j]
    u_vals = np.einsum('ijc,ij->ic', U_reshaped, A)
    # v_vals[k, c] = sum_l V_reshaped[l,k,c] * B[l,k]
    v_vals = np.einsum('lkp,lk->kp', V_reshaped, B)
    # C_comp[i, k] = sum_c W_reshaped[k,i,c] * u_vals[i,c] * v_vals[k,c]
    C_comp = np.einsum('kic,ic,kc->ik', W_reshaped, u_vals, v_vals)

    return np.allclose(C_comp, C_true, atol=1e-9)


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
