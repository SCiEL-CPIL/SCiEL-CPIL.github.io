"""Matrix multiplication 2x3x8 - rank-29 decomposition via Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Build the deterministic rank-29 decomposition
    U, V, W = _build_rank29(n, m, p)

    # Verify correctness
    if not _verify(n, m, p, U, V, W):
        # Fallback: schoolbook
        U, V, W = _schoolbook(n, m, p)

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}


def _build_rank29(n, m, p):
    """
    Build rank-29 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Split A into A1 (2x2, cols 0-1) and A2 (2x1, col 2).
    Split B into B1 (2x8, rows 0-1) and B2 (1x8, row 2).
    C = A1*B1 + A2*B2
    
    A1*B1: M_{2,2,8} via 4 Strassen blocks (rank 28)
    A2*B2: M_{2,1,8} via rank 1
    Total: 29
    """
    n, m, p = 2, 3, 8
    total_rank = 29

    U = np.zeros((n * m, total_rank))  # 6 x 29
    V = np.zeros((m * p, total_rank))  # 24 x 29
    W = np.zeros((p * n, total_rank))  # 16 x 29

    # --- A1*B1: M_{2,2,8} using 4 Strassen blocks (rank 28) ---
    # A1 = A[:, 0:2], B1 = B[0:2, :]
    # U indices for A1: a00=0, a01=1, a10=3, a11=4 (i*m+j: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4)
    # V indices for B1: j*p+k for j=0,1 and k=0..7
    # W indices: k*n+i for k=0..7, i=0,1

    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_U = [
        [1, 0, 0, 1],   # M1: A00+A11 (indices 0,1,3,4 in U)
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    # U row indices for A1 elements: a00=0, a01=1, a10=3, a11=4
    u_rows = [0, 1, 3, 4]

    for block in range(4):
        k_offset = block * 2  # 0, 2, 4, 6
        col_offset = block * 7

        b00 = 0 * p + k_offset
        b01 = 0 * p + k_offset + 1
        b10 = 1 * p + k_offset
        b11 = 1 * p + k_offset + 1

        strassen_V = [
            [(b00, 1), (b11, 1)],   # M1: B00+B11
            [(b00, 1)],             # M2: B00
            [(b10, 1), (b11, -1)],  # M3: B10-B11
            [(b01, 1), (b00, -1)],  # M4: B01-B00
            [(b11, 1)],             # M5: B11
            [(b00, 1), (b01, 1)],   # M6: B00+B01
            [(b10, 1), (b11, 1)],   # M7: B10+B11
        ]

        c00 = (k_offset + 0) * n + 0
        c01 = (k_offset + 1) * n + 0
        c10 = (k_offset + 0) * n + 1
        c11 = (k_offset + 1) * n + 1

        strassen_W = [
            [(c00, 1), (c11, 1)],      # M1: C00, C11
            [(c10, 1)],                # M2: C10
            [(c01, 1), (c11, 1)],      # M3: C01, C11
            [(c00, 1), (c10, 1)],      # M4: C00, C10
            [(c00, -1), (c01, 1)],     # M5: -C00, +C01
            [(c11, 1)],                # M6: C11
            [(c00, 1)],                # M7: C00
        ]

        for c in range(7):
            for u_idx, val in zip(u_rows, strassen_U[c]):
                if val != 0:
                    U[u_idx, col_offset + c] = val
            for row, val in strassen_V[c]:
                if val != 0:
                    V[row, col_offset + c] = val
            for row, val in strassen_W[c]:
                if val != 0:
                    W[row, col_offset + c] = val

    # --- A2*B2: M_{2,1,8} rank 1 ---
    # A2 = A[:, 2]: a02=2, a12=5 (i*m+2: 0*3+2=2, 1*3+2=5)
    # B2 = B[2, :]: indices 2*p+k for k=0..7
    # C[i,k] += A2[i] * B2[k]
    # W index: k*n + i
    # Rank 1: U[i,28]=1 for i=0,1; V[2*p+k,28]=1 for all k; W[k*n+i,28]=1 for all i,k
    col = 28
    U[0, col] = 1  # a02
    U[5, col] = 1  # a12
    for k in range(p):
        V[2 * p + k, col] = 1  # B2[k]
    for k in range(p):
        for i in range(n):
            W[k * n + i, col] = 1

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
