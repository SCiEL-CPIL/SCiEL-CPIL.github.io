"""Matrix multiplication 2x2 x 2x8 tensor decomposition."""
DIMENSIONS = [2, 2, 8]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28

    # For each of 4 blocks, apply Strassen's 7 multiplications.
    # A layout: index 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # B layout: index j*p+k for row j, col k
    # C layout: index k*n+i for row k, col i

    tensors = []  # list of (u_vec, v_vec, w_vec)

    # Strassen formulas per block
    for q in range(4):
        # B columns for this block: 2q, 2q+1
        # C rows for this block: 2q, 2q+1
        b00 = q * 2      # j=0, k=2q
        b01 = q * 2 + 1  # j=0, k=2q+1
        b10 = p + 2 * q      # j=1, k=2q
        b11 = p + 2 * q + 1  # j=1, k=2q+1

        c00 = 2 * q      # k=2q, i=0
        c10 = 2 * q + 1  # k=2q+1, i=0
        c01 = 2 * q * n + 0  # k=2q, i=0 -> wait, C[k*n+i]
        # C index = k*n + i
        # k=2q, i=0 -> 2q*2 + 0 = 4q
        # k=2q, i=1 -> 4q+1
        # k=2q+1, i=0 -> (2q+1)*2 = 4q+2
        # k=2q+1, i=1 -> 4q+3

        c_00 = 4 * q      # k=2q, i=0
        c_01 = 4 * q + 1  # k=2q, i=1
        c_10 = 4 * q + 2  # k=2q+1, i=0
        c_11 = 4 * q + 3  # k=2q+1, i=1

        # M1: (A00+A11)(B00+B11) -> C00, C11
        u = np.zeros(n * m); u[0] = 1; u[3] = 1
        v = np.zeros(m * p); v[b00] = 1; v[b11] = 1
        w = np.zeros(p * n); w[c_00] = 1; w[c_11] = 1
        tensors.append((u, v, w))

        # M2: (A10+A11)B00 -> C10, -C11
        u = np.zeros(n * m); u[2] = 1; u[3] = 1
        v = np.zeros(m * p); v[b00] = 1
        w = np.zeros(p * n); w[c_10] = 1; w[c_11] = -1
        tensors.append((u, v, w))

        # M3: A00(B01-B11) -> C01, C11
        u = np.zeros(n * m); u[0] = 1
        v = np.zeros(m * p); v[b01] = 1; v[b11] = -1
        w = np.zeros(p * n); w[c_01] = 1; w[c_11] = 1
        tensors.append((u, v, w))

        # M4: A11(B10-B00) -> C00, C10
        u = np.zeros(n * m); u[3] = 1
        v = np.zeros(m * p); v[b10] = 1; v[b00] = -1
        w = np.zeros(p * n); w[c_00] = 1; w[c_10] = 1
        tensors.append((u, v, w))

        # M5: (A00+A01)B11 -> -C00, C01
        u = np.zeros(n * m); u[0] = 1; u[1] = 1
        v = np.zeros(m * p); v[b11] = 1
        w = np.zeros(p * n); w[c_00] = -1; w[c_01] = 1
        tensors.append((u, v, w))

        # M6: (A10-A00)(B00+B01) -> C11
        u = np.zeros(n * m); u[2] = 1; u[0] = -1
        v = np.zeros(m * p); v[b00] = 1; v[b01] = 1
        w = np.zeros(p * n); w[c_11] = 1
        tensors.append((u, v, w))

        # M7: (A01-A11)(B10+B11) -> C00
        u = np.zeros(n * m); u[1] = 1; u[3] = -1
        v = np.zeros(m * p); v[b10] = 1; v[b11] = 1
        w = np.zeros(p * n); w[c_00] = 1
        tensors.append((u, v, w))

    # Assemble U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r, (u, v, w) in enumerate(tensors):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
