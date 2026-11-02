import time
import numpy as np

def construct():
    t0 = time.time()
    limit = t0 + 110.0

    # Dimensions: multiply (2x2) by (2x16) -> (2x16)
    # n=2 (rows of A, cols of C), m=2 (cols of A, rows of B), p=16 (cols of B, cols of C)
    n, m, p = 2, 2, 16

    # Build rank-7 Strassen decomposition for 2x2 x 2x2 as three 4x7 matrices
    # Using numpy for clean tensor construction
    # U: shape (n*m, 7) = (4, 7), V: shape (m*p, 7) = (4, 7), W: shape (p*n, 7) = (4, 7)

    # Strassen 7 terms for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B11-B21)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)

    # A is 2x2, indexed A[i,j] where i=row, j=col -> flat index i*m+j = i*2+j
    # B is 2x2, indexed B[j,k] where j=row, k=col -> flat index j*p_local+k, p_local=2
    # C is 2x2, indexed C[i,k] where i=row, k=col -> flat index k*n+i in W convention W[k*n+i, r]

    # U[flat_A_idx, l] = coefficient for A[i,j] in term l
    # V[flat_B_idx, l] = coefficient for B[j,k] in term l (k is local 0 or 1)
    # W[flat_C_idx, l] = coefficient for C[i,k] in term l (k is local 0 or 1)

    # Term definitions: u_coeffs[(i,j)] -> val, v_coeffs[(j,k_local)] -> val, w_coeffs[(k_local, i)] -> val
    terms = []
    # M1
    terms.append((
        {(0,0): 1, (1,1): 1},      # u
        {(0,0): 1, (1,1): 1},      # v
        {(0,0): 1, (1,1): 1},      # w
    ))
    # M2
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
    ))
    # M3
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
    ))
    # M4
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
    ))
    # M5
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (1,0): 1},
    ))
    # M6
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
    ))
    # M7
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
    ))

    # Build base 2x2x2 factor matrices as numpy arrays
    # Base U: (4, 7), Base V: (4, 7), Base W: (4, 7)
    base_U = np.zeros((4, 7), dtype=np.float64)
    base_V = np.zeros((4, 7), dtype=np.float64)
    base_W = np.zeros((4, 7), dtype=np.float64)

    for l, (uc, vc, wc) in enumerate(terms):
        for (i, j), val in uc.items():
            base_U[i * m + j, l] = val
        for (j, kl), val in vc.items():
            base_V[j * 2 + kl, l] = val
        for (kl, ip), val in wc.items():
            base_W[kl * n + ip, l] = val

    # Now extend to 2x2 x 2x16: 8 blocks of 2x2 in the k dimension
    # The k dimension has p=16, split into 8 blocks of size 2
    # For each block q (0..7), the local k index kl maps to global k = 2*q + kl
    #
    # U is shared: shape (n*m, rank) = (4, 56)
    #   U[:, 7*q + l] = base_U[:, l]
    # V is extended: shape (m*p, rank) = (32, 56)
    #   V[j*p + (2*q + kl), 7*q + l] = base_V[j*2 + kl, l]
    # W is extended: shape (p*n, rank) = (32, 56)
    #   W[(2*q + kl)*n + ip, 7*q + l] = base_W[kl*n + ip, l]

    rank = 7 * 8  # 56
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)

    n_blocks = 8
    for q in range(n_blocks):
        if time.time() > limit:
            break
        r0 = 7 * q
        # U: same coefficients for each block
        U_full[:, r0:r0+7] = base_U

        # V: for each (j, kl) in base_V, place at (j, 2*q + kl) in V_full
        for l in range(7):
            for j in range(m):
                for kl in range(2):
                    val = base_V[j * 2 + kl, l]
                    if val != 0:
                        V_full[j * p + (2 * q + kl), r0 + l] = val

        # W: for each (kl, ip) in base_W, place at (2*q + kl, ip) in W_full
        for l in range(7):
            for kl in range(2):
                for ip in range(n):
                    val = base_W[kl * n + ip, l]
                    if val != 0:
                        W_full[(2 * q + kl) * n + ip, r0 + l] = val

    # Convert to lists of integers (entries are 0, 1, or -1)
    U_list = [[int(round(U_full[i, r])) for r in range(rank)] for i in range(n * m)]
    V_list = [[int(round(V_full[i, r])) for r in range(rank)] for i in range(m * p)]
    W_list = [[int(round(W_full[i, r])) for r in range(rank)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}
