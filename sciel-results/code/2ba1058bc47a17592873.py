"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                coeff = 0.0
                for r in range(rank):
                    coeff += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(coeff - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: split p=6 into 3 blocks of size 2.
    # Each 2x3x2 block uses rank 11 (Strassen 7 + scalar 4).
    # Total rank = 33.
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33

    # Local indices for a 2x3x2 block:
    #   A (2x3): (i, j) -> i*3 + j,  i in 0..1, j in 0..2
    #   B (3x2): (j, k) -> j*2 + k,  j in 0..2, k in 0..1
    #   C (2x2): (i, k) -> i*2 + k,  i in 0..1, k in 0..1
    #
    # Strassen terms for A[:,:2] (2x2) * B[:2,:] (2x2):
    #   A00=0, A01=1, A10=3, A11=4
    #   B00=0, B01=1, B10=2, B11=3
    terms = []

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})

    # Scalar terms for A[:,2] * B[2,:]
    # A02=2, A12=5, B20=4, B21=5
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})   # A02*B20 -> C00
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})   # A02*B21 -> C01
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})   # A12*B20 -> C10
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})   # A12*B21 -> C11

    # Fill the global matrices for 3 blocks
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx

            # U: global A index is the same as local
            for idx, val in term['u'].items():
                U[idx][col] = val

            # V: local (j, k) -> global (j, block*2 + k)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val

            # W: local (i, k) -> global (block*2 + k, i), flat = (block*2+k)*n + i
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
