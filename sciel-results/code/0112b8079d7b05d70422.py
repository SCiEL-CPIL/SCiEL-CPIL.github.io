"""Rewritten program for matrix multiplication 4x5x6 rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS
    total_cells = n * m * p  # 120

    # Schoolbook fallback
    U_school = np.zeros((n * m, total_cells), dtype=object)
    V_school = np.zeros((m * p, total_cells), dtype=object)
    W_school = np.zeros((p * n, total_cells), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist(),
        "rank": total_cells
    }

    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition: T[i,j,k] = sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for t in range(rank):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if total != 1:
                        return False
        return True

    def build_decomposition(block_specs, n, m, p):
        """
        Build a decomposition from block specifications.
        """
        total_rank = sum(len(pattern) for _, _, _, _, _, _, pattern in block_specs)
        U = np.zeros((n * m, total_rank), dtype=object)
        V = np.zeros((m * p, total_rank), dtype=object)
        W = np.zeros((p * n, total_rank), dtype=object)

        col = 0
        for (i_s, i_e, j_s, j_e, k_s, k_e, pattern) in block_specs:
            li = i_e - i_s
            lj = j_e - j_s
            lk = k_e - k_s
            for (u_c, v_c, w_c) in pattern:
                for (li_i, li_j), coeff in u_c.items():
                    gi = i_s + li_i
                    gj = j_s + li_j
                    U[gi * m + gj, col] = coeff
                for (lj_j, lk_k), coeff in v_c.items():
                    gj = j_s + lj_j
                    gk = k_s + lk_k
                    V[gj * p + gk, col] = coeff
                for (li_i, lk_k), coeff in w_c.items():
                    gi = i_s + li_i
                    gk = k_s + lk_k
                    W[gk * n + gi, col] = coeff
                col += 1

        return U, V, W, total_rank

    # Strassen 2x2x2 pattern
    def strassen_2x2x2():
        """Return list of (u_dict, v_dict, w_dict) for Strassen 2x2x2."""
        patterns = []
        # M1
        patterns.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # M2
        patterns.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}))
        # M3
        patterns.append(({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):-1}))
        # M4
        patterns.append(({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):-1}))
        # M5
        patterns.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}))
        # M6
        patterns.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        # M7
        patterns.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}))
        return patterns

    strassen_pat = strassen_2x2x2()

    # Check deadline
    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result

    # Strategy: 4x5x6 = 4x3x6 + 4x2x6
    # 4x3x6: j=0,1 use Strassen, j=2 schoolbook
    # 4x2x6: j=3,4 use Strassen
    block_specs_2 = []
    # Part A: 4x3x6 (j=0,1,2)
    # j=0,1: Strassen blocks
    for bi in range(2):
        for bk in range(3):
            block_specs_2.append((2*bi, 2*bi+2, 0, 2, 2*bk, 2*bk+2, strassen_pat))
    # j=2: schoolbook
    for i in range(n):
        for k in range(p):
            block_specs_2.append((i, i+1, 2, 3, k, k+1, [({(0,0):1}, {(0,0):1}, {(0,0):1})]))

    # Part B: 4x2x6 (j=3,4)
    for bi in range(2):
        for bk in range(3):
            block_specs_2.append((2*bi, 2*bi+2, 3, 5, 2*bk, 2*bk+2, strassen_pat))

    U2, V2, W2, rank2 = build_decomposition(block_specs_2, n, m, p)
    if verify(U2, V2, W2, n, m, p):
        if rank2 < best_result["rank"]:
            best_result = {
                "U": U2.tolist(),
                "V": V2.tolist(),
                "W": W2.tolist(),
                "rank": rank2
            }

    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result

    # Try another partition: 4x5x6 = 2x5x6 + 2x5x6
    block_specs_3 = []
    # 2x5x6: i=0,1 (1 block), j=0,1,2,3,4, k=0,1,2,3,4,5
    # j=0,1: Strassen, j=2,3: Strassen, j=4: schoolbook
    for bk in range(3):
        block_specs_3.append((0, 2, 0, 2, 2*bk, 2*bk+2, strassen_pat))
        block_specs_3.append((0, 2, 2, 4, 2*bk, 2*bk+2, strassen_pat))
        for k in range(2*bk, 2*bk+2):
            for i in range(2):
                block_specs_3.append((i, i+1, 4, 5, k, k+1, [({(0,0):1}, {(0,0):1}, {(0,0):1})]))
    
    # Second 2x5x6: i=2,3
    for bk in range(3):
        block_specs_3.append((2, 4, 0, 2, 2*bk, 2*bk+2, strassen_pat))
        block_specs_3.append((2, 4, 2, 4, 2*bk, 2*bk+2, strassen_pat))
        for k in range(2*bk, 2*bk+2):
            for i in range(2, 4):
                block_specs_3.append((i, i+1, 4, 5, k, k+1, [({(0,0):1}, {(0,0):1}, {(0,0):1})]))

    U3, V3, W3, rank3 = build_decomposition(block_specs_3, n, m, p)
    if verify(U3, V3, W3, n, m, p):
        if rank3 < best_result["rank"]:
            best_result = {
                "U": U3.tolist(),
                "V": V3.tolist(),
                "W": W3.tolist(),
                "rank": rank3
            }

    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result

    # Try 4x5x6 = 4x2x6 + 4x2x6 + 4x1x6
    block_specs_4 = []
    # 4x2x6 (j=0,1)
    for bi in range(2):
        for bk in range(3):
            block_specs_4.append((2*bi, 2*bi+2, 0, 2, 2*bk, 2*bk+2, strassen_pat))
    # 4x2x6 (j=2,3)
    for bi in range(2):
        for bk in range(3):
            block_specs_4.append((2*bi, 2*bi+2, 2, 4, 2*bk, 2*bk+2, strassen_pat))
    # 4x1x6 (j=4)
    for i in range(n):
        for k in range(p):
            block_specs_4.append((i, i+1, 4, 5, k, k+1, [({(0,0):1}, {(0,0):1}, {(0,0):1})]))

    U4, V4, W4, rank4 = build_decomposition(block_specs_4, n, m, p)
    if verify(U4, V4, W4, n, m, p):
        if rank4 < best_result["rank"]:
            best_result = {
                "U": U4.tolist(),
                "V": V4.tolist(),
                "W": W4.tolist(),
                "rank": rank4
            }

    return best_result
# EVOLVE-BLOCK-END
