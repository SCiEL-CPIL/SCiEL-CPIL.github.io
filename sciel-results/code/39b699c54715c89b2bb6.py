"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 5
    # U: (n*m, rank) = (15, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (15, rank), index k*n + i

    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Helper: add a Strassen block for 2x2x2 sub-tensor
    # i_base, j_base, k_base: starting indices in the 5x5 grid
    # For the 3x5x5 problem:
    #   U index: i*m + j
    #   V index: j*p + k
    #   W index: k*n + i
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # Strassen 7 terms for 2x2x2:
        # M1 = A00*(B00+B11)
        # M2 = (A00+A10)*B11
        # M3 = (A01-A11)*(B10+B11)
        # M4 = A11*(B00-B10)
        # M5 = (A00+A11)*(B00+B11)
        # M6 = (A10-A00)*(B01+B11)
        # M7 = (A01-A11)*(B10+B11)  -- wait, let me use standard Strassen
        #
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)(B00+B01)
        # M6 = (A01+A11)(B10+B11)
        # M7 = (A00-A10)(B01-B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # Each term is a rank-1: U_row * V_col * W_row
        # Term structure: sum over s of (sum_i u_ia * A_ia) * (sum_j v_jb * B_jb) -> W output
        # Actually for tensor decomposition:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor T[i,j,k] = delta for MM
        # Decomposition: T = sum_r U[:,r] V[:,r] W[:,r] (outer product)
        # where U[i,j,r], V[j,k,r], W[k,i,r]
        # C[i,k] = sum_r sum_j U[i,j,r] V[j,k,r] W[k,i,r] * A[i,j] * B[j,k]
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,j,k} appropriate

        # For a Strassen block on indices i in {i0,i1}, j in {j0,j1}, k in {k0,k1}:
        # We're replacing the 8 schoolbook terms with 7 Strassen terms.
        # Each Strassen term r has coefficients for the U, V, W factors.

        i0, i1 = i_base, i_base + 1
        j0, j1 = j_base, j_base + 1
        k0, k1 = k_base, k_base + 1

        # U indices: i*m + j
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1

        # V indices: j*p + k
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1

        # W indices: k*n + i
        w00 = k0 * n + i0
        w01 = k0 * n + i1
        w10 = k1 * n + i0
        w11 = k1 * n + i1

        # Strassen terms (7 of them):
        # Each term: list of (U_index, coeff), list of (V_index, coeff), list of (W_index, coeff)

        terms = [
            # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
            {
                'U': [(u00, 1), (u11, 1)],
                'V': [(v00, 1), (v11, 1)],
                'W': [(w00, 1), (w11, 1)],
            },
            # M2 = (A10+A11)B00 -> contributes to C10
            {
                'U': [(u10, 1), (u11, 1)],
                'V': [(v00, 1)],
                'W': [(w10, 1)],
            },
            # M3 = A00(B10-B11) -> contributes to C01
            {
                'U': [(u00, 1)],
                'V': [(v10, 1), (v11, -1)],
                'W': [(w01, 1)],
            },
            # M4 = A11(B01-B00) -> contributes to C00 and C10
            {
                'U': [(u11, 1)],
                'V': [(v01, 1), (v00, -1)],
                'W': [(w00, 1), (w10, 1)],
            },
            # M5 = (A00+A10)(B00+B01) -> contributes to C00 and C01
            {
                'U': [(u00, 1), (u10, 1)],
                'V': [(v00, 1), (v01, 1)],
                'W': [(w00, 1), (w01, 1)],
            },
            # M6 = (A01+A11)(B10+B11) -> contributes to C11
            {
                'U': [(u01, 1), (u11, 1)],
                'V': [(v10, 1), (v11, 1)],
                'W': [(w11, 1)],
            },
            # M7 = (A00-A10)(B01-B11) -> contributes to C00
            {
                'U': [(u00, 1), (u10, -1)],
                'V': [(v01, 1), (v11, -1)],
                'W': [(w00, 1)],
            },
        ]

        for term in terms:
            for (idx, coeff) in term['U']:
                U[idx][col] += coeff
            for (idx, coeff) in term['V']:
                V[idx][col] += coeff
            for (idx, coeff) in term['W']:
                W[idx][col] += coeff
            col += 1

    # Identify 2x2 blocks in the j-k plane (since i has 3 values, we can do blocks per row)
    # Actually, the Strassen block operates on 2x2x2 sub-tensors (i,j,k all 2x2x2).
    # In a 3x5x5 problem, we can fit 1 full 2x2x2 block in i (since 3 >= 2),
    # and in j,k we can fit 2x2 blocks.
    #
    # Let's place 4 disjoint 2x2x2 blocks:
    # Block positions (i_base, j_base, k_base):
    # (0, 0, 0): i in {0,1}, j in {0,1}, k in {0,1}
    # (0, 0, 2): i in {0,1}, j in {0,1}, k in {2,3}
    # (0, 2, 0): i in {0,1}, j in {2,3}, k in {0,1}
    # (0, 2, 2): i in {0,1}, j in {2,3}, k in {2,3}
    #
    # These 4 blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Remaining schoolbook: all (i,j,k) not in these blocks
    # i=2: all j,k (25 terms)
    # i in {0,1}, j=4: all k (2*5=10 terms)
    # i in {0,1}, k=4: all j except j=4 (already counted) -> j in {0,1,2,3} (2*4=8 terms)
    # Total remaining: 25 + 10 + 8 = 43
    # Total rank: 4*7 + 43 = 28 + 43 = 71. Correct.

    blocks = [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)]
    for (ib, jb, kb) in blocks:
        if time.time() > deadline:
            break
        add_strassen_block(ib, jb, kb)

    # Now add schoolbook terms for remaining (i,j,k)
    # Mark which (i,j,k) are covered by Strassen blocks
    covered = set()
    for (ib, jb, kb) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((ib + di, jb + dj, kb + dk))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
