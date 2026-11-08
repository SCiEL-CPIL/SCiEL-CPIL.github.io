"""Matrix multiplication tensor decomposition for 2x5 by 5x6."""
from collections import defaultdict


def construct():
    n, m, p = 2, 5, 6

    # Strassen pattern for 2x2 blocks: 7 terms
    # Each term: list of (u_index, v_index, w_index, sign)
    # u_index = row*m + j, v_index = j*p + k, w_index = k*n + row
    # For block (j0,j1) x (k0,k1):
    # We encode relative offsets within the block
    strassen_terms = [
        # (u_off, v_off, w_off, sign)
        # P1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ((0, 0, 0, 1), (1, 1, 2, 1)),
        # P2: (A10+A11)B00 -> C10 += 1, C11 -= 1
        ((2, 0, 2, 1), (3, 0, 2, -1)),
        # P3: A00(B11-B01) -> C01 += 1, C11 += 1
        ((0, 3, 1, 1), (0, 2, 3, 1)),
        # P4: A11(B00-B10) -> C00 += 1, C10 += 1
        ((3, 2, 0, 1), (3, 0, 2, 1)),
        # P5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        ((0, 3, 0, -1), (1, 3, 1, 1)),
        # P6: (A10-A00)(B00+B01) -> C11 += 1
        ((2, 0, 3, 1), (0, 0, 3, -1)),
        # P7: (A01-A11)(B10+B11) -> C00 += 1
        ((1, 2, 0, 1), (3, 2, 0, -1)),
    ]

    # Build sparse entries
    u_entries = defaultdict(int)
    v_entries = defaultdict(int)
    w_entries = defaultdict(int)

    col = 0

    # Apply Strassen to 2x2 blocks for j in {0,1}x{0,1} and {2,3}x{0,1}
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]

    for (ja, jb) in j_blocks:
        for (ka, kb) in k_blocks:
            for (uo, vo, wo, _sign) in [t for term in strassen_terms for t in term]:
                # u index: row offset in {0,1} maps to u_off 0->row0, 1->row1, 2->row0, 3->row1
                # Actually u_off: 0=(0,ja), 1=(0,jb), 2=(1,ja), 3=(1,jb)
                row_u, j_u = (uo // 2, uo % 2)
                j_idx = ja if j_u == 0 else jb
                u_idx = row_u * m + j_idx

                # v_off: 0=(ja,ka), 1=(ja,kb), 2=(jb,ka), 3=(jb,kb)
                j_v, k_v = (vo // 2, vo % 2)
                j_idx_v = ja if j_v == 0 else jb
                k_idx_v = ka if k_v == 0 else kb
                v_idx = j_idx_v * p + k_idx_v

                # w_off: 0=(ka,0), 1=(ka,1), 2=(kb,0), 3=(kb,1)
                k_w, row_w = (wo // 2, wo % 2)
                k_idx_w = ka if k_w == 0 else kb
                w_idx = k_idx_w * n + row_w

                u_entries[(u_idx, col)] += 1
                v_entries[(v_idx, col)] += 1
                w_entries[(w_idx, col)] += 1

            col += 1

    # Handle remaining j=4: straightforward 12 products
    for k in range(p):
        # C[0,k] += A[0,4]*B[4,k]
        u_entries[(0 * m + 4, col)] += 1
        v_entries[(4 * p + k, col)] += 1
        w_entries[(k * n + 0, col)] += 1
        col += 1
        # C[1,k] += A[1,4]*B[4,k]
        u_entries[(1 * m + 4, col)] += 1
        v_entries[(4 * p + k, col)] += 1
        w_entries[(k * n + 1, col)] += 1
        col += 1

    rank = col

    # Convert sparse to dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for (i, j), val in u_entries.items():
        U[i][j] = val
    for (i, j), val in v_entries.items():
        V[i][j] = val
    for (i, j), val in w_entries.items():
        W[i][j] = val

    return {"U": U, "V": V, "W": W}
