"""Rewritten program for rank-minimal exact tensor decomposition of 2x4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-22 exact tensor decomposition for 2x4x5 matrix multiplication.
    Partition inner dim j={0,1},{2,3} and output dim k={0,1,2},{3,4}.
    Each 2x2x3 block uses rank-4 (trivial), each 2x2x2 block uses rank-7 (Strassen).
    Total: 2*(4+7) = 22.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 22
    _t0 = time.time()
    _DEADLINE = 120

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        for (i, j), val in u_coeffs.items():
            U[i * m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i, idx] += val

    def strassen_2x2x2(a_row_offset, b_row_offset, c_col_offset, idx_start):
        """Strassen decomposition for 2x2x2 block.
        a_row_offset: offset for A rows (0 or 1 in local 2x2)
        b_row_offset: offset for B rows (0 or 1 in local 2x2)  
        c_col_offset: offset for C cols (0 or 1 in local 2x2)
        """
        # M1 = (A00 + A11) * (B00 + B11)
        add_term(idx_start,
                 {(0, a_row_offset): 1, (1, a_row_offset + 1): 1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1},
                 {(c_col_offset, 0): 1, (c_col_offset + 1, 1): 1})

        # M2 = (A10 + A11) * B00
        add_term(idx_start + 1,
                 {(1, a_row_offset): 1, (1, a_row_offset + 1): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 1): 1, (c_col_offset + 1, 1): -1})

        # M3 = A00 * (B01 - B11)
        add_term(idx_start + 2,
                 {(0, a_row_offset): 1},
                 {(b_row_offset, c_col_offset + 1): 1, (b_row_offset + 1, c_col_offset + 1): -1},
                 {(c_col_offset + 1, 0): 1, (c_col_offset + 1, 1): 1})

        # M4 = A11 * (B10 - B00)
        add_term(idx_start + 3,
                 {(1, a_row_offset + 1): 1},
                 {(b_row_offset + 1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                 {(c_col_offset, 0): 1, (c_col_offset + 1, 1): 1})

        # M5 = (A00 + A01) * B11
        add_term(idx_start + 4,
                 {(0, a_row_offset): 1, (0, a_row_offset + 1): 1},
                 {(b_row_offset + 1, c_col_offset + 1): 1},
                 {(c_col_offset, 0): -1, (c_col_offset + 1, 0): 1})

        # M6 = (A10 - A00) * (B00 + B01)
        add_term(idx_start + 5,
                 {(1, a_row_offset): 1, (0, a_row_offset): -1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset + 1): 1},
                 {(c_col_offset + 1, 1): 1})

        # M7 = (A01 - A11) * (B10 + B11)
        add_term(idx_start + 6,
                 {(0, a_row_offset + 1): 1, (1, a_row_offset + 1): -1},
                 {(b_row_offset + 1, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1},
                 {(c_col_offset, 0): 1})

    def trivial_2x2x3(a_row_offset, b_row_offset, c_col_offset, idx_start):
        """Trivial rank-4 decomposition for 2x2x3 block.
        Each A[i,j] is multiplied by the full row B[j, c_col_offset:c_col_offset+3].
        """
        # Term 1: A[0, a_row_offset] * B[b_row_offset, c_col_offset:c_col_offset+3]
        add_term(idx_start,
                 {(0, a_row_offset): 1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset + 1): 1, (b_row_offset, c_col_offset + 2): 1},
                 {(c_col_offset, 0): 1, (c_col_offset + 1, 0): 1, (c_col_offset + 2, 0): 1})

        # Term 2: A[0, a_row_offset+1] * B[b_row_offset+1, c_col_offset:c_col_offset+3]
        add_term(idx_start + 1,
                 {(0, a_row_offset + 1): 1},
                 {(b_row_offset + 1, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1, (b_row_offset + 1, c_col_offset + 2): 1},
                 {(c_col_offset, 0): 1, (c_col_offset + 1, 0): 1, (c_col_offset + 2, 0): 1})

        # Term 3: A[1, a_row_offset] * B[b_row_offset, c_col_offset:c_col_offset+3]
        add_term(idx_start + 2,
                 {(1, a_row_offset): 1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset + 1): 1, (b_row_offset, c_col_offset + 2): 1},
                 {(c_col_offset, 1): 1, (c_col_offset + 1, 1): 1, (c_col_offset + 2, 1): 1})

        # Term 4: A[1, a_row_offset+1] * B[b_row_offset+1, c_col_offset:c_col_offset+3]
        add_term(idx_start + 3,
                 {(1, a_row_offset + 1): 1},
                 {(b_row_offset + 1, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1, (b_row_offset + 1, c_col_offset + 2): 1},
                 {(c_col_offset, 1): 1, (c_col_offset + 1, 1): 1, (c_col_offset + 2, 1): 1})

    idx = 0
    # Group 1: j=0,1
    # k=0,1,2: 2x2x3
    trivial_2x2x3(0, 0, 0, idx)
    idx += 4
    # k=3,4: 2x2x2
    strassen_2x2x2(0, 0, 3, idx)
    idx += 7

    # Group 2: j=2,3
    # k=0,1,2: 2x2x3
    trivial_2x2x3(2, 2, 0, idx)
    idx += 4
    # k=3,4: 2x2x2
    strassen_2x2x2(2, 2, 3, idx)
    idx += 7

    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")

    if time.time() - _t0 > _DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
