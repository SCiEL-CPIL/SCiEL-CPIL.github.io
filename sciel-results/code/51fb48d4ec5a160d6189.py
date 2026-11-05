"""
Matrix multiplication 2x4x4 with rank-28 decomposition.

This program constructs an exact tensor decomposition for multiplying a 2 by 4
matrix by a 4 by 4 matrix using a hybrid approach:
  - A custom 10-term bilinear identity for the 2x2x4 sub-problem
    (A_L * B[:, :2] contributing to C[:, :2])
  - Two independent Strassen rank-7 blocks for the remaining 2x2x2
    sub-problems (A_R * B[2:4, 0:2] and A_R * B[2:4, 2:4] to C[:, 2:4])
  - One Strassen rank-7 block for A_L * B[0:2, 2:4] to C[:, 2:4]

Total rank = 10 + 7 + 7 + 7 = 31?  No — we share structure.

Actually, let's use a cleaner decomposition:
  Split A (2x4) into A1 (cols 0:2), A2 (cols 2:4).
  Split B (4x4) into B1 (rows 0:2), B2 (rows 2:4).
  C = A1*B1 + A2*B2  (both are 2x4, so we need to be careful with block structure)

Wait: A is 2x4, B is 4x4, C = A*B is 2x4.
  C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
  = A1 * B_top + A2 * B_bottom

Where A1, A2 are 2x2, B_top is 2x4, B_bottom is 2x4.

So C = A1 * B_top + A2 * B_bottom, where A1, A2 are 2x2 and B_top, B_bottom are 2x4.

We can use Strassen for each 2x2 * 2x4 product.
For 2x2 * 2x4: split B_top into B_top[:, 0:2] and B_top[:, 2:4], both 2x2.
  A1 * B_top = A1 * B_top[:, 0:2] + A1 * B_top[:, 2:4]
Each 2x2*2x2 uses rank 7 (Strassen), so 14 for A1 * B_top.
Similarly 14 for A2 * B_bottom.
Total = 28.

Let's implement this cleanly.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2.
    Each term is (u_dict, v_dict, w_dict) with local 2x2 indices.
    Local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    return [
        # P1 = (x0+x3)(y0+y3) -> c0:+1, c3:+1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2 = (x2+x3)*y0 -> c2:+1, c3:-1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3 = x0*(y1-y3) -> c1:+1, c3:+1
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4 = x3*(y2-y0) -> c0:+1, c2:+1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5 = (x0+x1)*y3 -> c0:-1, c1:+1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6 = (x2-x0)*(y0+y1) -> c3:+1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7 = (x1-x3)*(y2+y3) -> c0:+1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]


def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    t0 = time.time()
    deadline = 150.0
    safety_margin = 5.0

    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28

    U = np.zeros((n * m, rank), dtype=float)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (8, 28)

    strassen_terms = _strassen_2x2_terms()

    # Four sub-products:
    # 1) A[:,0:2] * B[0:2, 0:2] -> C[:, 0:2]  (A1 * B11)
    # 2) A[:,2:4] * B[2:4, 0:2] -> C[:, 0:2]  (A2 * B21)
    # 3) A[:,0:2] * B[0:2, 2:4] -> C[:, 2:4]  (A1 * B12)
    # 4) A[:,2:4] * B[2:4, 2:4] -> C[:, 2:4]  (A2 * B22)

    def idx_u(r, c):
        # A is 2x4, row r (0..1), col c (0..3)
        return r * 4 + c

    def idx_v(r, c):
        # B is 4x4, row r (0..3), col c (0..3)
        return r * 4 + c

    def idx_w(r, c):
        # C is 2x4, trace convention W[k*n + i] where k=col, i=row
        # So W[col * 2 + row]
        return c * 2 + r

    # For each sub-product, define the local-to-global maps
    # Local 2x2: row lr in {0,1}, col lc in {0,1}
    # Global row gr, global col gc

    sub_products = [
        # (row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C)
        (0, 0, 0, 0, 0, 0),   # A1*B11 -> C11
        (0, 2, 2, 0, 0, 0),   # A2*B21 -> C11
        (0, 0, 0, 2, 0, 2),   # A1*B12 -> C12
        (0, 2, 2, 2, 0, 2),   # A2*B22 -> C12
    ]

    col = 0
    for (rA_off, cA_off, rB_off, cB_off, rC_off, cC_off) in sub_products:
        if time.time() - t0 > deadline - safety_margin:
            break

        for (u_dict, v_dict, w_dict) in strassen_terms:
            if time.time() - t0 > deadline - safety_margin:
                break

            for (lr, lc), coeff in u_dict.items():
                gr = lr + rA_off
                gc = lc + cA_off
                U[idx_u(gr, gc), col] = coeff

            for (lr, lc), coeff in v_dict.items():
                gr = lr + rB_off
                gc = lc + cB_off
                V[idx_v(gr, gc), col] = coeff

            for (lr, lc), coeff in w_dict.items():
                gr = lr + rC_off
                gc = lc + cC_off
                W[idx_w(gr, gc), col] = coeff

            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
