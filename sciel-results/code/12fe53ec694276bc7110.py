import numpy as np
import time

DIMENSIONS = [2, 2, 7]


def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = DIMENSIONS
    # n=2 rows of A, m=2 cols of A / rows of B, p=7 cols of B
    # A: n x m = 2x2
    # B: m x p = 2x7
    # C: n x p = 2x7
    # U: (n*m) x rank = 4 x rank
    # V: (m*p) x rank = 14 x rank
    # W: (p*n) x rank = 14 x rank, indexed W[k*n + i, r] for C[i, k]

    # Strategy: Use Strassen for the first 6 columns (3 pairs) and schoolbook for column 6.
    # Each pair of columns (k, k+1) uses 7 rank-1 terms.
    # Column 6 uses 4 rank-1 terms.
    # Total rank = 3 * 7 + 4 = 25.

    rank = 25
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Helper to set W index for C[i, k]
    def w_idx(i, k):
        return k * n + i

    # Strassen for a 2x2 * 2x2 block
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    # The 7 products:
    # M1 = (A00 + A11) * (B00 + B10)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B11 - B10)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B11 + B10)
    
    # C00 = M1 + M3 - M5 - M7
    # C01 = M4 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M7

    # Indices in U for A elements:
    # A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3

    def add_strassen_block(col_offset, k1, k2):
        # k1, k2 are the column indices of B for this block
        # B00 is at row 0*p + k1 = k1 in V
        # B01 is at row 0*p + k2 = k2 in V
        # B10 is at row 1*p + k1 = p + k1 in V
        # B11 is at row 1*p + k2 = p + k2 in V

        # M1
        r = col_offset
        U[0, r] += 1; U[3, r] += 1
        V[k1, r] += 1; V[p + k1, r] += 1
        W[w_idx(0, k1), r] += 1
        W[w_idx(1, k1), r] += 1
        W[w_idx(0, k2), r] -= 1
        W[w_idx(1, k2), r] -= 1

        # M2
        r = col_offset + 1
        U[2, r] += 1; U[3, r] += 1
        V[k1, r] += 1
        W[w_idx(1, k1), r] += 1
        W[w_idx(1, k2), r] -= 1

        # M3
        r = col_offset + 2
        U[0, r] += 1
        V[p + k2, r] += 1; V[p + k1, r] -= 1
        W[w_idx(0, k1), r] += 1
        W[w_idx(0, k2), r] -= 1

        # M4
        r = col_offset + 3
        U[3, r] += 1
        V[k2, r] += 1; V[k1, r] -= 1
        W[w_idx(0, k1), r] += 1
        W[w_idx(1, k2), r] += 1

        # M5
        r = col_offset + 4
        U[0, r] += 1; U[1, r] += 1
        V[p + k2, r] += 1
        W[w_idx(0, k1), r] -= 1
        W[w_idx(1, k2), r] += 1

        # M6
        r = col_offset + 5
        U[2, r] += 1; U[0, r] -= 1
        V[k1, r] += 1; V[k2, r] += 1
        W[w_idx(1, k1), r] += 1
        W[w_idx(0, k2), r] += 1

        # M7
        r = col_offset + 6
        U[1, r] += 1; U[3, r] -= 1
        V[p + k2, r] += 1; V[p + k1, r] += 1
        W[w_idx(0, k1), r] -= 1
        W[w_idx(0, k2), r] -= 1

    # Block 1: k1=0, k2=1, col_offset=0
    if time.time() < deadline:
        add_strassen_block(0, 0, 1)

    # Block 2: k1=2, k2=3, col_offset=7
    if time.time() < deadline:
        add_strassen_block(7, 2, 3)

    # Block 3: k1=4, k2=5, col_offset=14
    if time.time() < deadline:
        add_strassen_block(14, 4, 5)

    # Block 4: Schoolbook for k=6, col_offset=21
    col = 21
    k = 6

    # C0,6 = A00*B0,6 + A01*B1,6
    # C1,6 = A10*B0,6 + A11*B1,6

    # Term 1: A00 * B0,6 -> C0,6
    r = col
    U[0, r] += 1
    V[k, r] += 1
    W[w_idx(0, k), r] += 1

    # Term 2: A01 * B1,6 -> C0,6
    r = col + 1
    U[1, r] += 1
    V[p + k, r] += 1
    W[w_idx(0, k), r] += 1

    # Term 3: A10 * B0,6 -> C1,6
    r = col + 2
    U[2, r] += 1
    V[k, r] += 1
    W[w_idx(1, k), r] += 1

    # Term 4: A11 * B1,6 -> C1,6
    r = col + 3
    U[3, r] += 1
    V[p + k, r] += 1
    W[w_idx(1, k), r] += 1

    # Convert to lists of lists for the expected format?
    # The seed program returned lists. Let's check if the evaluator accepts arrays.
    # The task says "factor matrices of shapes... with integer/half-integer entries".
    # The seed returned `return {"U": U, "V": W}` where U,V,W were lists.
    # I will return lists to be safe, or arrays if numpy is preferred.
    # The prompt says "Keep the function names and signatures... same as seed".
    # Seed: `U = [[0] * rank for _ in range(n * m)]` etc.
    # So I should return lists.

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
