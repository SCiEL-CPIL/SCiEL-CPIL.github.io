"""Rewritten program: rank-28 Strassen decomposition for 2x4 by 4x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4

    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # shape (8, 28)

    # Strassen's 7 terms for 2x2 matrix multiplication
    # Each term is a list of (index, coefficient) for U, V, W factors
    # Indices 0..3 correspond to the 4 entries of a 2x2 matrix:
    #   0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2: (A10+A11)B00
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3: A00(B10-B11)
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4: A11(B01-B00)
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5: (A00+A01)B11
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6: (A10-A00)(B00+B01)
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7: (A01-A11)(B10+B11)
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]

    # We multiply A (2x4) by B (4x4) to get C (2x4).
    # Split the middle dimension m=4 into two halves of size 2:
    #   A = [A_left | A_right] where A_left is 2x2 (cols 0,1), A_right is 2x2 (cols 2,3)
    #   B = [B_top; B_bottom] where B_top is 2x4 (rows 0,1), B_bottom is 2x4 (rows 2,3)
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # Further split B_top and B_bottom horizontally into 2x2 blocks:
    #   B_top = [B_tl | B_tr]  (cols 0,1 and cols 2,3)
    #   B_bottom = [B_bl | B_br]
    #
    # C[:, 0:2] = A_left * B_tl + A_right * B_bl
    # C[:, 2:4] = A_left * B_tr + A_right * B_br
    #
    # This gives 4 independent 2x2 * 2x2 multiplications, each with rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.

    # For each block, we need:
    #   - j_offset: which pair of columns of A (and rows of B) to use
    #   - k_offset: which pair of columns of B (and rows of C) to use
    blocks = [
        {'j_offset': 0, 'k_offset': 0},  # A_left * B_tl -> C[:, 0:2]
        {'j_offset': 2, 'k_offset': 0},  # A_right * B_bl -> C[:, 0:2]
        {'j_offset': 0, 'k_offset': 2},  # A_left * B_tr -> C[:, 2:4]
        {'j_offset': 2, 'k_offset': 2},  # A_right * B_br -> C[:, 2:4]
    ]

    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            # U factor: entries of A (2x2 block at rows 0..1, cols j_off..j_off+1)
            for (idx, coeff) in term['U']:
                i = idx // 2  # row within 2x2 block (0 or 1)
                j = idx % 2   # col within 2x2 block (0 or 1)
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff

            # V factor: entries of B (2x2 block at rows j_off..j_off+1, cols k_off..k_off+1)
            for (idx, coeff) in term['V']:
                j = idx // 2   # row within 2x2 block (0 or 1)
                k = idx % 2    # col within 2x2 block (0 or 1)
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff

            # W factor: entries of C (2x2 block at rows k_off..k_off+1, cols 0..1)
            # W index convention: W[k*n + i, r]
            for (idx, coeff) in term['W']:
                k = idx // 2   # row within 2x2 block (0 or 1)
                i = idx % 2    # col within 2x2 block (0 or 1)
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff

            col += 1

    # Trim to exactly 28 columns (should already be 28)
    assert col == 28, f"Expected 28 columns, got {col}"
    U = U[:col]
    V = V[:col]
    W = W[:col]

    # Cast to integers for cleanliness
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]

    return {"U": U, "V": V, "W": W}
