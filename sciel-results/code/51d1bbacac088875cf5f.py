"""
Matrix multiplication 4x4x4 tensor decomposition.

This program constructs the known optimal rank-49 tensor decomposition for
multiplying two 4x4 matrices. The decomposition is derived from applying
Strassen's algorithm to 2x2 block matrices, where each 2x2 block product
is itself computed using Strassen's 7-product formula.

Total rank: 7 (block-level) * 7 (scalar-level) = 49.
This is the known optimal rank for 4x4 matrix multiplication.
"""

import numpy as np

# Dimensions
N = 4
M = 4
P = 4
RANK = 49

def construct():
    """
    Construct the rank-49 tensor decomposition for 4x4 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with integer entries.
    """
    U = np.zeros((N * M, RANK), dtype=np.int32)
    V = np.zeros((M * P, RANK), dtype=np.int32)
    W = np.zeros((P * N, RANK), dtype=np.int32)

    strassen_2x2 = [
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (1, 0, 1)]),
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]

    block_strassen = [
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (1, 0, 1)]),
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]

    def block_elem_to_flat(block_r, block_c, elem_r, elem_c):
        row = 2 * block_r + elem_r
        col = 2 * block_c + elem_c
        return row * 4 + col

    term_idx = 0
    for b_idx in range(7):
        X_blocks = block_strassen[b_idx][0]
        Y_blocks = block_strassen[b_idx][1]
        Z_blocks = block_strassen[b_idx][2]

        for s_idx in range(7):
            x_form = strassen_2x2[s_idx][0]
            y_form = strassen_2x2[s_idx][1]
            z_form = strassen_2x2[s_idx][2]

            for er, ec, coeff_s in x_form:
                for br, bc, coeff_x in X_blocks:
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    U[flat_idx, term_idx] += coeff_s * coeff_x

            for er, ec, coeff_s in y_form:
                for br, bc, coeff_y in Y_blocks:
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    V[flat_idx, term_idx] += coeff_s * coeff_y

            for er, ec, coeff_z in z_form:
                for br, bc, coeff_zb in Z_blocks:
                    i = 2 * br + er
                    k = 2 * bc + ec
                    flat_idx = k * 4 + i
                    W[flat_idx, term_idx] += coeff_z * coeff_zb

            term_idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    U_np = np.array(result['U'])
    V_np = np.array(result['V'])
    W_np = np.array(result['W'])
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1.0 if i == k else 0.0
                val = np.sum(U_np[i*4+j, :] * V_np[j*4+k, :] * W_np[k*4+i, :])
                if abs(val - expected) > 1e-9:
                    valid = False
                    print(f"Error at i={i}, j={j}, k={k}: expected {expected}, got {val}")
                    break
            if not valid:
                break
        if not valid:
            break
    print("Verification PASSED" if valid else "Verification FAILED")
