import numpy as np
import time

def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    This uses 7 Strassen terms for each of the 7 2x2 blocks of the B matrix.
    The rank is minimal (49) as the tensor is a direct sum of 7 copies of the 2x2x2 tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = 2, 2, 14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen terms for 2x2 * 2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Term k: U_k = sum a_i, V_k = sum b_j, W_k = sum c_l
    # 1: (a00+a11)(b00+b11) -> c00 +=, c01 +=, c10 +=, c11 -=
    # 2: (a10+a11)b00       -> c10 +=, c11 -=
    # 3: a00(b11-b10)       -> c01 +=, c11 +=
    # 4: a11(b00-b01)       -> c00 +=, c10 +=
    # 5: (a00+a01)b11       -> c00 -=, c01 +=
    # 6: (a10-a00)(b00+b01) -> c11 +=
    # 7: (a01-a11)(b10+b11) -> c00 +=
    
    # Coefficients: (a_dict, b_dict, c_dict)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            # Fill U
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            # Fill V
            # B is 2 x 14. Block t uses columns 2t and 2t+1.
            # b_idx in {0,1,2,3} corresponds to B[0, 2t], B[0, 2t+1], B[1, 2t], B[1, 2t+1]
            for b_idx, coeff in b_coeffs.items():
                b_row = b_idx // 2
                b_col_offset = b_idx % 2
                b_col = 2 * t + b_col_offset
                V[b_row * p + b_col, col] = coeff
            
            # Fill W
            # W follows trace convention W[k*n+i, r].
            # C is 2 x 14. Block t uses columns 2t and 2t+1.
            # c_idx in {0,1,2,3} corresponds to C[0, 2t], C[0, 2t+1], C[1, 2t], C[1, 2t+1]
            for c_idx, coeff in c_coeffs.items():
                c_row = c_idx // 2
                c_col_offset = c_idx % 2
                c_col = 2 * t + c_col_offset
                # W index: c_col * n + c_row
                W[c_col * n + c_row, col] = coeff

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
