"""Rewritten program using tensor-slicing / linear-algebra approach for 2x4 x 4x4."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # A is n x m = 2x4, B is m x p = 4x4, C is p x n = 4x2
    # Output convention: W[k*n + i, r] where k is row of C (0..p-1), i is col of C (0..n-1)
    # So C[k, i] = sum_j A[i, j] * B[j, k]
    # Global index for C: k*n + i = k*2 + i
    
    # We use a block-based approach with 4 blocks of 2x2 multiplications
    # Each block uses Strassen's 7-term algorithm
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen terms for 2x2 matrix multiplication
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # terms[s] = (U_dict, V_dict, W_dict)
    terms = [
        # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = a11(b10-b00) -> c00 += M4, c10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (a10-a00)(b00+b01) -> c11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (a01-a11)(b10+b11) -> c00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block structure:
    # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, :]
    # Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[0:2, :]
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[2:4, :]
    # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[2:4, :]
    
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_dict, v_dict, w_dict = terms[s]
            
            for local_idx, coeff in u_dict.items():
                li = local_idx // 2  # row in local 2x2
                lj = local_idx % 2   # col in local 2x2
                # Global A index: i * m + j
                i = li  # 0 or 1
                if b == 0:
                    j = lj
                elif b == 1:
                    j = 2 + lj
                elif b == 2:
                    j = lj
                elif b == 3:
                    j = 2 + lj
                g_idx = i * m + j
                U[g_idx][col] += coeff
            
            for local_idx, coeff in v_dict.items():
                li = local_idx // 2  # row in local 2x2 (j index)
                lj = local_idx % 2   # col in local 2x2 (k index)
                # Global B index: j * p + k
                if b == 0:
                    j = li
                    k = lj
                elif b == 1:
                    j = 2 + li
                    k = lj
                elif b == 2:
                    j = li
                    k = 2 + lj
                elif b == 3:
                    j = 2 + li
                    k = 2 + lj
                g_idx = j * p + k
                V[g_idx][col] += coeff
            
            for local_idx, coeff in w_dict.items():
                li = local_idx // 2  # row in local 2x2 (i index)
                lj = local_idx % 2   # col in local 2x2 (k index)
                # Global C index: k * n + i
                i = li
                if b == 0:
                    k = lj
                elif b == 1:
                    k = lj
                elif b == 2:
                    k = 2 + lj
                elif b == 3:
                    k = 2 + lj
                g_idx = k * n + i
                W[g_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
