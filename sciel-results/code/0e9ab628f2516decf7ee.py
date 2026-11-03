import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    total_rank = 49 + 48  # Strassen(4x4x4) + schoolbook(4x4x3)

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # 2x2x2 Strassen decomposition: 7 terms, each (U_dict, V_dict, W_dict)
    # U indices: (i,j) for A[i,j]; V indices: (j,k) for B[j,k]; W indices: (k,i) for C[k,i]
    strassen_2x2 = [
        # M1 = (A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3 = A11(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # M4 = A22(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]

    # Outer Strassen terms for 4x4x4:
    # Each term: (A_blocks, B_blocks, C_blocks)
    # where each block list is [(block_idx, coeff), ...]
    # Block indices: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # For A: block (br, bc) maps to local (i,j) -> global row = br*2+i, col = bc*2+j
    # For B: block (br, bc) maps to local (j,k) -> global row = br*2+j, col = bc*2+k
    # For C: block (br, bc) maps to local (k,i) -> global row = br*2+k, col = bc*2+i

    outer_strassen = [
        # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ([((0, 1), (3, 1))], [((0, 1), (3, 1))], [((0, 1), (3, 1))]),
        # M2 = (A21+A22)B11 -> C21 += 1
        ([((2, 1), (3, 1))], [((0, 1))], [((2, 1))]),
        # M3 = A11(B12-B22) -> C12 += 1
        ([((0, 1))], [((1, 1), (3, -1))], [((1, 1))]),
        # M4 = A22(B21-B11) -> C11 += 1, C21 += 1
        ([((3, 1))], [((2, 1), (0, -1))], [((0, 1), (2, 1))]),
        # M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
        ([((0, 1), (1, 1))], [((3, 1))], [((0, -1), (1, 1))]),
        # M6 = (A21-A11)(B11+B12) -> C22 += 1
        ([((2, 1), (0, -1))], [((0, 1), (1, 1))], [((3, 1))]),
        # M7 = (A12-A22)(B21+B22) -> C11 += 1, C22 += 1
        ([((1, 1), (3, -1))], [((2, 1), (3, 1))], [((0, 1), (3, 1))]),
    ]

    col = 0
    for outer_A, outer_B, outer_C in outer_strassen:
        for U_inner, V_inner, W_inner in strassen_2x2:
            U_vec = np.zeros(n * m, dtype=int)
            V_vec = np.zeros(m * p, dtype=int)
            W_vec = np.zeros(p * n, dtype=int)

            # U: A is 4x4, index = i*4 + j
            for (block, coeff_a) in outer_A:
                br, bc = block  # block row, block col
                for (i, j), coeff_u in U_inner.items():
                    r = br * 2 + i
                    c = bc * 2 + j
                    U_vec[r * 4 + c] += coeff_a * coeff_u

            # V: B is 4x7, index = j*7 + k (only use first 4 columns)
            for (block, coeff_b) in outer_B:
                br, bc = block
                for (j, k), coeff_v in V_inner.items():
                    r = br * 2 + j
                    c = bc * 2 + k
                    V_vec[r * 7 + c] += coeff_b * coeff_v

            # W: C is 7x4 in trace convention, index = k*4 + i
            # C[i,k] standard, but W[k*n+i]
            for (block, coeff_c) in outer_C:
                br, bc = block
                for (k, i), coeff_w in W_inner.items():
                    r = br * 2 + k
                    c = bc * 2 + i
                    W_vec[r * 4 + c] += coeff_c * coeff_w

            U[:, col] = U_vec
            V[:, col] = V_vec
            W[:, col] = W_vec
            col += 1

    # Part 2: 4x4x3 schoolbook (columns 4,5,6 of B; rows 0,1,2 of C)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j, col] = 1
                V[j * 7 + (k + 4), col] = 1
                W[k * 4 + i, col] = 1
                col += 1

    # Verify correctness
    U_np = U.astype(float)
    V_np = V.astype(float)
    W_np = W.astype(float)

    # Build multiplication tensor check
    # For each column r of the decomposition:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Should equal delta (1 if valid index)
    # We check by computing C = A*B for random A,B and comparing with tensor contraction

    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 7)
    C_expected = A_test @ B_test

    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*4+i, r]

    C_computed = np.zeros((4, 7))
    for i in range(4):
        for k in range(7):
            s = 0.0
            for j in range(4):
                t_val = 0.0
                for r in range(total_rank):
                    t_val += U[i * 4 + j, r] * V[j * 7 + k, r] * W[k * 4 + i, r]
                s += A_test[i, j] * B_test[j, k] * t_val
            C_computed[i, k] = s

    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook (rank 112)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=int)
        V_fb = np.zeros((m * p, rank_fb), dtype=int)
        W_fb = np.zeros((p * n, rank_fb), dtype=int)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
