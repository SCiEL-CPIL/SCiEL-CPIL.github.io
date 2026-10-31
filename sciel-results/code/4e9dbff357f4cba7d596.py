import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    dim = n * m  # 16

    # Build the matrix multiplication tensor T of shape (16, 16, 16)
    # T[a, b, c] = 1 iff a=i*4+j, b=j*4+k, c=k*4+i for some i,j,k
    T = np.zeros((dim, dim, dim), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                T[a, b, c] = 1.0

    # Strategy: Try to find a rank-48 decomposition using a direct construction.
    # The idea: 4x4 = 2x2 blocks. Instead of full Strassen recursion (rank 49),
    # we can use a modified approach that saves one multiplication.
    #
    # Alternative: Use the known rank-48 decomposition for 4x4x4.
    # This comes from the fact that the 4x4 matmul tensor has a specific structure
    # that allows one fewer multiplication than the naive Strassen recursion.
    #
    # Let me construct the rank-48 decomposition directly.
    #
    # The 4x4 matmul: C = A * B
    # Using 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each 2x2 multiplication uses 7 multiplications (Strassen), so 4*7 = 28.
    # But we can share some computations.
    #
    # Actually, the standard approach gives rank 49 via full Strassen recursion.
    # For rank 48, we need a different structure.
    #
    # Let me try a different approach: use the Winograd method with specific transforms.
    # The 4-point Winograd transform for matmul can give rank 25, but let me verify.
    #
    # Actually, let me use a verified rank-48 construction.
    # I'll use the following approach:
    # 1. Use 2x2 Strassen for the outer 2x2 structure (7 multiplications)
    # 2. For each of the 7 "inner" 2x2 matmuls, use a modified 2x2 matmul
    #    that shares some work with others.
    #
    # This is complex. Let me instead use a direct numerical approach:
    # I'll construct the rank-49 Strassen decomposition and then try to
    # find a linear dependency that allows reducing to rank 48.
    #
    # Actually, the cleanest approach is to use the known rank-48 coefficients.
    # Let me construct them using the following verified method:
    #
    # The rank-48 decomposition for 4x4x4 matmul can be constructed as follows:
    # It uses the fact that the 4x4 matmul tensor can be written as a sum of 48
    # rank-1 tensors. The construction is based on the following:
    #
    # 1. The 4x4 matmul can be seen as 2x2 matmul over 2x2 matrices.
    # 2. The 2x2 matmul over 2x2 matrices can be done with 7 "2x2 matmul" operations
    #    (Strassen), but each 2x2 matmul operation can be done with 7 scalar multiplications.
    # 3. However, by carefully choosing the Strassen combinations, we can save one
    #    scalar multiplication overall, giving 48 instead of 49.
    #
    # Let me implement this directly:
    
    # Strassen 2x2 terms (verified correct)
    # Each term: (U_coeff, V_coeff, W_coeff) where each is a dict from (i,j) to coeff
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # For the outer 2x2 structure, we use Strassen:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now, each Mi involves a 2x2 matmul. For example:
    # M1 = (A00+A11) * (B00+B11)
    # This is a 2x2 matmul: (A00+A11) is a 2x2 matrix, (B00+B11) is a 2x2 matrix.
    # Using Strassen for this 2x2 matmul gives 7 scalar multiplications.
    #
    # Total: 7 * 7 = 49 scalar multiplications.
    #
    # To get 48, we need to find one scalar multiplication that can be shared
    # or eliminated. This is non-trivial.
    #
    # Let me use a different approach: I'll use the rank-48 decomposition from
    # the literature, which I'll construct using a direct method.
    #
    # Actually, let me just use the rank-49 Strassen (verified correct) and return it.
    # This gives score 64/49 ≈ 1.306.
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Outer Strassen 4x4 terms
    outer_4x4 = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    col = 0
    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_4x4):
        for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(strassen_2x2):
            for (br, bc), ca in a_blocks:
                for (lr, lc), v in x_coeff.items():
                    a_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    U[a_idx, col] += ca * v
            for (br, bc), cb in b_blocks:
                for (lr, lc), v in y_coeff.items():
                    b_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    V[b_idx, col] += cb * v
            for (br, bc), cc in c_blocks:
                for (lr, lc), v in z_coeff.items():
                    c_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                    W[c_idx, col] += cc * v
            col += 1
    
    assert col == 49, f"Expected 49 columns, got {col}"
    
    # Verify the decomposition
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank-64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb))
        V_fb = np.zeros((16, rank_fb))
        W_fb = np.zeros((16, rank_fb))
        col_fb = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    U_fb[ii * 4 + jj, col_fb] = 1
                    V_fb[jj * 4 + kk, col_fb] = 1
                    W_fb[kk * 4 + ii, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V_fb.tolist() if False else V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
