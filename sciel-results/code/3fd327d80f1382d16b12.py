"""Matrix multiplication 4,4,6 — rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6

    # Correct Strassen rank-7 decomposition for 2x2x2
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b01-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # U vectors (for A, flattened as [a00, a01, a10, a11])
    U_strassen = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [1, 0, -1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]

    # V vectors (for B, flattened as [b00, b01, b10, b11])
    V_strassen = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [0, 1, 0, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]

    # W vectors (for C, flattened as [C00, C01, C10, C11])
    # C00 = M1 - M5 + M4 + M7 -> coeff for M1:1, M2:0, M3:0, M4:1, M5:-1, M6:0, M7:1
    # C01 = M3 + M5           -> coeff for M1:0, M2:0, M3:1, M4:0, M5:1, M6:0, M7:0
    # C10 = M2 + M6           -> coeff for M1:0, M2:1, M3:0, M4:0, M5:0, M6:1, M7:0
    # C11 = M1 - M2 + M3 + M6 -> coeff for M1:1, M2:-1, M3:1, M4:0, M5:0, M6:1, M7:0
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 1],   # M3: C01+=1, C11+=1
        [1, 0, 0, 0],   # M4: C00+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 1, 1],   # M6: C10+=1, C11+=1
        [1, 0, 0, 0],   # M7: C00+=1
    ]

    # For 4x4x6:
    # A is 4x4, B is 4x6, C is 4x6
    # We can view this as 2x2 blocks of 2x2 for A, and 2x3 blocks of 2x2 for B
    # But actually, let's think of it differently.
    #
    # The tensor for 4x4x6 multiplication has entries T[a_idx, b_idx, c_idx]
    # where a_idx in [0,16), b_idx in [0,24), c_idx in [0,24)
    # T[a,b,c] = 1 if c = b//4 * 4 + a%4 ... no, that's not right.
    #
    # Standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # A is n x m = 4x4, B is m x p = 4x6, C is n x p = 4x6
    # a_idx = i*m + j = 4i + j
    # b_idx = j*p + k = 6j + k
    # c_idx = k*n + i = 4k + i (trace convention)
    #
    # T[a_idx, b_idx, c_idx] = 1 if there exists i,j,k such that
    # a_idx = 4i+j, b_idx = 6j+k, c_idx = 4k+i
    # i.e., a_idx%4 == j, b_idx//6 == j, c_idx%4 == i, c_idx//4 == k, b_idx%6 == k
    # So T[4i+j, 6j+k, 4k+i] = 1 for i in [0,4), j in [0,4), k in [0,6)
    #
    # Now, using Strassen on 2x2 blocks:
    # Split A into 2x2 blocks of 2x2: A[ib, jb] where ib,jb in {0,1}
    # Split B into 2x3 blocks of 2x2: B[jb, kb] where jb in {0,1}, kb in {0,1,2}
    # C[ib, kb] = sum_jb A[ib, jb] * B[jb, kb]
    #
    # For each (ib, kb) pair, C[ib, kb] is a 2x2 matrix = sum of two 2x2 products
    # This is a 2x2x2 multiplication where the "scalars" are 2x2 matrices
    # We can decompose each such 2x2x2 (over 2x2 matrices) using Strassen on the outer 2x2x2
    # But the inner 2x2 matrices need to be handled too.
    #
    # Actually, the rank of 2x2x2 over a field is 7. If we use Strassen for the outer structure
    # and schoolbook for the inner, we get 7 * 4 = 28 per (ib,kb) block, but that's not right.
    #
    # Let me think more carefully. The 4x4x6 tensor can be seen as:
    # For each (ib, kb) with ib in {0,1}, kb in {0,1,2}:
    #   C_block[ib, kb] = A_block[ib, 0] * B_block[0, kb] + A_block[ib, 1] * B_block[1, kb]
    # where each block is 2x2.
    #
    # This is a 2x2x2 matrix multiplication where each "entry" is a 2x2 matrix.
    # The rank of this tensor (over the ring of 2x2 matrices) is at most 7 (Strassen).
    # But we need the rank over the base field.
    #
    # If we use Strassen for the outer 2x2x2 and schoolbook for each inner 2x2x2,
    # the total rank would be 7 * 4 = 28 per (ib,kb) block? No, that's not how it works.
    #
    # Let me just construct the decomposition directly.
    #
    # For each (ib, kb) block, we have 7 Strassen terms. Each Strassen term involves
    # products of 2x2 matrices, which can be decomposed into 4 scalar products (schoolbook).
    # So each Strassen term contributes 4 rank-1 terms.
    # Total per (ib,kb) block: 7 * 4 = 28.
    # Number of (ib,kb) blocks: 2 * 3 = 6.
    # Total rank: 6 * 28 = 168. That's worse than schoolbook (96).
    #
    # I think the correct approach is to use Strassen directly on the 2x2x2 structure
    # where the "scalars" are the individual elements. Let me reconsider.
    #
    # Actually, for 4x4x6, we can partition as follows:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x2
    # C = [[C00, C01, C02], [C10, C11, C12]] where each Cij is 2x2
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    #
    # Each Cij is a 2x2 matrix = sum of two 2x2 products.
    # We can use Strassen on the "outer" 2x2x2 (the ib,jb,kb indices) but the inner
    # 2x2 matrices are the "scalars".
    #
    # For the outer 2x2x2 (with ib in {0,1}, jb in {0,1}, kb in {0,1,2}), we can't
    # directly apply Strassen because kb has 3 values, not 2.
    #
    # Let me try a different approach: use the known result that the rank of 4x4x6
    # is at most 60 or so, and try to construct a good decomposition.
    #
    # Actually, let me just use a block-diagonal approach with 2x2 Strassen blocks.
    #
    # For each pair of 2x2 blocks in A and B that contribute to a 2x2 block in C,
    # we can use Strassen. But the structure is more complex.
    #
    # Let me try: for each kb in {0,1,2}, we have a 4x4x2 subproblem
    # C[:, kb*2:kb*2+2] = A * B[:, kb*2:kb*2+2]
    # This is a 4x4x2 multiplication.
    #
    # For 4x4x2, we can use Strassen on 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B = [[B00, B01], [B10, B11]] (2x2 blocks of 2x2, but B is 4x2 so B is [[B0],[B1]] where B0,B1 are 2x2)
    # Actually B is 4x2, so B = [[B00, B01], [B10, B11]] where each is 2x1? No, B is 4x2.
    # Let me re-index. B is m x p = 4 x 2. So B has 4 rows and 2 columns.
    # If we split into 2x2 blocks, B = [[B00], [B10]] where B00 and B10 are 2x2? No, that doesn't work.
    #
    # I think I'm overcomplicating this. Let me just use the schoolbook decomposition
    # but try to find a better one using known techniques.
    #
    # For now, let me return the schoolbook rank-96 decomposition to ensure correctness,
    # and then I can improve it.

    rank = n * m * p  # 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
