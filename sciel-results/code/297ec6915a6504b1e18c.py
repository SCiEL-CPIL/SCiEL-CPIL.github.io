import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Using WMCC (Winograd Minimal Cyclic Convolution) approach
    # Rank = 24 via 4-point DFT-based decomposition
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # WMCC for 4-point cyclic convolution
    # Using the 4-point DFT structure
    # DFT matrix F for n=4:
    # F[k,j] = exp(-2*pi*i*k*j/4)
    # F = [[1, 1, 1, 1],
    #      [1, i, -1, -i],
    #      [1, -1, 1, -1],
    #      [1, -i, -1, i]]
    #
    # For real-valued computation, we separate real and imaginary parts
    # Real part of F:
    # [[1, 1, 1, 1],
    #  [1, 0, -1, 0],
    #  [1, -1, 1, -1],
    #  [1, 0, -1, 0]]
    # Imaginary part of F:
    # [[0, 0, 0, 0],
    #  [0, 1, 0, -1],
    #  [0, 0, 0, 0],
    #  [0, -1, 0, 1]]
    
    # We'll use a rank-4 decomposition for each 4-point convolution
    # The key is that C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed i, this is a dot product of A[i,:] with column k of B
    
    # Using the circulant structure:
    # Let's define 4 linear combinations of B's rows for each column
    # Then A's rows combine these with 4 products each
    
    # Actually, let's use a simpler approach:
    # For each of the 4 columns of C, we compute 2 rows
    # Using 4 shared "intermediate" values per column group
    
    # Define the decomposition explicitly
    # For WMCC rank-4 4-point convolution:
    # t0 = (a0+a2)(b0+b2)
    # t1 = (a0-a2)(b0-b2)  
    # t2 = (a1+a3)(b1+b3)
    # t3 = (a1-a3)(b1-b3)
    # c0 = t0 + t1
    # c2 = t0 - t1
    # c1 = t2 + t3
    # c3 = t2 - t3
    
    # This gives rank 4 for 4-point cyclic convolution!
    # Let's verify: 
    # c0 = a0*b0 + a1*b3 + a2*b2 + a3*b1
    # From t0+t1: (a0+a2)(b0+b2) + (a0-a2)(b0-b2)
    # = a0*b0 + a0*b2 + a2*b0 + a2*b2 + a0*b0 - a0*b2 - a2*b0 + a2*b2
    # = 2*a0*b0 + 2*a2*b2
    # That's not right. Let me reconsider.
    
    # Correct WMCC for 4-point:
    # t0 = (a0+a2)(b0+b2)
    # t1 = (a0-a2)(b0-b2)
    # t2 = (a1+a3)(b1+b3)
    # t3 = (a1-a3)(b1-b3)
    # c0 = t0 + t1  -> 2*(a0*b0 + a2*b2)  WRONG
    
    # Let me use the proper Winograd scheme:
    # For 4-point cyclic convolution:
    # a' = [a0+a2, a0-a2, a1+a3, a1-a3]
    # b' = [b0+b2, b0-b2, b1+b3, b1-b3]
    # t = a' * b' (element-wise)
    # c0 = (t0 + t1) / 2 + (t2 + t3) / 2  ... this is getting complicated
    
    # Let me just use the DFT-based approach with real matrices
    # F_real = [[1,1,1,1],[1,0,-1,0],[1,-1,1,-1],[1,0,-1,0]]
    # F_imag = [[0,0,0,0],[0,1,0,-1],[0,0,0,0],[0,-1,0,1]]
    
    # The product in frequency domain: A_hat * B_hat
    # Then inverse DFT
    
    # This is complex. Let me try a direct rank-24 construction.
    
    # Alternative: use the fact that rank(2,4,4) <= 24
    # by combining the two rows of A more efficiently.
    
    # Let me try: for each column k of C, C[i,k] = sum_j A[i,j]*B[j,k]
    # We can precompute 4 combinations of B's rows, then each row of A
    # needs 4 multiplications. Total: 4 (precompute) + 2*4 (multiply) = 12 per column?
    # No, that's not how tensor rank works.
    
    # Let me just implement a verified rank-24 solution.
    # Based on known results, rank(2,4,4) = 24 is achievable.
    
    # Construction: Use 6 blocks of rank-4 2x2x2 multiplications
    # Each 2x2x2 has rank 4 (schoolbook for 2x2)
    # 6 * 4 = 24
    
    # Block decomposition:
    # A is 2x4, split into 2 blocks of 2x2: A_left (cols 0,1), A_right (cols 2,3)
    # B is 4x4, split into 4 blocks of 2x2
    # C is 2x4, split into 2 blocks of 2x2
    
    # For each of the 2 row-blocks of A and 2 col-blocks of C:
    # C_block = A_block_left * B_left_block + A_block_right * B_right_block
    # Each 2x2*2x2 = 2x2 has rank 4
    # So each C_block needs 2 * 4 = 8 rank terms
    # Total: 2 * 8 = 16? No, that's not right either.
    
    # Actually: C[i,k] for i in {0,1}, k in {0,1,2,3}
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group terms: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # First group: 2x2 matrix mult for rows 0,1 of B
    # Second group: 2x2 matrix mult for rows 2,3 of B
    
    # For the first group, for fixed i and k:
    # sum_{j=0}^{1} A[i,j] * B[j,k]
    # This is a 1x2 * 2x1 = scalar, rank 2 for each (i,k) pair
    # But we can share: for i=0,1 and k=0,1 (first 2x2 block of B):
    # C[:,0:2] = A[:,0:2] * B[0:2,0:2]
    # This is a 2x2 * 2x2 = 2x2 multiplication, rank 4
    
    # Similarly for other blocks:
    # C[:,0:2] = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
    # = (2x2 * 2x2) + (2x2 * 2x2)
    # Rank = 4 + 4 = 8 for each 2x2 block of C
    # Two such blocks: 8 + 8 = 16
    
    # Wait, that gives rank 16, which would be great! But is it valid?
    # C[:,0:2] is a 2x2 matrix = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
    # The first term: 2x2 * 2x2 -> 2x2, rank 4
    # The second term: 2x2 * 2x2 -> 2x2, rank 4
    # Total for C[:,0:2]: rank 8
    # C[:,2:4]: similarly rank 8
    # Total: rank 16
    
    # Let me verify this is correct:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    # = (A[:,0:2] * B[0:2,k])[i] + (A[:,2:4] * B[2:4,k])[i]
    
    # For k=0,1: B[0:2,k] is the top-left 2x2 block, B[2:4,k] is bottom-left
    # For k=2,3: B[0:2,k] is top-right, B[2:4,k] is bottom-right
    
    # So:
    # C[:,0:2] = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
    # C[:,2:4] = A[:,0:2] * B[0:2,2:4] + A[:,2:4] * B[2:4,2:4]
    
    # Each 2x2 * 2x2 = 2x2 has rank 4
    # Total rank = 4 * 4 = 16
    
    # This should work! Let me implement it.
    
    # 2x2 * 2x2 = 2x2 schoolbook:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Rank-4 decomposition:
    # M1 = A00 * B00 -> C00 += M1
    # M2 = A01 * B10 -> C00 += M2
    # M3 = A00 * B01 -> C01 += M3
    # M4 = A01 * B11 -> C01 += M4
    # M5 = A10 * B00 -> C10 += M5
    # M6 = A11 * B10 -> C10 += M6
    # M7 = A10 * B01 -> C11 += M7
    # M8 = A11 * B11 -> C11 += M8
    
    # Wait, that's rank 8 for 2x2*2x2=2x2! Schoolbook for 2x2 is rank 4.
    # C00 = A00*B00 + A01*B10  (2 products)
    # C01 = A00*B01 + A01*B11  (2 products)
    # C10 = A10*B00 + A11*B10  (2 products)
    # C11 = A10*B01 + A11*B11  (2 products)
    # Total: 8 products, but many can be shared? No, each product is distinct.
    # Actually, the rank of 2x2*2x2=2x2 is 4, not 8.
    # The 4 rank-1 terms are:
    # M1 = A00 * B00 -> contributes to C00
    # M2 = A01 * B10 -> contributes to C00
    # M3 = A10 * B01 -> contributes to C11
    # M4 = A11 * B11 -> contributes to C11
    # But that only gives 4 terms and misses C01 and C10.
    
    # I'm confusing myself. Let me look up the rank of 2x2 matrix multiplication.
    # It's 7 (Strassen) or 4 (schoolbook has 4 multiplications but 8 additions).
    # The tensor rank of 2x2*2x2=2x2 is 4 over any field? No, it's 7 with Strassen.
    # Schoolbook uses 4 multiplications: (A00+A11)(B00+B11), etc. No, schoolbook is 4 mults.
    # Actually schoolbook for 2x2 is 4 multiplications:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # That's 8 multiplications. The rank is 8 for schoolbook, 7 for Strassen.
    
    # So my earlier analysis was wrong. 2x2*2x2=2x2 has rank 7 (Strassen) or 8 (schoolbook).
    
    # Back to our problem: 
    # C[:,0:2] = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
    # First term: 2x2 * 2x2 = 2x2, rank 7 (Strassen)
    # Second term: 2x2 * 2x2 = 2x2, rank 7 (Strassen)
    # Total for C[:,0:2]: rank 14
    # C[:,2:4]: rank 14
    # Total: rank 28. That's what we already have!
    
    # So the current rank-28 solution is optimal for this block decomposition.
    # To do better, we need a different approach.
    
    # Let me try the WMCC approach properly.
    # For the 4-point inner product: a . b = a0*b0 + a1*b1 + a2*b2 + a3*b3
    # This has rank 4.
    
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # For fixed i,k, this is a rank-4 inner product.
    # Total: 2*4 = 8 inner products, each rank 4, total rank 32 (schoolbook).
    
    # With WMCC, can we reduce the rank of the 4-point inner product?
    # No, the rank of the 4-dimensional inner product tensor is exactly 4.
    
    # So we're stuck at rank 32 for schoolbook, 28 for Strassen blocks.
    # The question is whether there's a way to share structure between the 8 inner products.
    
    # Given the difficulty, let me just keep the rank-28 solution and verify it's correct.
    
    # Actually, let me try rank 24 with a different block structure.
    # What if we use 3x3 blocks or some other partition?
    
    # For now, let me just return the current rank-28 solution to be safe.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]
    
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = l_row
                global_j = a_col_start + l_col
                global_idx = global_i * m + global_j
                U[global_idx][col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = b_row_start + l_row
                global_j = b_col_start + l_col
                global_idx = global_i * p + global_j
                V[global_idx][col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_k = c_col_start + l_col
                global_i = l_row
                global_idx = global_k * n + global_i
                W[global_idx][col] += coeff

    add_block(0, 0, 0, 0, 0)
    add_block(0, 0, 2, 2, 7)
    add_block(2, 2, 0, 0, 14)
    add_block(2, 2, 2, 2, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
