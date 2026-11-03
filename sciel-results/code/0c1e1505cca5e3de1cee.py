def construct():
    n, m, p = 2, 4, 6
    
    # Use rank-30 algorithm for 3x3x3x3 tiled:
    # Split m=4 into 3+1, p=6 into 3+3
    # Block 1: 3x3x3x3 (rows 0-2 of A, cols 0-2 of B) -> rank 30
    # Block 2: 3x3x3x3 (rows 0-2 of A, cols 3-5 of B) -> rank 30
    # Block 3: 1x1x1x1 (row 3 of A, col 3 of B) -> rank 1... 
    # Wait, that's not right. Let me think more carefully.
    
    # Actually, the directive says:
    # "Apply the known rank-30 algorithm for 3x3 x 3x3 matrix multiplication 
    #  to a 3x4 x 4x6 sub-problem (by padding the 2x4 input to 3x4) and combine 
    #  it with a direct schoolbook multiplication for the remaining 1x4 x 4x6 slice,
    #  yielding a total rank of 30 + 4 = 34"
    
    # So: pad 2x4 to 3x4, multiply 3x4 by 4x6 using rank-30 for the 3x3 part
    # and schoolbook for the 1x4 x 4x6 part.
    
    # 3x4 x 4x6: split the 4 middle dimension into 3+1
    # Part 1: 3x3 x 3x6 -> use rank-30 for 3x3x3x3, tiled 2 times for p=6 -> rank 60
    # That's too high.
    
    # Let me re-read: "3x4 x 4x6 sub-problem" - this means A is 3x4, B is 4x6
    # Split m=4 into 3+1:
    #   A1 = 3x3 (first 3 cols of A), A2 = 3x1 (last col of A)
    #   B1 = 3x6 (first 3 rows of B), B2 = 1x6 (last row of B)
    #   C1 = A1 * B1 (3x6), C2 = A2 * B2 (3x6)
    #   C = C1 + C2
    # 
    # For C1: 3x3 x 3x6, split p=6 into 3+3
    #   C1a = A1 * B1a (3x3), C1b = A1 * B1b (3x3)
    #   Use rank-30 for each: total 60
    # For C2: 3x1 x 1x6, schoolbook: rank 3*1*6 = 18
    # Total: 60 + 18 = 78. That's worse.
    
    # I think the directive means something different. Let me re-read.
    # "Apply the known rank-30 algorithm for 3x3 x 3x3 matrix multiplication 
    #  to a 3x4 x 4x6 sub-problem (by padding the 2x4 input to 3x4)"
    
    # Maybe: pad A from 2x4 to 3x4, then use a rank-30 algorithm that handles
    # 3x4 x 4x6 directly? But rank-30 is for 3x3x3x3.
    
    # Or maybe: split the problem as:
    # A is 3x4 (padded), B is 4x6
    # Split m=4 into 3+1:
    #   C = A[:,0:3] * B[0:3,:] + A[:,3:4] * B[3:4,:]
    # First term: 3x3 x 3x6. Split p=6 into 2+2+2? No, use 3x3 blocks.
    # 3x3 x 3x6 = two 3x3x3x3 blocks, each rank 30, total 60.
    # Second term: 3x1 x 1x6, rank 18.
    # Total: 78. Still bad.
    
    # I think the directive might be wrong or I'm misunderstanding it.
    # Let me just implement Strassen tiling for rank 42, which I know works.
    
    import numpy as np
    
    # Strassen 2x2x2x2 factors (verified):
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # (a+d)(e+h) -> i+l
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),   # (c+d)e -> k-l
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),   # a(f-h) -> j+l
        ([0, 0, 0, 1], [0, 1, -1, 0], [0, 1, 0, 0]),   # d(f-g) -> k
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]),    # (a+b)h -> i+j
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # (c-a)(e+f) -> l
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, -1]),  # (b-d)(g+h) -> j-l
    ]
    
    # Verify
    T = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i*2+j, j*2+k, k*2+i] = 1
    
    T_recon = np.zeros((4, 4, 4))
    for U, V, W in strassen_factors:
        T_recon += np.array(U)[:, None, None] * np.array(V)[None, :, None] * np.array(W)[None, None, :]
    
    assert np.allclose(T_recon, T), f"Strassen verification failed"
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Tile: 6 blocks of 2x2x2x2
    # mb in {0,1} for m-split (cols 0-1, 2-3 of A; rows 0-1, 2-3 of B)
    # pb in {0,1,2} for p-split (cols 0-1, 2-3, 4-5 of B; rows 0-1, 2-3, 4-5 of C)
    col = 0
    for mb in range(2):
        for pb in range(3):
            for Uf, Vf, Wf in strassen_factors:
                # U: A rows 0-1, cols mb*2 to mb*2+1
                for ri in range(2):
                    for cj in range(2):
                        U[ri * m + (mb * 2 + cj)][col] = Uf[ri * 2 + cj]
                # V: B rows mb*2 to mb*2+1, cols pb*2 to pb*2+1
                for ri in range(2):
                    for cj in range(2):
                        V[(mb * 2 + ri) * p + (pb * 2 + cj)][col] = Vf[ri * 2 + cj]
                # W: C rows pb*2 to pb*2+1, cols 0-1
                for ri in range(2):
                    for cj in range(2):
                        W[(pb * 2 + ri) * n + cj][col] = Wf[ri * 2 + cj]
                col += 1
    
    return {"U": U, "V": V, "W": W}
