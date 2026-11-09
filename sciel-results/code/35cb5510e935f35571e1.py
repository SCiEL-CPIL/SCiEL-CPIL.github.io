"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen factors for 2x2x2
    # U, V, W are 4x7 matrices
    # U[i,j] corresponds to A[i,j]
    # V[i,j] corresponds to B[i,j]
    # W[i,j] corresponds to C[i,j]
    
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # We decompose 4x4x6 into 6 blocks of 2x2x2?
    # No, we decompose into 12 blocks of 2x2x2?
    # A is 4x4 -> 4 blocks of 2x2.
    # B is 4x6 -> 6 blocks of 2x2.
    # C is 4x6 -> 6 blocks of 2x2.
    
    # C_ab = sum_g A_ag B_gb
    # For fixed a, b:
    # C_ab = A_a0 B_0b + A_a1 B_1b
    # This is M_{2,2,2} + M_{2,2,2}.
    # Rank 7 + 7 = 14.
    # Total rank = 6 * 14 = 84.
    
    total_rank = 84
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 84
    
    # Loop over output blocks (a, b)
    # a in {0, 1} (row block of A/C)
    # b in {0, 1, 2} (col block of B/C)
    
    for a in range(2):
        for b in range(3):
            # This output block C_ab requires 14 rank terms.
            # Term 0-6: A_a0 * B_0b (Strassen)
            # Term 7-13: A_a1 * B_1b (Strassen)
            
            base_offset = (a * 3 + b) * 14
            
            # Part 1: A_a0 * B_0b
            # A block index: a*2 + 0 (col block 0) -> a
            # B block index: 0*3 + b (row block 0, col block b) -> b
            # Wait, B is 4x6. Rows 0-3, Cols 0-5.
            # Block (r, c) of B is B[r*2:r*2+2, c*2:c*2+2].
            # We need B_0b. Row block 0, Col block b.
            
            for c in range(7):
                col = base_offset + c
                
                # U part: A_a0
                # A is 4x4. Block (a, 0).
                # Rows a*2 to a*2+2. Cols 0 to 2.
                for di in range(2):
                    for dj in range(2):
                        i = a * 2 + di
                        j = 0 * 2 + dj
                        U[i * m + j][col] = S_U[di * 2 + dj][c]
                        
                # V part: B_0b
                # B is 4x6. Block (0, b).
                # Rows 0 to 2. Cols b*2 to b*2+2.
                for dj in range(2):
                    for dk in range(2):
                        j = 0 * 2 + dj
                        k = b * 2 + dk
                        V[j * p + k][col] = S_V[dj * 2 + dk][c]
                        
                # W part: C_ab
                # C is 4x6. Block (a, b).
                # Rows a*2 to a*2+2. Cols b*2 to b*2+2.
                # Convention: W[k*n + i, col]
                for dk in range(2):
                    for di in range(2):
                        k = b * 2 + dk
                        i = a * 2 + di
                        W[k * n + i][col] = S_W[dk * 2 + di][c]
                        
            # Part 2: A_a1 * B_1b
            # A block index: a*2 + 1 (col block 1) -> a*2+1
            # B block index: 1*3 + b (row block 1, col block b) -> 3+b
            
            for c in range(7):
                col = base_offset + 7 + c
                
                # U part: A_a1
                for di in range(2):
                    for dj in range(2):
                        i = a * 2 + di
                        j = 1 * 2 + dj
                        U[i * m + j][col] = S_U[di * 2 + dj][c]
                        
                # V part: B_1b
                for dj in range(2):
                    for dk in range(2):
                        j = 1 * 2 + dj
                        k = b * 2 + dk
                        V[j * p + k][col] = S_V[dj * 2 + dk][c]
                        
                # W part: C_ab
                for dk in range(2):
                    for di in range(2):
                        k = b * 2 + dk
                        i = a * 2 + di
                        W[k * n + i][col] = S_W[dk * 2 + di][c]

    if time.time() > deadline:
        # Fallback to schoolbook if something goes wrong or timeout?
        # Schoolbook rank 96.
        # But we are likely fast enough.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
