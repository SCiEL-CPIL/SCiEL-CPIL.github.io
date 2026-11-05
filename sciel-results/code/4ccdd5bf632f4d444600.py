<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 = 3x3 grid of 2x2 blocks
    # Outer: 3x3 matrix multiply (schoolbook, rank 27)
    # Inner: 2x2 matrix multiply (Strassen, rank 7)
    # Total rank = 27 * 7 = 189
    
    outer_blocks = 3  # 3x3 grid
    inner_size = 2    # each block is 2x2
    dim = n  # 6
    
    # Strassen coefficients for 2x2 multiplication
    # For each of 7 Strassen products:
    # U_coeffs[s]: linear form on 2x2 A block (4 entries: [a00, a01, a10, a11])
    # V_coeffs[s]: linear form on 2x2 B block (4 entries: [b00, b01, b10, b11])
    # W_coeffs[s]: linear form on 2x2 C block (4 entries: [c00, c01, c10, c11])
    
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U_coeffs[s][li][lj] = coefficient of A_block[li][lj] in Strassen product s
    U_coeffs = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[-1, 0], [1, 0]],  # M6: -A00 + A10
        [[0, 1], [0, -1]]   # M7: A01 - A11
    ]
    
    # V_coeffs[s][lj][lk] = coefficient of B_block[lj][lk] in Strassen product s
    V_coeffs = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01 - B11
        [[-1, 0], [1, 0]],  # M4: -B00 + B10
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00 + B01
        [[0, 0], [1, 1]]    # M7: B10 + B11
    ]
    
    # W_coeffs[s][li][lk] = coefficient of C_block[li][lk] from Strassen product s
    W_coeffs = [
        [[1, 0], [0, 1]],   # M1: C00 += 1, C11 += 1
        [[0, 0], [1, -1]],  # M2: C10 += 1, C11 -= 1
        [[0, 1], [0, 1]],   # M3: C01 += 1, C11 += 1
        [[1, 0], [1, 0]],   # M4: C00 += 1, C10 += 1
        [[-1, 1], [0, 0]],  # M5: C00 -= 1, C01 += 1
        [[0, 0], [0, 1]],   # M6: C11 += 1
        [[1, 0], [0, 0]]    # M7: C00 += 1
    ]
    
    total_rank = 27 * 7  # 189
    
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    col = 0
    for i in range(outer_blocks):
        for j in range(outer_blocks):
            for k in range(outer_blocks):
                for s in range(7):
                    # U: A block at (i, j), inner indices (li, lj)
                    for li in range(inner_size):
                        for lj in range(inner_size):
                            coeff = U_coeffs[s][li][lj]
                            if coeff != 0:
                                row_u = (i * inner_size + li) * dim + (j * inner_size + lj)
                                U[row_u][col] = coeff
                    
                    # V: B block at (j, k), inner indices (lj, lk)
                    for lj in range(inner_size):
                        for lk in range(inner_size):
                            coeff = V_coeffs[s][lj][lk]
                            if coeff != 0:
                                row_v = (j * inner_size + lj) * dim + (k * inner_size + lk)
                                V[row_v][col] = coeff
                                
                    # W: C block at (i, k), inner indices (li, lk)
                    # Trace convention: W[k*n + i, r]
                    for li in range(inner_size):
                        for lk in range(inner_size):
                            coeff = W_coeffs[s][li][lk]
                            if coeff != 0:
                                # Global row of C = i * inner_size + li
                                # Global col of C = k * inner_size + lk
                                # W index = col * n + row = (k*inner_size + lk) * 6 + (i*inner_size + li)
                                row_w = (k * inner_size + lk) * dim + (i * inner_size + li)
                                W[row_w][col] = coeff
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [6, 6, 6]
DEADLINE_SECONDS = 120

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    dim = n  # 6
    
    # 6x6 matrix multiplication via 4x4 Strassen (rank 7) + 2x2 Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    #
    # 4x4 Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # 4x4 Strassen coefficients
    # U_coeffs4[s][li][lj] for 4x4 A block
    U4 = [
        [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],   # M1: A00+A11
        [[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,1]],   # M2: A10+A11
        [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],   # M3: A00
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],   # M4: A11
        [[1,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],   # M5: A00+A01
        [[-1,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]],  # M6: -A00+A10
        [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,-1,0]],  # M7: A01-A11
    ]
    
    V4 = [
        [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],   # M1: B00+B11
        [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],   # M2: B00
        [[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,-1,0]],  # M3: B01-B11
        [[-1,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]],  # M4: -B00+B10
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],   # M5: B11
        [[1,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],   # M6: B00+B01
        [[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,1]],   # M7: B10+B11
    ]
    
    W4 = [
        [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],   # M1: C00+=1, C11+=1
        [[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,-1]],  # M2: C10+=1, C11-=1
        [[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,1]],   # M3: C01+=1, C11+=1
        [[1,0,0,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]],   # M4: C00+=1, C10+=1
        [[-1,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]],  # M5: C00-=1, C01+=1
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],   # M6: C11+=1
        [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],   # M7: C00+=1
    ]
    
    # 2x2 Strassen coefficients
    U2 = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    
    V2 = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    
    W2 = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    total_rank = 7 * 7  # 49
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for s in range(7):
                    # U: A block at (i,j) in 4x4 grid, each block is 2x2
                    for li in range(2):
                        for lj in range(2):
                            coeff = U4[s][i][j]
                            if coeff != 0:
                                row_u = (i * 2 + li) * dim + (j * 2 + lj)
                                U[row_u][col] = coeff
                            
                    # V: B block at (j,k) in 4x4 grid, each block is 2x2
                    for lj in range(2):
                        for lk in range(2):
                            coeff = V4[s][j][k]
                            if coeff != 0:
                                row_v = (j * 2 + lj) * dim + (k * 2 + lk)
                                V[row_v][col] = coeff
                                
                    # W: C block at (i,k) in 4x4 grid, each block is 2x2
                    # Trace convention: W[k*n + i, r]
                    for li in range(2):
                        for lk in range(2):
                            coeff = W4[s][i][k]
                            if coeff != 0:
                                row_w = (k * 2 + lk) * dim + (i * 2 + li)
                                W[row_w][col] = coeff
                    col += 1
                    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 5:
        # Safety margin: return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to reconsider. The 4×4 Strassen approach has an issue: the 4×4 Strassen decomposes the 4×4 matrix multiplication, but our matrix is 6×6, not 4×4. I can't directly use 4×4 Strassen on a 6×6 matrix.

Let me reconsider the approach. The current approach uses 3×3 blocks of 2×2, which is correct for 6×6. To improve, I should look at:

1. **Using a better algorithm for the outer 3×3 multiplication** - but the minimal rank for 3×3 is not known to be less than 27 (schoolbook), though there are some results.

2. **Actually, wait** - I can use a 6×6 = 2×3 grid where one dimension uses 2×2 Strassen and the other uses something else. But that's not a standard block decomposition.

3. **Better idea**: Use the fact that 6 = 2+2+2, and apply Strassen at the 2×2 block level, but also try to find a lower-rank decomposition for the 3×3 outer product. The minimal tensor rank of 3×3 matrix multiplication is known to be at most 23 (Bini et al.), but constructing such a decomposition is complex.

4. **Simpler improvement**: The current code is correct with rank 189. Let me verify it works and add the deadline guard. The main improvement opportunity within this family would be to use a known lower-rank 3×3 decomposition.

Actually, let me reconsider the 4×4 approach. I could pad the 6×6 matrix to 8×8, use 4×4 Strassen, and then extract the 6×6 result. But that would give rank 7×7=49 for the 8×8 multiplication, which is worse than 189 for 6×6.

Let me stick with the 3×3 blocks of 2×2 approach but verify correctness and add the deadline guard:

