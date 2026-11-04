"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Strassen-style decomposition for 3x3 @ 3x4
    # Split A (3x3) into 2x2 block A11 and row/col A12, A21, A22
    # Split B (3x4) into 2x4 block B11 and col B12, B21, B22
    # 
    # A = [[A11, A12],     B = [[B11, B12],
    #      [A21, A22]]          [B21, B22]]
    # where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    #       B11 is 2x4, B12 is 2x1, B21 is 1x4, B22 is 1x1
    #
    # C = A @ B (3x4)
    # C = [[C11, C12],
    #      [C21, C22]]
    # where C11 is 2x2, C12 is 2x1, C21 is 1x2, C22 is 1x1
    #
    # Using Strassen for the 2x2 blocks:
    # M1 = (A11 + A22) @ (B11 + B22)
    # M2 = (A21 + A22) @ B11
    # M3 = A11 @ (B12 - B22)
    # M4 = A22 @ (B21 - B11)
    # M5 = (A11 + A12) @ B22
    # M6 = (A21 - A11) @ (B11 + B12)
    # M7 = (A12 - A22) @ (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # But we need to handle the rectangular blocks carefully.
    # Let's use a simpler approach: direct Strassen on 2x2 blocks
    # treating the 1x1 and 1x2/2x1 parts as separate.
    
    # Actually, let me think more carefully.
    # A is 3x3, B is 3x4, C is 3x4.
    # 
    # Partition A as:
    # A = [A_top, A_bottom] where A_top is 2x3, A_bottom is 1x3
    # Partition B as:
    # B = [B_left, B_right] where B_left is 3x2, B_right is 3x2
    # 
    # Then C = A @ B
    # C_top = A_top @ B (2x4)
    # C_bottom = A_bottom @ B (1x4)
    #
    # For C_top = A_top @ B:
    # A_top is 2x3, B is 3x4
    # Split A_top = [A11, A12] where A11 is 2x2, A12 is 2x1
    # Split B = [B1, B2] where B1 is 3x2, B2 is 3x2
    # C_top = A11 @ B1 + A12 @ B2 (but dimensions don't match directly)
    
    # Let me use a different partitioning.
    # A (3x3): rows 0,1 form top (2x3), row 2 is bottom (1x3)
    # B (3x4): cols 0,1 form left (3x2), cols 2,3 form right (3x2)
    # C (3x4): rows 0,1 form top (2x4), row 2 is bottom (1x4)
    #           cols 0,1 form left (3x2), cols 2,3 form right (3x2)
    #
    # C_top_left (2x2) = A_top(2x3) @ B_left(3x2)
    # C_top_right (2x2) = A_top(2x3) @ B_right(3x2)
    # C_bottom_left (1x2) = A_bottom(1x3) @ B_left(3x2)
    # C_bottom_right (1x2) = A_bottom(1x3) @ B_right(3x2)
    #
    # For C_top_left (2x2 from 2x3 @ 3x2):
    # A_top = [A11(2x2), A12(2x1)]
    # B_left = [B11(2x2); B21(1x2)]
    # C_top_left = A11@B11 + A12@B21
    # Rank: 4 + 2 = 6 (schoolbook would be 4)
    # Wait, that's worse. Let me use Strassen on the 2x2 part.
    
    # Better approach: Use Strassen on the 2x2 subproblem.
    # For 2x2 @ 2x2 -> 2x2, Strassen uses 7 multiplications.
    # 
    # Let me define the full decomposition using 7 Strassen terms for the 2x2 core
    # plus additional terms for the 1-dimensional extensions.
    
    # A (3x3):
    #   A11 (2x2): rows 0-1, cols 0-1
    #   A12 (2x1): rows 0-1, col 2
    #   A21 (1x2): row 2, cols 0-1
    #   A22 (1x1): row 2, col 2
    #
    # B (3x4):
    #   B11 (2x4): rows 0-1, cols 0-3
    #   B12 (2x1): rows 0-1, col 3 -- wait, this doesn't work
    #
    # Let me use a cleaner partition:
    # A (3x3):
    #   A11 (2x2): rows 0-1, cols 0-1
    #   A12 (2x1): rows 0-1, col 2
    #   A21 (1x2): row 2, cols 0-1
    #   A22 (1x1): row 2, col 2
    #
    # B (3x4):
    #   B11 (2x2): rows 0-1, cols 0-1
    #   B12 (2x2): rows 0-1, cols 2-3
    #   B21 (1x2): row 2, cols 0-1
    #   B22 (1x2): row 2, cols 2-3
    #
    # C (3x4):
    #   C11 (2x2): rows 0-1, cols 0-1
    #   C12 (2x2): rows 0-1, cols 2-3
    #   C21 (1x2): row 2, cols 0-1
    #   C22 (1x2): row 2, cols 2-3
    #
    # C11 = A11@B11 + A12@B21
    # C12 = A11@B12 + A12@B22
    # C21 = A21@B11 + A22@B21
    # C22 = A21@B12 + A22@B22
    #
    # Schoolbook rank: 4+2+2+2+2+2+2+2 = 18
    # 
    # Now apply Strassen to the 2x2 @ 2x2 -> 2x2 part for C11:
    # C11 = A11@B11 + A12@B21
    # Let M = A11@B11 (2x2 result)
    # M can be computed with 7 Strassen multiplications:
    # M1 = (A11_11 + A11_22) @ (B11_11 + B11_22)
    # M2 = (A11_21 + A11_22) @ B11_11
    # M3 = A11_11 @ (B11_12 - B11_22)
    # M4 = A11_22 @ (B11_21 - B11_11)
    # M5 = (A11_11 + A11_12) @ B11_22
    # M6 = (A11_21 - A11_11) @ (B11_11 + B11_12)
    # M7 = (A11_12 - A11_22) @ (B11_21 + B11_22)
    # where A11_11, A11_12, A11_21, A11_22 are 1x1 blocks of A11
    # and B11_11, B11_12, B11_21, B11_22 are 1x1 blocks of B11
    #
    # This is getting complex. Let me just use the schoolbook for now
    # and try to find a known lower rank decomposition.
    
    # Known result: For 3x3 @ 3x3, the rank is at most 23 (Strassen-based).
    # For 3x3 @ 3x4, we can extend this.
    
    # Let me use a direct construction with rank 23 or lower.
    # Actually, let me try a simpler approach: use the known 23-rank
    # decomposition for 3x3 @ 3x3 and extend to 3x3 @ 3x4.
    
    # For 3x3 @ 3x3 @ 3x3, the rank is known to be at most 23.
    # We can construct this using Strassen recursion.
    
    # Let me implement the standard 23-rank decomposition for 3x3 @ 3x3.
    # Then extend to handle the 4th column.
    
    # The 23-rank decomposition for 3x3 @ 3x3 uses the following:
    # Split into 2x2 and 1x1 blocks, use 7 Strassen terms for the 2x2 part,
    # plus 6 terms for the cross terms, plus 1 for the 1x1 part.
    # 7 + 6 + 1 = 14? No, that's not right.
    
    # Actually, the known bound is 23 for 3x3 @ 3x3. Let me look up the construction.
    # 
    # One construction:
    # M1 = A11 @ B11
    # M2 = A12 @ B21
    # M3 = A21 @ B12
    # M4 = A22 @ B22
    # M5 = (A11+A22) @ (B11+B22)
    # M6 = (A12+A22) @ B11
    # M7 = A11 @ (B12-B22)
    # M8 = A22 @ (B21-B11)
    # M9 = (A11+A12) @ B22
    # M10 = (A21-A11) @ (B11+B12)
    # M11 = (A12-A22) @ (B21+B22)
    # M12 = (A21+A22) @ (B11+B12)
    # ... this is getting complicated.
    
    # Strassen-style decomposition for 3x3 @ 3x4
    # 
    # We'll use a rank-23 decomposition based on known results.
    # 
    # A (3x3), B (3x4), C (3x4)
    # 
    # Index: A[i,j] for i,j in 0..2; B[j,k] for j in 0..2, k in 0..3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We'll define 23 terms that compute all C[i,k].
    # 
    # Let me use a construction based on splitting into 2x2 and 1x1:
    # 
    # A11 = A[0:2, 0:2] (2x2)
    # A12 = A[0:2, 2]   (2x1)
    # A21 = A[2, 0:2]   (1x2)
    # A22 = A[2, 2]     (1x1)
    # 
    # B11 = B[0:2, 0:2] (2x2)
    # B12 = B[0:2, 2:4] (2x2)
    # B21 = B[2, 0:2]   (1x2)
    # B22 = B[2, 2:4]   (1x2)
    # 
    # C11 = A11@B11 + A12@B21  (2x2)
    # C12 = A11@B12 + A12@B22  (2x2)
    # C21 = A21@B11 + A22@B21  (1x2)
    # C22 = A21@B12 + A22@B22  (1x2)
    # 
    # For the 2x2 @ 2x2 -> 2x2 parts, use Strassen (7 terms each).
    # But we can share some terms between C11 and C12.
    # 
    # Let me define the 7 Strassen terms for A11 @ B11:
    # M1 = (A11_00 + A11_11) @ (B11_00 + B11_11)
    # M2 = (A11_10 + A11_11) @ B11_00
    # M3 = A11_00 @ (B11_01 - B11_11)
    # M4 = A11_11 @ (B11_10 - B11_00)
    # M5 = (A11_00 + A11_01) @ B11_11
    # M6 = (A11_10 - A11_00) @ (B11_00 + B11_01)
    # M7 = (A11_01 - A11_11) @ (B11_10 + B11_11)
    # 
    # Let me implement a concrete rank-23 decomposition.
    # 
    # Based on known results, we can use the following approach:
    # 
    # A (3x3), B (3x4), C (3x4)
    # 
    # We'll define 23 bilinear terms. Each term has coefficients
    # for U (input A), V (input B), and W (output C).
    # 
    # The key insight is that we can use Strassen-like combinations
    # to reduce the number of multiplications.
    # 
    # Let me define the terms explicitly.
    # 
    # Term 0-6: Strassen for the 2x2 core (A[0:2,0:2] @ B[0:2,0:2])
    # Term 7-13: Strassen for (A[0:2,0:2] @ B[0:2,2:4])
    # Term 14: A[0:2,2] @ B[2,0:2]
    # Term 15: A[0:2,2] @ B[2,2:4]
    # Term 16: A[2,0:2] @ B[0:2,0:2]
    # Term 17: A[2,0:2] @ B[0:2,2:4]
    # Term 18: A[2,2] @ B[2,0:2]
    # Term 19: A[2,2] @ B[2,2:4]
    # 
    # That's 20 terms. We can potentially share some.
    # 
    # Actually, let me just use 20 terms for now.
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeff, v_coeff, w_coeff):
        nonlocal col
        for idx, val in u_coeff.items():
            U[idx][col] += val
        for idx, val in v_coeff.items():
            V[idx][col] += val
        for idx, val in w_coeff.items():
            W[idx][col] += val
        col += 1
    
    # A[i,j] -> U index i*m + j = i*3 + j
    # B[j,k] -> V index j*p + k = j*4 + k
    # C[i,k] -> W index k*n + i = k*3 + i
    
    # Term 0-6: Strassen for A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    # A11_00=A[0,0], A11_01=A[0,1], A11_10=A[1,0], A11_11=A[1,1]
    # B11_00=B[0,0], B11_01=B[0,1], B11_10=B[1,0], B11_11=B[1,1]
    # C11_00=C[0,0], C11_01=C[0,1], C11_10=C[1,0], C11_11=C[1,1]
    
    # M1 = (A[0,0]+A[1,1]) @ (B[0,0]+B[1,1])
    add_term({0:1, 4:1}, {0:1, 5:1}, {0:1, 3:1, 6:1, 9:1})  # C[0,0],C[0,1],C[1,0],C[1,1]
    # Wait, this is wrong. M1 contributes to multiple C entries.
    
    # Let me be more careful. In Strassen:
    # C11_00 = M1 + M4 - M5 + M7
    # C11_01 = M3 + M5
    # C11_10 = M2 + M4
    # C11_11 = M1 - M2 + M3 + M6
    # 
    # So each M term contributes to specific C entries with specific coefficients.
    
    # Let me restart with a cleaner approach.
    
    # Reset
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_coeff, v_coeff, w_coeff):
        nonlocal col
        for idx, val in u_coeff.items():
            U[idx][col] += val
        for idx, val in v_coeff.items():
            V[idx][col] += val
        for idx, val in w_coeff.items():
            W[idx][col] += val
        col += 1
    
    # A indices: A[i,j] -> i*3+j
    # B indices: B[j,k] -> j*4+k
    # C indices: C[i,k] -> k*3+i
    
    # Strassen for 2x2 @ 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # For A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]:
    # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[3], d=A[1,1]=U[4]
    # e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[4], h=B[1,1]=V[5]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[3], C[1,1]=W[4]
    
    # M1 = (a+d)(e+h)
    add_term({0:1, 4:1}, {0:1, 5:1}, {0:1, 1:0, 3:0, 4:1})  # contributes to C[0,0] and C[1,1]
    # Actually, M1 contributes to C[0,0] with +1 and C[1,1] with +1
    # But we need to track this properly.
    
    # Use schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    result = {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        pass
    
    return result
# EVOLVE-BLOCK-END
