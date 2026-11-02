DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 3x5 x 5x5 matrix multiplication, we can use a rank-19 decomposition
    # for each 5x5x5 block (one per row of the 3x5 matrix), giving total rank 3*19=57.
    
    # The rank-19 decomposition for 5x5 matrix multiplication uses:
    # - 7 Strassen terms for the top-left 2x2 block
    # - 7 Strassen terms for the bottom-right 2x2 block  
    # - 5 cross terms for the remaining entries
    
    # Let me define the rank-19 decomposition for C = A*B where A,B are 5x5.
    # Using Strassen on two disjoint 2x2 blocks plus cross terms.
    
    # Strassen 2x2 decomposition:
    # M1 = A[0,0]*B[0,0]
    # M2 = (A[0,0]+A[0,1])*(B[1,0]+B[1,1])
    # M3 = A[0,1]*B[1,1]
    # M4 = (A[1,0]+A[1,1])*(B[0,0]+B[0,1])
    # M5 = A[1,0]*B[0,0]
    # M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # M7 = (A[0,0]+A[1,0])*(B[0,1]-B[1,1])
    
    # C[0,0] = M1 - M3 + M5 + M7
    # C[0,1] = M2 - M3 + M5 - M6
    # C[1,0] = M4 - M5 + M6 + M7
    # C[1,1] = M2 - M1 + M3 - M7
    
    # For 5x5, partition into:
    # Block 1: rows 0-1, cols 0-1 (top-left 2x2)
    # Block 2: rows 3-4, cols 3-4 (bottom-right 2x2)
    # Cross: all other entries
    
    # This gives 7 + 7 = 14 terms for the two 2x2 blocks.
    # Remaining entries: 
    # Row 0: C[0,2], C[0,3], C[0,4] -> 3 entries
    # Row 1: C[1,2], C[1,3], C[1,4] -> 3 entries
    # Row 2: C[2,0], C[2,1], C[2,2], C[2,3], C[2,4] -> 5 entries
    # Row 3: C[3,0], C[3,1], C[3,2] -> 3 entries
    # Row 4: C[4,0], C[4,1], C[4,2] -> 3 entries
    # Total: 3+3+5+3+3 = 17 entries, but many can be shared.
    
    # Actually, let me use a simpler approach. The known rank-19 for 5x5
    # can be constructed by applying Strassen to a 4x4 submatrix and handling
    # the 5th row/col separately.
    
    # Let me use a different strategy: apply Strassen to the 4x4 top-left,
    # then handle the 5th row and column.
    
    # For 4x4 Strassen: 2^3 = 8 multiplications
    # Then for the 5th row of A and 5th column of B:
    # C[4,0:4] = A[4,0:4] * B[0:4,0:4] -> 4 terms
    # C[0:4,4] = A[0:4,4] * B[4,0:4] -> 4 terms  
    # C[4,4] = A[4,4]*B[4,4] -> 1 term
    # Total: 8 + 4 + 4 + 1 = 17. Still not 19.
    
    # Let me just use the schoolbook for now with rank 75, but try to be smarter.
    
    # Actually, for the (3,5,5) problem, we can use rank 3*25=75 schoolbook,
    # or we can try to find a better decomposition.
    
    # Let me implement the rank-19 decomposition properly.
    
    # I'll use the following approach for 5x5x5 rank-19:
    # Use Strassen on two 2x2 blocks (positions (0,0) and (2,2)) and handle
    # the remaining 5 entries with 5 additional terms.
    
    # Strassen block 1: indices (0,0), (0,1), (1,0), (1,1)
    # Strassen block 2: indices (2,0), (2,1), (3,0), (3,1) -- no, let me use (3,3),(3,4),(4,3),(4,4)
    
    # Let me define the 19 bilinear forms explicitly.
    
    rank_5x5 = 19
    rank = n * rank_5x5  # 3 * 19 = 57
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 57
    V = [[0] * rank for _ in range(m * p)]  # 25 x 57
    W = [[0] * rank for _ in range(p * n)]  # 15 x 57
    
    # For each row i of the 3x5 matrix, we have a 5x5 matrix multiplication.
    # The U matrix has entries U[i*5+j, col] = coefficient of A[i,j] in the col-th bilinear form.
    # The V matrix has entries V[j*5+k, col] = coefficient of B[j,k] in the col-th bilinear form.
    # The W matrix has entries W[k*3+i, col] = coefficient of C[k,i] in the col-th bilinear form.
    
    # Wait, the output is C of size 3x5, where C[i,k] = sum_j A[i,j] * B[j,k].
    # So W[k*n+i, col] where n=3, so W[k*3+i, col].
    
    # Let me define the 19 bilinear forms for a single 5x5 multiplication.
    # I'll use Strassen on the top-left 2x2 and bottom-right 2x2, plus cross terms.
    
    # Strassen on A[0:2, 0:2] * B[0:2, 0:2]:
    # M1 = A[0,0]*B[0,0]
    # M2 = (A[0,0]+A[0,1])*(B[1,0]+B[1,1])
    # M3 = A[0,1]*B[1,1]
    # M4 = (A[1,0]+A[1,1])*(B[0,0]+B[0,1])
    # M5 = A[1,0]*B[0,0]
    # M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # M7 = (A[0,0]+A[1,0])*(B[0,1]-B[1,1])
    
    # C[0,0] = M1 - M3 + M5 + M7
    # C[0,1] = M2 - M3 + M5 - M6
    # C[1,0] = M4 - M5 + M6 + M7
    # C[1,1] = M2 - M1 + M3 - M7
    
    # Strassen on A[3:5, 3:5] * B[3:5, 3:5]:
    # M8 = A[3,3]*B[3,3]
    # M9 = (A[3,3]+A[3,4])*(B[4,3]+B[4,4])
    # M10 = A[3,4]*B[4,4]
    # M11 = (A[4,3]+A[4,4])*(B[3,3]+B[3,4])
    # M12 = A[4,3]*B[3,3]
    # M13 = (A[3,4]-A[4,4])*(B[4,3]+B[4,4])
    # M14 = (A[3,3]+A[4,3])*(B[3,4]-B[4,4])
    
    # C[3,3] = M8 - M10 + M12 + M14
    # C[3,4] = M9 - M10 + M12 - M13
    # C[4,3] = M11 - M12 + M13 + M14
    # C[4,4] = M9 - M8 + M10 - M14
    
    # Cross terms (5 more):
    # M15 = A[0,2]*B[2,0]
    # M16 = A[0,2]*B[2,1]
    # M17 = A[1,2]*B[2,0]
    # M18 = A[1,2]*B[2,1]
    # M19 = A[2,0]*B[0,2] + A[2,1]*B[1,2] -- no, this is not bilinear.
    
    # I need to be more careful. Let me just use schoolbook for now.
    
    rank = n * m * p  # 75
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
