"""Rewritten program: Strassen-based rank-35 decomposition for 2x2 * 2x10."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # p=10, so we have 5 independent 2x2 blocks along the k dimension
    num_blocks = p // m  # 5 blocks

    # Strassen's 7-term decomposition for 2x2 matrix multiplication
    # For C = A * B where A,B are 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # 
    # The 7 products are:
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    #
    # Then:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6

    # For the tensor decomposition T_{i,j,k}:
    # U[i*m+j, r] - coefficient of A[i,j] in product r
    # V[j*p+k, r] - coefficient of B[j,k] in product r  
    # W[k*n+i, r] - coefficient of C[i,k] in product r
    # 
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So W is indexed by (k, i) where k is the column of C (which is the same as k in B)
    # and i is the row of C (same as i in A).
    # 
    # The tensor entry T_{(i,j), (j,k), (k,i)} = 1
    # Decomposition: T_{(i,j),(j,k),(k,i)} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # For Strassen on a 2x2 block (fixed k_block with 2 k-values):
    # We need U[i,j,r], V[j,k_local,r], W[k_local,i,r] for r in 0..6
    #
    # Let me define the Strassen coefficients:
    # Product r=0 (M1): (a11+a22)(b11+b22)
    #   U: a11->1, a22->1 => U[0,0]=1, U[1,1]=1 (using 0-indexed: U[i,j] with i=row, j=col of A)
    #   Wait, A[i,j] where i is row (0,1) and j is... 
    #   Actually in the problem, A is 2x2 (n x m = 2x2), B is 2x10 (m x p = 2x10)
    #   So A has rows i=0,1 and columns j=0,1
    #   B has rows j=0,1 and columns k=0..9
    #   C has rows i=0,1 and columns k=0..9
    #
    # For a 2x2 sub-block of B (columns 2*b, 2*b+1) and corresponding C columns:
    # The sub-problem is: C_block[i, k_local] = sum_j A[i,j] * B_block[j, k_local]
    # where k_local in {0,1}
    #
    # Strassen on this 2x2 x 2x2:
    # A_sub = A (full 2x2), B_sub = B[:, 2*b:2*b+2]
    #
    # Coefficients for U (row i of A, col j of A):
    # M1: U[0,0]=1, U[1,1]=1 (a11 and a22)
    # M2: U[1,0]=1, U[1,1]=1 (a21 and a22)  
    # M3: U[0,0]=1 (a11)
    # M4: U[1,1]=1 (a22)
    # M5: U[0,0]=1, U[0,1]=1 (a11 and a12)
    # M6: U[1,0]=1, U[0,0]=-1 (a21 - a11)
    # M7: U[0,1]=1, U[1,1]=-1 (a12 - a22)
    
    # Coefficients for V (row j of B, col k_local of B_sub):
    # M1: V[0,0]=1, V[1,1]=1 (b11 and b22)
    # M2: V[0,0]=1 (b11)
    # M3: V[0,1]=1, V[1,1]=-1 (b12 - b22)
    # M4: V[1,0]=1, V[0,0]=-1 (b21 - b11)
    # M5: V[1,1]=1 (b22)
    # M6: V[0,0]=1, V[0,1]=1 (b11 + b12)
    # M7: V[1,0]=1, V[1,1]=1 (b21 + b22)
    
    # Coefficients for W (row k_local of C, col i of C):
    # M1: W[0,0]=1, W[1,1]=1 (c11 and c22)
    # M2: W[1,0]=1 (c21)
    # M3: W[0,1]=1 (c12)
    # M4: W[0,0]=1, W[1,0]=1 (c11 and c21)
    # M5: W[0,0]=-1, W[0,1]=1 (c11 gets -1, c12 gets +1)
    # M6: W[1,1]=1 (c22)
    # M7: W[0,0]=1 (c11)
    
    # Let me re-verify:
    # c11 = M1 + M4 - M5 + M7
    #   M1: W[0,0]+=1
    #   M4: W[0,0]+=1
    #   M5: W[0,0]+=-1
    #   M7: W[0,0]+=1
    #   Total W[0,0] for c11: 1+1-1+1 = 2? That's wrong.
    #
    # Let me recheck. The W coefficient should be the contribution to C[i,k_local].
    # W[k_local*n + i, r] gives the coefficient for C[i, k_local] from product r.
    #
    # For M1 = (a11+a22)(b11+b22):
    #   c11 += M1, c22 += M1
    #   So W[0,0] += 1 (for c11, k_local=0, i=0), W[1,1] += 1 (for c22, k_local=1, i=1)
    #
    # For M2 = (a21+a22)b11:
    #   c21 += M2
    #   W[0,1] += 1 (for c21, k_local=0, i=1)
    #
    # For M3 = a11(b12-b22):
    #   c12 += M3
    #   W[1,0] += 1 (for c12, k_local=1, i=0)
    #
    # For M4 = a22(b21-b11):
    #   c11 += M4, c21 += M4
    #   W[0,0] += 1, W[0,1] += 1
    #
    # For M5 = (a11+a12)b22:
    #   c11 -= M5, c12 += M5
    #   W[0,0] += -1, W[1,0] += 1
    #
    # For M6 = (a21-a11)(b11+b12):
    #   c22 += M6
    #   W[1,1] += 1
    #
    # For M7 = (a12-a22)(b21+b22):
    #   c11 += M7
    #   W[0,0] += 1
    #
    # Let's verify c11: M1 + M4 - M5 + M7 = 1 + 1 - 1 + 1 = 2? No wait, that's the sum of coefficients.
    # The actual value is: c11 = M1 + M4 - M5 + M7 where each M is a scalar product.
    # So W[0,0] should have entries: M1:1, M4:1, M5:-1, M7:1 => total contribution to c11 is correct.
    
    # Now let's build the full matrices.
    # Rank = 7 * num_blocks = 7 * 5 = 35
    
    rank = 7 * num_blocks
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficient tables (7 products)
    # U_coeffs[r][i][j] = coefficient of A[i,j] in product r
    U_coeffs = [
        # M1
        [[1, 0],
         [0, 1]],
        # M2
        [[0, 0],
         [1, 1]],
        # M3
        [[1, 0],
         [0, 0]],
        # M4
        [[0, 0],
         [0, 1]],
        # M5
        [[1, 1],
         [0, 0]],
        # M6
        [[-1, 0],
         [1, 0]],
        # M7
        [[0, 1],
         [0, -1]],
    ]
    
    # V_coeffs[r][j][k_local] = coefficient of B[j,k_local] in product r
    V_coeffs = [
        # M1
        [[1, 0],
         [0, 1]],
        # M2
        [[1, 0],
         [0, 0]],
        # M3
        [[0, 1],
         [0, -1]],
        # M4
        [[-1, 0],
         [1, 0]],
        # M5
        [[0, 0],
         [0, 1]],
        # M6
        [[1, 1],
         [0, 0]],
        # M7
        [[0, 0],
         [1, 1]],
    ]
    
    # W_coeffs[r][k_local][i] = coefficient for C[i,k_local] from product r
    # W index is k_local * n + i
    W_coeffs = [
        # M1: c11+=1, c22+=1
        [[1, 0],
         [0, 1]],
        # M2: c21+=1
        [[0, 1],
         [0, 0]],
        # M3: c12+=1
        [[0, 0],
         [1, 0]],
        # M4: c11+=1, c21+=1
        [[1, 1],
         [0, 0]],
        # M5: c11+=-1, c12+=1
        [[-1, 0],
         [1, 0]],
        # M6: c22+=1
        [[0, 0],
         [0, 1]],
        # M7: c11+=1
        [[1, 0],
         [0, 0]],
    ]
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = U_coeffs[r][i][j]
            
            # Fill V
            for j in range(m):
                for k_local in range(m):
                    k_global = block * m + k_local
                    V[j * p + k_global][col] = V_coeffs[r][j][k_local]
            
            # Fill W
            for k_local in range(m):
                for i in range(n):
                    k_global = block * m + k_local
                    W[k_global * n + i][col] = W_coeffs[r][k_local][i]
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
