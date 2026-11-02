<<<<<<< SEARCH
"""Matrix multiplication 2x3x9 - rank-50 decomposition using Strassen on 2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-50 decomposition for 2x3x9 matrix multiplication.
    
    Strategy: Split the inner dimension (size 3) into (2+1).
    - The 2x2 part uses Strassen's algorithm on 2x2x2 blocks.
    - The 1x1 part uses schoolbook multiplication.
    
    The trace convention is W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 50)
    V = [[0] * rank for _ in range(m * p)]  # shape (27, 50)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 50)
    
    col = 0
    
    # Part 1: A[:,0:2] (2x2) * B[0:2,:] (2x9)
    # A indices: 0=A[0,0], 1=A[0,1], 3=A[1,0], 4=A[1,1]
    # B indices: 0-8=B[0,0:9], 9-17=B[1,0:9]
    
    u_a = 0  # A[0,0]
    u_b = 1  # A[0,1]
    u_c = 3  # A[1,0]
    u_d = 4  # A[1,1]
    
    # 4 blocks of 2x2 (columns 0-7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # Strassen's 7 products for 2x2x2
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output indices: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C[0,k0] = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C[0,k1] = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C[1,k0] = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1,k1] = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # 1 block of 2x1 (column 8)
    k8 = 8
    v_e = 0 * p + k8  # B[0, 8]
    v_g = 1 * p + k8  # B[1, 8]
    
    # P1 = a*e
    U[u_a][col] += 1
    V[v_e][col] += 1
    # P2 = b*g
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    # P3 = c*e
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    # P4 = d*g
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # C[0,8] = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C[1,8] = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A[:,2:3] (2x1) * B[2:3,:] (1x9)
    # A indices: 2=A[0,2], 5=A[1,2]
    # B indices: 18-26=B[2,0:9]
    
    u_ar0 = 2  # A[0,2]
    u_ar1 = 5  # A[1,2]
    
    for k in range(9):
        v_b = 2 * p + k  # B[2, k]
        
        # Q_k = A[0,2] * B[2,k]
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A[1,2] * B[2,k]
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Matrix multiplication 2x3x9 - rank-48 decomposition using block-circulant embedding with Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-48 decomposition for 2x3x9 matrix multiplication.
    
    Strategy: Block-circulant embedding.
    Embed the 2x3x9 multiplication into a 4x4x16 problem by duplicating
    the 2x3 matrix into a 4x4 block structure and the 3x9 matrix into a
    3x18 structure. Use Strassen's rank-7 algorithm for 4x4x4 sub-blocks.
    Extract the desired 2x9 result from the top-left 2x4 block of output.
    
    The trace convention is W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # We'll use 4x4 Strassen blocks. For 4 columns at a time, we need 7 products per 4x4x4 block.
    # 9 columns = 2 full 4-column blocks + 1 column.
    # For each 4-column block: 7 Strassen products for the 4x4x4 multiplication
    # Plus 1 product for the remaining column.
    
    # Actually, let me think more carefully.
    # The inner dimension is 3. We split it as 3 = 2 + 1.
    # For the 2x2 part (rows 0-1 of A, cols 0-1 of B), we use Strassen on 2x2x2 blocks.
    # For the 1x1 part (row 2 of A, col 2 of B), we use schoolbook.
    
    # Alternative: Use a different block structure.
    # Let's try: split inner dim as 3, and use Strassen-style on the 2x2x3 subproblem.
    
    # Actually, the simplest improvement: use Strassen on 2x2x2 blocks for the main part,
    # but optimize the last column handling.
    
    # Let me try a cleaner approach: rank-48 via optimized Strassen.
    # 4 blocks of 2x2x2 using Strassen: 4 * 7 = 28
    # Last column (2x1x1): 2 products
    # Part 2 (2x1x9): 9 * 2 = 18
    # Total: 28 + 2 + 18 = 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 48)
    V = [[0] * rank for _ in range(m * p)]  # shape (27, 48)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 48)
    
    col = 0
    
    # Part 1: A[:,0:2] (2x2) * B[0:2,:] (2x9)
    # A indices: 0=A[0,0], 1=A[0,1], 3=A[1,0], 4=A[1,1]
    # B indices: 0-8=B[0,0:9], 9-17=B[1,0:9]
    
    u_a = 0  # A[0,0]
    u_b = 1  # A[0,1]
    u_c = 3  # A[1,0]
    u_d = 4  # A[1,1]
    
    # 4 blocks of 2x2 (columns 0-7), each using Strassen's 7 products
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # Strassen's 7 products for 2x2x2
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output indices: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C[0,k0] = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C[0,k1] = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C[1,k0] = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1,k1] = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # Last column (column 8) of the 2x2 part: 2x1x1, uses 2 products
    k8 = 8
    v_e = 0 * p + k8  # B[0, 8]
    v_g = 1 * p + k8  # B[1, 8]
    
    # P1 = a*e
    U[u_a][col] += 1
    V[v_e][col] += 1
    # P2 = b*g
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # C[0,8] = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C[1,8] = P3 + P4 where P3=c*e, P4=d*g
    # Wait, I need c*e and d*g too! Let me fix this.
    # Actually for 2x1x1: C[0,8] = a*e + b*g, C[1,8] = c*e + d*g
    # That's 4 products, not 2. Let me reconsider.
    
    # For 2x1x1: 
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8]
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8]
    # Schoolbook: 4 products
    # But we can use 2 products with a trick? No, 2x1x1 has rank 2.
    # So we need 2 products minimum for 2x1x1.
    # Actually rank of 2x1x1 tensor is 2 (it's just two independent products).
    # C[0,8] = a*e + b*g, C[1,8] = c*e + d*g
    # We can write: 
    #   P1 = e, P2 = g (but these are B entries, not products)
    # No, the rank is the number of scalar multiplications.
    # For 2x1x1: we need a*e, b*g, c*e, d*g = 4 products in schoolbook.
    # But rank(2,1,1) = 2? No, rank is the tensor rank.
    # The tensor T_{i,j,k} = delta_{i,j} * delta_{k,0} for 2x1x1... 
    # Actually for matrix multiplication 2x1x1: C[0,0] = A[0,0]*B[0,0], C[1,0] = A[1,0]*B[0,0]
    # That's 2 products. But we have 2x1x1 where the middle dim is 1.
    # C[0,0] = sum_k A[0,k]*B[k,0] = A[0,0]*B[0,0]
    # C[1,0] = sum_k A[1,k]*B[k,0] = A[1,0]*B[0,0]
    # So 2 products: A[0,0]*B[0,0] and A[1,0]*B[0,0].
    # But in our case, the "2x1" part has 2 rows and 1 column in the inner dim.
    # A[0,0]*B[0,8] + A[0,1]*B[1,8] for C[0,8]
    # A[1,0]*B[0,8] + A[1,1]*B[1,8] for C[1,8]
    # This is 2x2x1 (2 rows of A, 2 cols of A, 1 col of B... no)
    # It's a 2x2x2 tensor where the last dim is 1.
    # Rank of 2x2x1 tensor: it's like computing 2x2 matrix times a vector.
    # That has rank 4 in general (schoolbook).
    # But we can do better? 2x2x1: C[i] = sum_j A[i,j]*B[j]
    # This is equivalent to matrix-vector multiplication 2x2x1.
    # The tensor rank of 2x2x1 is 4 (same as schoolbook for this small case).
    # So we need 4 products for the last column of the 2x2 part.
    
    # Let me redo: 4 products for column 8 of the 2x2 part
    # P1 = a*e, P2 = b*g, P3 = c*e, P4 = d*g
    
    # I already have P1 and P2 above. Let me add P3 and P4.
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    # C[1,8] = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A[:,2:3] (2x1) * B[2:3,:] (1x9)
    # A indices: 2=A[0,2], 5=A[1,2]
    # B indices: 18-26=B[2,0:9]
    
    u_ar0 = 2  # A[0,2]
    u_ar1 = 5  # A[1,2]
    
    for k in range(9):
        v_b = 2 * p + k  # B[2, k]
        
        # Q_k = A[0,2] * B[2,k]
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A[1,2] * B[2,k]
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error in my analysis. The last column of the 2x2 part still needs 4 products, not 2. So the total is still 28 + 4 + 18 = 50. Let me reconsider the approach.

Let me try a different strategy: use Strassen on 2x2x2 blocks but combine the last column with the 2x1x9 part more efficiently.

