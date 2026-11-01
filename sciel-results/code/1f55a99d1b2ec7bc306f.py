"""Matrix multiplication 2,4,7 with rank-minimal decomposition using Strassen and block structure."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    # We split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}
    # For each block, we split p=7 into 3 blocks of size 2 (k in {0,1}, {2,3}, {4,5}) 
    # and 1 block of size 1 (k in {6}).
    
    # For 2x2 * 2x2 blocks, we use Strassen (rank 7).
    # For 2x2 * 2x1 blocks, we use schoolbook (rank 4).
    
    # Total rank: 2 j-blocks * (3 * 7 + 1 * 4) = 2 * (21 + 4) = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    # Strassen 2x2*2x2 coefficients (standard formulation):
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11+b12)
    # M3 = (a11-a22)(b21+b22)
    # M4 = a11(b21-b22)
    # M5 = a22(b12-b11)
    # M6 = (a11+a12)b22
    # M7 = (a21-a11)b11
    # C11 = M1 + M4 - M5 + M7
    # C12 = M6 + M5
    # C21 = M2 - M7
    # C22 = M1 - M4 + M3 - M6
    
    # Mapping to our indices:
    # A[i, j] where i in {0,1}, j in {j_base, j_base+1}
    # B[j, k] where j in {j_base, j_base+1}, k in {k_base, k_base+1}
    # C[i, k] = sum_j A[i,j] B[j,k]
    # Output W[k*n + i, col]
    # So for the 2x2 block:
    #   C[0, k_base]   -> W[k_base*2 + 0, col]     (i=0, k=k_base)
    #   C[0, k_base+1] -> W[(k_base+1)*2 + 0, col] (i=0, k=k_base+1)
    #   C[1, k_base]   -> W[k_base*2 + 1, col]     (i=1, k=k_base)
    #   C[1, k_base+1] -> W[(k_base+1)*2 + 1, col] (i=1, k=k_base+1)
    
    # Strassen products for a given (j_base, k_base):
    # a11 = A[0, j_base], a12 = A[0, j_base+1]
    # a21 = A[1, j_base], a22 = A[1, j_base+1]
    # b11 = B[j_base, k_base], b12 = B[j_base, k_base+1]
    # b21 = B[j_base+1, k_base], b22 = B[j_base+1, k_base+1]
    
    # U row indices: i*m + j
    #   A[0, j_base]   -> 0*4 + j_base = j_base
    #   A[0, j_base+1] -> 0*4 + j_base+1 = j_base+1
    #   A[1, j_base]   -> 1*4 + j_base = 4+j_base
    #   A[1, j_base+1] -> 1*4 + j_base+1 = 4+j_base+1
    
    # V row indices: j*p + k
    #   B[j_base, k_base]     -> j_base*7 + k_base
    #   B[j_base, k_base+1]   -> j_base*7 + k_base+1
    #   B[j_base+1, k_base]   -> (j_base+1)*7 + k_base
    #   B[j_base+1, k_base+1] -> (j_base+1)*7 + k_base+1
    
    # W row indices: k*n + i
    #   W[k_base*2 + 0, col]     (C[0, k_base])
    #   W[(k_base+1)*2 + 0, col] (C[0, k_base+1])
    #   W[k_base*2 + 1, col]     (C[1, k_base])
    #   W[(k_base+1)*2 + 1, col] (C[1, k_base+1])
    
    for jb in range(2):
        j_base = jb * 2  # 0 or 2
        
        for kb in range(4):
            if kb < 3:
                # size 2 k-block, use Strassen (rank 7)
                k_base = kb * 2  # 0, 2, 4
                
                # M1 = (a11+a22)(b11+b22)
                U[j_base][col] = 1
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + k_base][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
                W[k_base * 2 + 0][col] = 1  # C11
                W[(k_base + 1) * 2 + 1][col] = 1  # C22
                col += 1
                
                # M2 = (a21+a22)(b11+b12)
                U[4 + j_base][col] = 1
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + k_base][col] = 1
                V[j_base * 7 + (k_base + 1)][col] = 1
                W[k_base * 2 + 1][col] = 1  # C21
                W[(k_base + 1) * 2 + 1][col] = 1  # C22
                col += 1
                
                # M3 = (a11-a22)(b21+b22)
                U[j_base][col] = 1
                U[4 + j_base + 1][col] = -1
                V[(j_base + 1) * 7 + k_base][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
                W[(k_base + 1) * 2 + 1][col] = 1  # C22
                col += 1
                
                # M4 = a11(b21-b22)
                U[j_base][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = -1
                W[k_base * 2 + 0][col] = 1  # C11
                W[(k_base + 1) * 2 + 1][col] = -1  # C22
                col += 1
                
                # M5 = a22(b12-b11)
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + (k_base + 1)][col] = 1
                V[j_base * 7 + k_base][col] = -1
                W[k_base * 2 + 0][col] = -1  # C11
                W[(k_base + 1) * 2 + 0][col] = 1  # C12
                col += 1
                
                # M6 = (a11+a12)b22
                U[j_base][col] = 1
                U[j_base + 1][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
                W[(k_base + 1) * 2 + 0][col] = 1  # C12
                W[(k_base + 1) * 2 + 1][col] = -1  # C22
                col += 1
                
                # M7 = (a21-a11)b11
                U[4 + j_base][col] = 1
                U[j_base][col] = -1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = 1  # C11
                W[k_base * 2 + 1][col] = -1  # C21
                col += 1
                
            else:
                # k_block 3, size 1
                k_base = 6
                # Schoolbook for 2x2 * 2x1
                # C[0, k_base] = A[0, j_base]*B[j_base, k_base] + A[0, j_base+1]*B[j_base+1, k_base]
                # C[1, k_base] = A[1, j_base]*B[j_base, k_base] + A[1, j_base+1]*B[j_base+1, k_base]
                
                # Product 1: A[0, j_base] * B[j_base, k_base]
                U[j_base][col] = 1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = 1
                col += 1
                
                # Product 2: A[0, j_base+1] * B[j_base+1, k_base]
                U[j_base + 1][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = 1
                col += 1
                
                # Product 3: A[1, j_base] * B[j_base, k_base]
                U[4 + j_base][col] = 1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 1][col] = 1
                col += 1
                
                # Product 4: A[1, j_base+1] * B[j_base+1, k_base]
                U[4 + j_base + 1][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                W[k_base * 2 + 1][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
