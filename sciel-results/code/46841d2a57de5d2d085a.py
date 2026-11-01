"""Matrix multiplication 2,4,7 with rank-minimal decomposition using Strassen and block structure."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Split m=4 into two blocks of 2: jb in {0, 1}
    # For each m-block, split p=7 into 3 blocks of 2 and 1 block of 1
    # Use Strassen (rank 7) for 2x2*2x2 and schoolbook (rank 4) for 2x2*2x1
    
    # Strassen 2x2*2x2 decomposition (correct formulation):
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
    
    # Output convention: W[k*n+i, r] where C[i,k] = sum_j A[i,j]*B[j,k]
    # So W index is k*n + i
    
    for jb in range(2):
        j_base = jb * 2  # 0 or 2
        
        for kb in range(4):
            if kb < 3:
                # Strassen 2x2*2x2 block
                k_base = kb * 2  # 0, 2, 4
                
                # A sub-block: rows i=0,1; cols j_base, j_base+1
                # B sub-block: rows j_base, j_base+1; cols k_base, k_base+1
                # C sub-block: rows k_base, k_base+1; cols i=0,1
                # W indices: W[k*n+i]
                #   W[k_base*2+0] = C[0, k_base]  (i=0, k=k_base)
                #   W[k_base*2+1] = C[1, k_base]  (i=1, k=k_base)
                #   W[(k_base+1)*2+0] = C[0, k_base+1]  (i=0, k=k_base+1)
                #   W[(k_base+1)*2+1] = C[1, k_base+1]  (i=1, k=k_base+1)
                
                # U indices: i*m + j
                #   A[0, j_base] -> 0*4+j_base = j_base
                #   A[0, j_base+1] -> j_base+1
                #   A[1, j_base] -> 4+j_base
                #   A[1, j_base+1] -> 4+j_base+1
                
                # V indices: j*p + k
                #   B[j_base, k_base] -> j_base*7+k_base
                #   B[j_base, k_base+1] -> j_base*7+k_base+1
                #   B[j_base+1, k_base] -> (j_base+1)*7+k_base
                #   B[j_base+1, k_base+1] -> (j_base+1)*7+k_base+1
                
                # M1 = (a11+a22)(b11+b22)
                U[j_base][col] = 1
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + k_base][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
                W[k_base * 2 + 0][col] = 1
                W[(k_base + 1) * 2 + 1][col] = 1
                col += 1
                
                # M2 = (a21+a22)(b11+b12)
                U[4 + j_base][col] = 1
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + k_base][col] = 1
                V[j_base * 7 + (k_base + 1)][col] = 1
                W[k_base * 2 + 1][col] = 1
                col += 1
                
                # M3 = (a11-a22)(b21+b22)
                U[j_base][col] = 1
                U[4 + j_base + 1][col] = -1
                V[(j_base + 1) * 7 + k_base][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
                W[(k_base + 1) * 2 + 1][col] = 1
                col += 1
                
                # M4 = a11(b21-b22)
                U[j_base][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = -1
                W[k_base * 2 + 0][col] = 1
                W[(k_base + 1) * 2 + 1][col] = -1
                col += 1
                
                # M5 = a22(b12-b11)
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + (k_base + 1)][col] = 1
                V[j_base * 7 + k_base][col] = -1
                W[k_base * 2 + 0][col] = -1
                W[(k_base + 1) * 2 + 0][col] = 1
                col += 1
                
                # M6 = (a11+a12)b22
                U[j_base][col] = 1
                U[j_base + 1][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
                W[(k_base + 1) * 2 + 0][col] = 1
                W[(k_base + 1) * 2 + 1][col] = -1
                col += 1
                
                # M7 = (a21-a11)b11
                U[4 + j_base][col] = 1
                U[j_base][col] = -1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = 1
                W[k_base * 2 + 1][col] = -1
                col += 1
            else:
                # Schoolbook 2x2*2x1 block
                k_base = 6
                
                # C[0, k_base] = A[0,j0]*B[j0,k] + A[0,j1]*B[j1,k]
                # C[1, k_base] = A[1,j0]*B[j0,k] + A[1,j1]*B[j1,k]
                
                # Product 1: A[0,j_base] * B[j_base, k_base]
                U[j_base][col] = 1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = 1
                col += 1
                
                # Product 2: A[0,j_base+1] * B[j_base+1, k_base]
                U[j_base + 1][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = 1
                col += 1
                
                # Product 3: A[1,j_base] * B[j_base, k_base]
                U[4 + j_base][col] = 1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 1][col] = 1
                col += 1
                
                # Product 4: A[1,j_base+1] * B[j_base+1, k_base]
                U[4 + j_base + 1][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                W[k_base * 2 + 1][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
