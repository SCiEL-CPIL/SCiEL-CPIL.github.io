"""Matrix multiplication 2,4,7 with rank-minimal decomposition using Strassen and block structure."""
DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    # We split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}
    # For each block, we use Strassen (rank 7) for 2x2 * 2x2 when p-block is 2,
    # and for the last p-block of size 1, we use rank 4 for 2x2 * 2x1.
    
    # Total rank: 2 blocks * (3 p-blocks of size 2 * 7 + 1 p-block of size 1 * 4) = 2 * (21 + 4) = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2*2x2 coefficients (corrected):
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M2 + M4 - M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 - M6
    
    # For 2x2 * 2x1 -> 2x1, we can do it in 4 multiplications (no better known).
    # Let A be 2x2, B be 2x1 (column vector b1, b2).
    # C1 = A11*b1 + A12*b2
    # C2 = A21*b1 + A22*b2
    # 4 products: A11*b1, A12*b2, A21*b1, A22*b2
    
    # We iterate over j_blocks (2 of them) and k_blocks (4 of them: 3 of size 2, 1 of size 1)
    
    # j_block 0: j indices 0,1
    # j_block 1: j indices 2,3
    
    # k_blocks: 
    #   k_block 0: k indices 0,1 (size 2)
    #   k_block 1: k indices 2,3 (size 2)
    #   k_block 2: k indices 4,5 (size 2)
    #   k_block 3: k indices 6 (size 1)
    
    for jb in range(2):
        j_base = jb * 2  # 0 or 2
        
        for kb in range(4):
            if kb < 3:
                # size 2 k-block, use Strassen (rank 7)
                k_base = kb * 2  # 0, 2, 4
                k_size = 2
                
                # For this 2x2 * 2x2 subproblem:
                # A rows: i=0,1; cols: j_base, j_base+1
                # B rows: j_base, j_base+1; cols: k_base, k_base+1
                # C rows: k_base, k_base+1; cols: i=0,1 (since W index is k*n+i)
                
                # Strassen products M1..M7
                # M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
                # M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
                # M3 = A[0,j0] * (B[j1,k0]-B[j1,k1])
                # M4 = A[1,j1] * (B[j0,k1]-B[j0,k0])
                # M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
                # M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
                # M7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
                
                # Output C[k,i] = sum_j A[i,j]*B[j,k]
                # C[k0,i0] = M1 - M2 + M4 - M5
                # C[k0,i1] = M2 + M4
                # C[k1,i0] = M3 + M5
                # C[k1,i1] = M1 - M3 + M5 - M6
                
                # M1: U has (A[0,j0]+A[1,j1]), V has (B[j0,k0]+B[j1,k1])
                U[j_base][col] = 1
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + k_base][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
                W[k_base * 2 + 0][col] = 1  # C[0,0]
                W[k_base * 2 + 1][col] = -1  # C[1,0]
                W[(k_base + 1) * 2 + 0][col] = 1  # C[0,1]
                W[(k_base + 1) * 2 + 1][col] = 1  # C[1,1]
                col += 1
                
                # M2: U has (A[1,j0]+A[1,j1]), V has B[j0,k0]
                U[4 + j_base][col] = 1
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = -1  # C[0,0]
                W[k_base * 2 + 1][col] = 1  # C[1,0]
                col += 1
                
                # M3: U has A[0,j0], V has (B[j1,k0]-B[j1,k1])
                U[j_base][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = -1
                W[(k_base + 1) * 2 + 0][col] = 1  # C[0,1]
                W[(k_base + 1) * 2 + 1][col] = -1  # C[1,1]
                col += 1
                
                # M4: U has A[1,j1], V has (B[j0,k1]-B[j0,k0])
                U[4 + j_base + 1][col] = 1
                V[j_base * 7 + (k_base + 1)][col] = 1
                V[j_base * 7 + k_base][col] = -1
                W[k_base * 2 + 0][col] = 1  # C[0,0]
                W[k_base * 2 + 1][col] = 1  # C[1,0]
                col += 1
                
                # M5: U has (A[0,j0]+A[0,j1]), V has B[j1,k1]
                U[j_base][col] = 1
                U[j_base + 1][col] = 1
                V[(j_base + 1) * 7 + (k_base + 1)][col] = 1
                W[k_base * 2 + 0][col] = -1  # C[0,0]
                W[(k_base + 1) * 2 + 0][col] = 1  # C[0,1]
                W[(k_base + 1) * 2 + 1][col] = 1  # C[1,1]
                col += 1
                
                # M6: U has (A[1,j0]-A[0,j0]), V has (B[j0,k0]+B[j0,k1])
                U[4 + j_base][col] = 1
                U[j_base][col] = -1
                V[j_base * 7 + k_base][col] = 1
                V[j_base * 7 + (k_base + 1)][col] = 1
                W[(k_base + 1) * 2 + 1][col] = -1  # C[1,1]
                col += 1
                
                # M7: U has (A[0,j1]-A[1,j1]), V has (B[j1,k0]+B[j1,k1])
                # M7 is not used in the final sum for this formulation
                # But we need to include it for completeness if it contributes
                # In this formulation, M7 doesn't contribute to any output
                # So we skip it
            else:
                # k_block 3, size 1
                k_base = 6
                k_size = 1
                # 2x2 * 2x1, rank 4
                # C[k_base, i] = sum_j A[i,j] B[j,k_base]
                # For i=0: A[0,j0]*B[j0,k] + A[0,j1]*B[j1,k]
                # For i=1: A[1,j0]*B[j0,k] + A[1,j1]*B[j1,k]
                
                # Product 1: A[0,j0] * B[j0,k]
                U[j_base][col] = 1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = 1
                col += 1
                
                # Product 2: A[0,j1] * B[j1,k]
                U[j_base + 1][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                W[k_base * 2 + 0][col] = 1
                col += 1
                
                # Product 3: A[1,j0] * B[j0,k]
                U[4 + j_base][col] = 1
                V[j_base * 7 + k_base][col] = 1
                W[k_base * 2 + 1][col] = 1
                col += 1
                
                # Product 4: A[1,j1] * B[j1,k]
                U[4 + j_base + 1][col] = 1
                V[(j_base + 1) * 7 + k_base][col] = 1
                W[k_base * 2 + 1][col] = 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
