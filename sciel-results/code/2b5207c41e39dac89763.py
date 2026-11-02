"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We partition into 4 blocks of 2x2x3.
    # Each 2x2x3 block is decomposed into:
    # - Strassen (rank 7) for the first 2 columns
    # - Schoolbook (rank 4) for the 3rd column
    # Total rank per block = 11.
    # Total rank = 4 * 11 = 44.
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    for i_block in range(2):
        for j_block in range(2):
            # Block indices
            i0 = i_block * 2
            j0 = j_block * 2
            # k columns are 0..8. 
            # We split k into 3 groups of 3? 
            # The problem is 4x4x9. 
            # A is 4x4. B is 4x9.
            # We partition A into 2x2 blocks (i, j).
            # We partition B into 2x3 blocks (j, k).
            # So for each (i_block, j_block), we have a 2x2 A sub-block and a 2x3 B sub-block.
            # The output C sub-block is 2x3.
            
            # Strassen for first 2 columns of the 2x3 B block.
            # Schoolbook for the 3rd column.
            
            # Strassen 7 terms for 2x2x2
            # A_sub: 2x2. B_sub1: 2x2 (cols 0,1 of the 2x3 block).
            # A_sub: 2x2. B_sub2: 2x1 (col 2 of the 2x3 block).
            
            # Strassen terms:
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)b00
            # M3 = a00(b01-b11)
            # M4 = (a00+a01)(b10+b11)
            # M5 = (a10-a00)(b00+b01)
            # M6 = (a01+a11)(b10-b11)
            # M7 = (a00-a10)(b10+b11)
            
            # Schoolbook for 3rd column (index 2 in local 2x3 block):
            # M8 = a00*b02
            # M9 = a01*b12
            # M10 = a10*b02
            # M11 = a11*b12
            
            # Local indices for A: (i, j) where i,j in 0..1
            # Local indices for B: (j, k) where j in 0..1, k in 0..2
            # Local indices for C: (i, k) where i in 0..1, k in 0..2
            
            # Map to global indices
            # A[i0+i, j0+j]
            # B[j0+j, k_global]
            # C[i0+i, k_global]
            
            # The 2x3 B block has columns k_global = j0*0 + 3*? No.
            # B is 4x9. 
            # We partition B into 2x3 blocks.
            # Row block j0 (2 rows). Col block? 
            # There are 9 columns. 
            # We can't partition 9 into 2x3 blocks evenly for 4 blocks?
            # 4 blocks of A (2x2). 
            # Each A block (i0, j0) multiplies a B block (j0, :).
            # The B block is 2x9.
            # We partition the 9 columns into 3 groups of 3.
            # So for each (i0, j0), we have 3 sub-problems of 2x2x3?
            # No, the A block is 2x2. The B block is 2x9.
            # We can split the 9 columns into three 3-column chunks.
            # So for each (i0, j0), we have 3 instances of 2x2x3.
            # Total instances: 4 (A blocks) * 3 (col chunks) = 12 instances.
            # Rank per instance: 11 (7 Strassen + 4 Schoolbook).
            # Total rank: 12 * 11 = 132.
            # This is barely better than 144.
            
            # Let's try to use rank 7 for 2x2x3.
            # If R(2,2,3)=7, then total rank = 12 * 7 = 84.
            # This is a significant improvement.
            
            # I will implement the 7-term decomposition for 2x2x3.
            # Since I don't have the exact 7 terms for 3 columns, I will use the 7 terms of Strassen for the first 2 columns and assume the 3rd is 0? 
            # No, I will use the 7 terms of Strassen for the first 2 columns and 0 for the 3rd, which is incorrect.
            
            # I will use the schoolbook for all, but optimized?
            # No, I will use the 7-term Strassen for 2x2x2 and 4-term schoolbook for 2x2x1.
            # Total 11 per 2x2x3 block.
            # 12 blocks * 11 = 132.
            
            pass

    # Given the complexity and risk of incorrect 2x2x3 decomposition, 
    # I will return the schoolbook decomposition to ensure correctness, 
    # but I will try to implement the 44-rank version for 4x4x4 and 0 for the rest? No.
    
    # I will implement the 132-rank version (Strassen + Schoolbook).
    rank_total = 132
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    for i_block in range(2):
        for j_block in range(2):
            for k_chunk in range(3):
                # 2x2x3 block
                # A sub-block: rows i0..i0+1, cols j0..j0+1
                # B sub-block: rows j0..j0+1, cols k0..k0+2
                # C sub-block: rows i0..i0+1, cols k0..k0+2
                
                i0 = i_block * 2
                j0 = j_block * 2
                k0 = k_chunk * 3
                
                # Strassen for first 2 columns (k0, k0+1)
                # 7 terms
                # a00=A[i0,j0], a01=A[i0,j0+1], a10=A[i0+1,j0], a11=A[i0+1,j0+1]
                # b00=B[j0,k0], b01=B[j0,k0+1], b10=B[j0+1,k0], b11=B[j0+1,k0+1]
                
                # M1
                U[i0*m + j0][col_offset] += 1
                U[i0*m + j0+1][col_offset] += 1
                V[j0*p + k0][col_offset] += 1
                V[j0+1*p + k0+1][col_offset] += 1
                W[k0*n + i0][col_offset] += 1
                W[(k0+1)*n + i0][col_offset] += 1
                W[k0*n + i0+1][col_offset] += 1
                W[(k0+1)*n + i0+1][col_offset] += 1
                col_offset += 1
                
                # M2
                U[i0+1*m + j0][col_offset] += 1
                U[i0+1*m + j0+1][col_offset] += 1
                V[j0*p + k0][col_offset] += 1
                W[k0*n + i0+1][col_offset] += 1
                W[(k0+1)*n + i0+1][col_offset] += 1
                col_offset += 1
                
                # M3
                U[i0*m + j0][col_offset] += 1
                V[j0*p + k0+1][col_offset] += 1
                V[j0+1*p + k0+1][col_offset] -= 1
                W[(k0+1)*n + i0][col_offset] += 1
                W[(k0+1)*n + i0+1][col_offset] -= 1
                col_offset += 1
                
                # M4
                U[i0*m + j0][col_offset] += 1
                U[i0*m + j0+1][col_offset] += 1
                V[j0+1*p + k0][col_offset] += 1
                V[j0+1*p + k0+1][col_offset] += 1
                W[k0*n + i0][col_offset] += 1
                W[(k0+1)*n + i0][col_offset] += 1
                col_offset += 1
                
                # M5
                U[i0+1*m + j0][col_offset] += 1
                U[i0*m + j0][col_offset] -= 1
                V[j0*p + k0][col_offset] += 1
                V[j0*p + k0+1][col_offset] += 1
                W[k0*n + i0+1][col_offset] += 1
                W[(k0+1)*n + i0+1][col_offset] += 1
                col_offset += 1
                
                # M6
                U[i0*m + j0+1][col_offset] += 1
                U[i0+1*m + j0+1][col_offset] += 1
                V[j0+1*p + k0][col_offset] += 1
                V[j0+1*p + k0+1][col_offset] -= 1
                W[k0*n + i0][col_offset] += 1
                W[(k0+1)*n + i0][col_offset] -= 1
                col_offset += 1
                
                # M7
                U[i0*m + j0][col_offset] += 1
                U[i0+1*m + j0][col_offset] -= 1
                V[j0+1*p + k0][col_offset] += 1
                V[j0+1*p + k0+1][col_offset] += 1
                W[k0*n + i0][col_offset] += 1
                W[(k0+1)*n + i0][col_offset] += 1
                col_offset += 1
                
                # Schoolbook for 3rd column (k0+2)
                # M8 = a00*b02
                U[i0*m + j0][col_offset] += 1
                V[j0*p + k0+2][col_offset] += 1
                W[(k0+2)*n + i0][col_offset] += 1
                col_offset += 1
                
                # M9 = a01*b12
                U[i0*m + j0+1][col_offset] += 1
                V[j0+1*p + k0+2][col_offset] += 1
                W[(k0+2)*n + i0][col_offset] += 1
                col_offset += 1
                
                # M10 = a10*b02
                U[i0+1*m + j0][col_offset] += 1
                V[j0*p + k0+2][col_offset] += 1
                W[(k0+2)*n + i0+1][col_offset] += 1
                col_offset += 1
                
                # M11 = a11*b12
                U[i0+1*m + j0+1][col_offset] += 1
                V[j0+1*p + k0+2][col_offset] += 1
                W[(k0+2)*n + i0+1][col_offset] += 1
                col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
