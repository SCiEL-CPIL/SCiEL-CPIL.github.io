"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 115.0

    # Strassen-based block decomposition for 4x4 * 4x6 -> 4x6
    # Split into 2x2 blocks along n and m dimensions
    # A (4x4) -> 4 blocks of 2x2
    # B (4x6) -> 4 blocks of 2x3
    # C (4x6) -> 4 blocks of 2x3
    #
    # Strassen 7 multiplications for 2x2x2 core, extended to 2x2x3
    # Each "multiplication" Mi = Xi * Yi where Xi is 2x2, Yi is 2x3
    # We decompose each Mi using 7 terms per column (3 columns) = 21 terms
    # But we can be smarter: use the 7-multiplication structure directly
    #
    # For a 2x2 * 2x3 product: C[i,k] = sum_j A[i,j]*B[j,k]
    # Strassen gives 7 "products" but each product is still a scalar*scalar
    # Actually for 2x2x2x3, we can use rank 7*3 = 21 schoolbook per block
    # Total: 7 blocks * 21 = 147. Too high.
    #
    # Better: use the known 4x4x4 rank-21 + 4x4x2 extension
    # 4x4x4 Strassen rank 21, 4x4x2 needs additional terms
    #
    # Alternative: direct rank-84 construction
    # Use 2x2 Strassen on the 4x4x4 part (rank 21) 
    # and handle the 4x4x2 part with rank 63? No...
    #
    # Let me construct the rank-84 decomposition directly.
    # The idea: for each of the 2 blocks in m (each 2x2 sub-block of A
    # times 2x3 sub-block of B), use 7 multiplications instead of 8.
    # But the multiplications are 2x2 * 2x3 which have rank 12 schoolbook.
    # Using Strassen structure: 7 "atomic" products where each is a
    # linear combination. The rank contribution per block is 7 * 3 = 21.
    # Two blocks: 42. But we need cross-block terms too.
    #
    # Actually the cleanest: just use schoolbook for the 4x4x6 but
    # apply the 2x2 Strassen identity to reduce.
    #
    # Let me just build the rank-84 via the known construction:
    # 96 - 12 = 84, saving 12 from Strassen on the 4x4x4 sub-problem
    # (saves 4 per 2x2 block pair, 3 block pairs in 4x4 gives 12 saved)
    
    # Build rank-84 decomposition using Strassen on 2x2 blocks
    # For 4x4x4: rank 21 (known). For the remaining 4x4x2: 
    # We can use 2x2 Strassen on 2x2x2 (rank 7) for each of the 
    # 4 row-blocks and 2 col-blocks... 
    
    # Let me use a direct construction:
    # Split rows into 2 groups of 2, cols of B into 2 groups of 2 (for first 4 cols)
    # and handle last 2 cols separately.
    
    # 4x4x4 part: use rank-21 Strassen decomposition
    # 4x4x2 part: use rank-24 (schoolbook 32 reduced by Strassen on 2x2x2)
    # Total: 21 + 24 = 45? That seems too optimistic.
    
    # Let me just implement the verified rank-84 from the knowledge.
    # The construction: apply 2x2 Strassen (7 mults instead of 8) to 
    # each 2x2x2 sub-block. In 4x4x6, we have (4/2)*(4/2)*(6/2) = 2*2*3 = 12
    # sub-blocks of 2x2x2. Each saves 1 (8->7), total saving 12. 96-12=84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    # For each 2x2x2 sub-block, use 7 Strassen terms instead of 8
    # Sub-block at (row_group i_g, col_group j_g, out_group k_g)
    # where i_g in {0,1}, j_g in {0,1}, k_g in {0,1,2}
    # Rows: 2*i_g to 2*i_g+1, Cols of A/B: 2*j_g to 2*j_g+1
    # Output cols: 2*k_g to 2*k_g+1
    
    for i_g in range(2):
        for j_g in range(2):
            for k_g in range(3):
                # 7 Strassen terms for this 2x2x2 block
                # i = 2*i_g + di, j = 2*j_g + dj, k = 2*k_g + dk
                # Strassen 7 products:
                # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
                # M2 = (A[i0,j1]+A[i1,j1])*(B[j0,k0])
                # M3 = (A[i0,j0])*(B[j1,k1]-B[j0,k0])
                # M4 = (A[i1,j0])*(B[j1,k0]-B[j0,k0])
                # M5 = (A[i0,j0]+A[i0,j1])*(B[j1,k1])
                # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
                # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
                
                i0, i1 = 2*i_g, 2*i_g+1
                j0, j1 = 2*j_g, 2*j_g+1
                k0, k1 = 2*k_g, 2*k_g+1
                
                # Each Strassen product contributes to specific (i,k) outputs
                # For term t, U[i*m+j], V[j*p+k], W[k*n+i]
                
                # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
                # Contributes to C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
                # U entries: for (i0,j0): +1, (i1,j1): +1
                # V entries: for (j0,k0): +1, (j1,k1): +1
                # W entries: for (k0,i0): +1, (k1,i0): +1, (k0,i1): +1, (k1,i1): +1
                
                # Actually each Strassen term is ONE rank-1 tensor
                # U[i*m+j] * V[j*p+k] * W[k*n+i]
                
                # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
                # This is not a single rank-1 tensor. It expands to 4 terms.
                # But Strassen replaces 8 schoolbook terms with 7 "products"
                # where each product is a single scalar multiplication.
                
                # The 7 products are:
                # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
                # P2 = (A[i0,j1]+A[i1,j1]) * B[j0,k0]
                # P3 = A[i0,j0] * (B[j1,k1]-B[j0,k0])
                # P4 = A[i1,j0] * (B[j1,k0]-B[j0,k0])
                # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
                # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
                # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
                
                # Each P is a product of two linear combinations.
                # To make this a rank-1 tensor, we need:
                # U[i*m+j] * V[j*p+k] * W[k*n+i] = P for the relevant indices
                
                # This is getting complex. Let me use the known 4x4x4 rank-21
                # and extend it.
                
                pass
    
    # This approach is getting too complex. Let me use the schoolbook
    # with the 2x2 Strassen reduction properly implemented.
    
    # Actually, let me just return the schoolbook (rank 96) as fallback
    # and implement the rank-84 properly.
    
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Now implement rank-84 via Strassen on 2x2x2 sub-blocks
    # Each 2x2x2 sub-block: 8 schoolbook terms -> 7 Strassen terms
    # 12 sub-blocks * 7 = 84 terms total
    
    rank_84 = 84
    U84 = [[0] * rank_84 for _ in range(n * m)]
    V84 = [[0] * rank_84 for _ in range(m * p)]
    W84 = [[0] * rank_84 for _ in range(p * n)]
    
    col = 0
    for i_g in range(2):
        for j_g in range(2):
            for k_g in range(3):
                i0, i1 = 2*i_g, 2*i_g+1
                j0, j1 = 2*j_g, 2*j_g+1
                k0, k1 = 2*k_g, 2*k_g+1
                
                # 7 Strassen terms
                # Each term: U has entries at specific (i,j), V at (j,k), W at (k,i)
                
                # Term 1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
                # This contributes to C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
                # But as a single rank-1 tensor, we need U[i*m+j]*V[j*p+k]*W[k*n+i]
                # The issue: a single rank-1 tensor only contributes to ONE (i,j,k) triple
                # So we need to expand each Strassen product into its component rank-1 tensors
                
                # Wait, I think I'm confusing myself. Let me reconsider.
                
                # In tensor decomposition, each term c gives:
                # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # This contributes to the (i,j,k) entry of the tensor.
                
                # For Strassen, the 7 "products" are not single rank-1 tensors.
                # Each product P = (sum of A terms) * (sum of B terms) expands to
                # multiple rank-1 tensors.
                
                # The key insight: Strassen reduces the NUMBER of scalar
                # multiplications from 8 to 7, but in tensor rank terms,
                # each "product" still requires multiple rank-1 terms.
                
                # So the tensor rank reduction comes from a different mechanism.
                
                # Let me just use the schoolbook rank 96 as the safe answer.
                pass
    
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
