"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen-Winograd 2x2x2 decomposition (rank 7)
    # Each term: (i, j, k, u, v, w) contributes u*A[i,j] * v*B[j,k] * w*C[k,i]
    # For 2x2x2, the 7 terms are:
    strassen_2x2 = [
        (0, 0, 0, 1,  1,  1),   # A00*B00 -> C00
        (0, 0, 1, 1,  1,  0),   # A00*B01 -> C01
        (0, 1, 0, 1,  0,  1),   # A01*B10 -> C00
        (0, 1, 1, 0,  1,  1),   # A01*B11 -> C01
        (1, 0, 0, 0,  1,  1),   # A10*B00 -> C10
        (1, 0, 1, 1,  1,  0),   # A10*B01 -> C11
        (1, 1, 0, -1, 0,  1),   # A11*B10 -> C10
        (1, 1, 1, 1,  1,  1),   # A11*B11 -> C11
    ]
    # Actually, let's use the standard Strassen 7-term formula:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me construct a proper 4x4x6 decomposition using block decomposition.
    # We'll use Strassen for 2x2 blocks.
    
    # For 4x4x6, we can decompose into:
    # - Two 2x2x6 blocks (for rows 0-1 and 2-3)
    # - Each 2x2x6 can be further decomposed
    
    # Actually, let's use a simpler approach: use Strassen on 2x2 sub-blocks
    # and schoolbook for the rest.
    
    # The simplest correct approach that reduces rank:
    # Use Strassen for each 2x2x2 sub-block where possible.
    
    # For 4x4x6:
    # Split into 4x4x4 + 4x4x2
    # For 4x4x4: use two Strassen 2x2x2 blocks (rank 14) + schoolbook for remaining (rank 32) = 46? No.
    
    # Let me think differently. Use a verified construction.
    # 
    # For 4x4x4, the optimal rank is 18. Let's use a known rank-18 decomposition.
    # But to be safe, let's use a block-based approach that's verifiably correct.
    
    # Approach: Use Strassen for 2x2 blocks.
    # 4x4x4 = four 2x2x2 blocks, each using Strassen (rank 7) = rank 28
    # But we can do better with overlapping.
    
    # For now, let's use a correct but not optimal decomposition:
    # 4x4x6 = schoolbook rank 96, but we can reduce using Strassen on 2x2 sub-blocks.
    
    # Let's construct: for each pair of rows (0,1) and (2,3), and for each 2-column block
    # of B, use Strassen.
    
    # Actually, the cleanest approach: use the full schoolbook but verify it's correct.
    # Then try to reduce rank using Strassen where applicable.
    
    # For a guaranteed correct answer, use schoolbook:
    rank = n * m * p  # 4*4*6 = 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
