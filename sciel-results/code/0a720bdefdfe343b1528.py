"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Use Strassen-based decomposition:
    # Split A (4x4) into four 2x2 blocks: A11, A12, A21, A22
    # Split B (4x5) into two 2x5 blocks: B_top, B_bottom
    # C (4x5) has blocks C_top (2x5) = A11*B_top + A12*B_bottom, C_bottom (2x5) = A21*B_top + A22*B_bottom
    # Each 2x2 x 2x5 product uses 7*5=35 multiplications via per-column Strassen
    # Total: 4 * 35 = 140, which is worse than schoolbook 80.
    
    # Better approach: use a rank-64 construction.
    # The matrix multiplication tensor for 4x4 * 4x5 has rank at most 64.
    # Construct using the following identity based on tensor rank decomposition.
    
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # where i in [0,4), j in [0,4), k in [0,5)
    
    # Use rank 64 with the following structured decomposition:
    # Index r by (a, b, c) where a,b in [0,4), c in [0,4)
    # For k in [0,4), use the 4x4x4 all-ones tensor decomposition
    # For k=4, handle separately
    
    # Actually, let's use a verified approach: construct rank-64 decomposition
    # using the fact that the 4x4x4 multiplication tensor has rank 49 (Strassen)
    # and we need to extend to 4x4x5.
    
    # Simplest valid improvement: use rank 72 with a structured pattern.
    # But let's aim for rank 64.
    
    # Here's a valid rank-64 decomposition:
    # Use 64 terms where each term is indexed by (i, j, k) with i,j in [0,4), k in [0,4)
    # and for k=4, we need additional terms. This doesn't work directly.
    
    # Let me use a different approach: verify rank 64 works.
    # The bilinear complexity of 4x4 * 4x5 is at most 64.
    
    # Construct the decomposition:
    rank = 64
    U = [[0] * rank for _ in range(n * m)]  # 16 x 64
    V = [[0] * rank for _ in range(m * p)]  # 20 x 64
    W = [[0] * rank for _ in range(p * n)]  # 20 x 64
    
    # Use the following construction based on tensor decomposition:
    # For r = a*16 + b*4 + c where a,b,c in [0,4):
    # U[a*m + b, r] = 1
    # V[b*p + c, r] = 1  (for c in [0,4))
    # W[c*n + a, r] = 1  (for c in [0,4))
    # This handles k in [0,4) but not k=4.
    
    # For k=4, we need additional structure. Let's use a different indexing.
    
    # Alternative: use rank 64 with a more clever pattern.
    # Let r = i*16 + j*4 + k where i,j in [0,4), k in [0,4)
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1 for k < 4
    # W[k*n+i, r] = 1 for k < 4
    # This gives 64 terms covering k=0,1,2,3 but not k=4.
    
    # I need to include k=4. Let me use a different approach.
    
    # Use rank 64: index by (a, b, c) with a,b in [0,4), c in [0,4)
    # For each (a,b,c):
    #   U[a*m+b, r] = 1
    #   V[b*p + c, r] = 1 if c < 4, else 0
    #   V[b*p + 4, r] = 0
    #   W[c*n + a, r] = 1 if c < 4, else 0
    # This only covers k=0,1,2,3.
    
    # I'm stuck. Let me just use the schoolbook with rank 80 as a fallback,
    # but that scores 1.0. Let me try rank 72.
    
    # Actually, let me just return a valid schoolbook decomposition.
    rank = n * m * p  # 80
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
