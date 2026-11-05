"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-term decomposition for 2x2, applied to each 2x2 block
    # We split 3x3 into 2x2 blocks:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # B encoded = [[B11, B12, B13],
    #              [B21, B22, B23],
    #              [B31, B32, B33]]
    # where B[i,j] = B_orig[i, 2j] + t*B_orig[i, 2j+1]
    #
    # For the 2x2 block multiplication, Strassen uses:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # For 3x3, we can use a block approach. But actually, let me use a simpler method:
    # Treat the 3x3 x 3x3 multiplication as using the 21-term decomposition.
    # However, for the idempotent algebra encoding, each "scalar" multiplication
    # in the 21-term decomposition becomes 2 scalar multiplications.
    # Total: 21 * 2 = 42.
    
    # Actually, let me use the direct approach:
    # The 3x3 x 3x3 matrix multiplication tensor has rank at most 27 (schoolbook).
    # With the idempotent algebra encoding, each entry is 2 scalars.
    # So the effective rank is 27 * 2 = 54 (same as schoolbook for 3x6).
    # 
    # To do better, I need a lower-rank decomposition of 3x3 x 3x3.
    # The known minimum for 3x3 x 3x3 is 21 (from 1970s work).
    # With idempotent encoding: 21 * 2 = 42.
    #
    # But implementing the 21-term decomposition correctly is complex.
    # Let me try a different approach: use the fact that we can apply
    # Strassen to 2x2 sub-blocks.
    
    # Alternative: Use the 21-term decomposition for 3x3x3.
    # I'll implement a known rank-21 decomposition.
    
    # From literature, one rank-21 decomposition for 3x3x3:
    # This is complex. Let me use a safer approach.
    
    # Actually, let me use the following approach:
    # Split 3x3 into 1x3 + 2x3 for A, and 3x1 + 3x2 for B.
    # This doesn't directly help.
    
    # Let me just implement the schoolbook to ensure correctness,
    # then try to improve.
    
    # Schoolbook: rank 54
    rank = n * m * p
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
