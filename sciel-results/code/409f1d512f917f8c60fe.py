"""Matrix multiplication 3x3 x 3x6 with rank-21 bilinear algorithm."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # We'll use a rank-21 algorithm based on decomposing B into two 3x3 blocks
    # and using a shared structure.
    
    # B is 3x6, so B = [B1 | B2] where B1, B2 are 3x3.
    # C = A*B = [A*B1 | A*B2], where A*B1 and A*B2 are 3x3.
    
    # For 3x3 * 3x3, we can use a rank-19 algorithm (Laderman's).
    # But we need to share computations between A*B1 and A*B2.
    
    # Actually, let's use a simpler approach: use the standard rank-27 for 3x3*3x3,
    # but since we have two such products, that's 54. We need to do better.
    
    # Let me use a direct construction with rank 21.
    
    # The key insight: for 3x3 * 3x6, we can write B as a sum of 6 rank-1 matrices
    # B[:,k] * e_k^T, but that gives rank 18 if we're clever.
    
    # Actually, let me try a different approach. Use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can group the 18 output entries (3x6) into groups.
    
    # Let me try to construct a rank-18 algorithm directly.
    
    # For each of the 18 output entries C[i,k], we need sum_j A[i,j]*B[j,k].
    # If we could compute all 18 with 18 multiplications, that would be ideal.
    
    # One approach: use the identity that for fixed i, C[i,:] = A[i,:] * B
    # which is a row vector (1x3) times (3x6) = (1x6).
    
    # For a 1x3 * 3x6 multiplication, the rank is at most 18 (schoolbook),
    # but can be lower. Actually, for 1x3 * 3x6, the rank is 18 (it's just
    # 18 scalar multiplications with no sharing possible in the bilinear sense
    # unless we use more structure).
    
    # Wait, that's not right. For 1x3 * 3x6, we compute 6 outputs, each being
    # a sum of 3 products. The tensor rank of this is 18 (each output needs its
    # own 3 products, and there's no sharing).
    
    # So for 3 rows, that's 3*18 = 54, which is the schoolbook rank.
    
    # To do better, we need to share across rows.
    
    # Let me try a different strategy: use the fact that 3x3*3x6 can be seen
    # as a subcase of 4x4*4x8 or similar, but that's complex.
    
    # For now, let me implement a valid rank-54 algorithm (schoolbook) and
    # see if I can improve it.
    
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]  # 9 x 54
    V = [[0] * rank for _ in range(m * p)]  # 18 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
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
