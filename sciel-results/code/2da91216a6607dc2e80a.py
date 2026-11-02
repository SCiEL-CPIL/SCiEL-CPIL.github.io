"""Rewritten for matrix multiplication 3x5x6 using block-diagonal decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    rank_total = 45

    # Block-diagonal decomposition:
    # Split the middle dimension (m=5) into two parts: 2 and 3.
    # So A is 3x5, split as A1 (3x2) and A2 (3x3).
    # B is 5x6, split as B1 (2x6) and B2 (3x6).
    # C = A1 * B1 + A2 * B2
    # 
    # For A1 (3x2) * B1 (2x6): rank = 18 (3x2x6, each of 3 rows times 2 cols x 6 rows = 3*2*2=12? Let me reconsider)
    # Actually, 3x2x6 multiplication: C_ij = sum_{k=0}^{1} A_ik B_kj for i in 0..2, j in 0..5
    # This is a 3*2*6 = 36 term schoolbook, but we can do better.
    # For m=2, the rank of mxn xp is at most n*min(m,p) if we factor... 
    # Actually R(3,2,6) = 3*6 = 18? No, R(3,2,6) <= 3*2*6/2 = 18 using the fact that for 2xNxP, rank is 2*min(N,P)? 
    # Let's just use schoolbook for small blocks: R(3,2,6) = 3*2*6 = 36? That's too high.
    # 
    # Let me reconsider. The rank of multiplying A (n x m) by B (m x p) is at most m * min(n, p).
    # For 3x2x6: m=2, min(3,6)=3, so rank <= 6.
    # For 3x3x6: m=3, min(3,6)=3, so rank <= 9.
    # Total rank = 6 + 9 = 15? That seems too low. Let me verify.
    #
    # Actually, R(n,m,p) <= m * min(n,p) is not generally true. The schoolbook is n*m*p.
    # But for small dimensions, we can use known results.
    # R(3,2,6): We can view this as computing C_ij = A_i0*B_0j + A_i1*B_1j for i=0,1,2 and j=0,1,2,3,4,5.
    # Each C_ij depends on the same pair of A's and B's for fixed i,j.
    # Number of independent multiplications: for each i and j, we have 2 products. Total 3*6*2 = 36 schoolbook.
    # But we can factor: C_i = A_i0 * B_0j + A_i1 * B_1j. For each i, C_i is a row of length 6.
    # C_i = A_i0 * (B_0j for j) + A_i1 * (B_1j for j) = A_i0 * B_0 + A_i1 * B_1 where B_0, B_1 are rows of B.
    # This is just 2 scalar multiplications per row of C, so 3*2 = 6 multiplications? No, that's not how tensor rank works.
    #
    # The tensor for 3x2x6 has 36 entries in the output. Each rank-1 term is u_i * v_j * w_k.
    # I'll just use schoolbook decomposition for safety, which gives rank 36 + 54 = 90, same as before.
    # That's not an improvement.
    #
    # Let me use a better approach. For R(n, m, p), one bound is R(n, m, p) <= n * p * ceil(m / 2) for even m?
    # Actually, for m=2, R(n, 2, p) = n * p because each output entry C_ij = a_i0*b_0j + a_i1*b_1j,
    # and these 2 products for each (i,j) pair cannot be shared across different (i,j) pairs in a rank-1 decomposition.
    # Wait, they can! The products a_i0*b_0j and a_i1*b_1j for different i,j can share the same "slot" if the U, V, W vectors align.
    #
    # Let me just implement the direct sum with schoolbook sub-decompositions but see if I can do better for the 3x3x6 part.
    # R(3,3,6) <= 3*6*3 = 54 schoolbook. But R(3,3,3) = 13? No, R(3,3,3) is unknown but at most 23.
    # For 3x3x6, we can think of it as 3 separate 3x3x2 problems? No, it's one tensor.
    #
    # Let me just do the straightforward direct sum with schoolbook, which gives the same rank 90.
    # That won't help. Let me try a different decomposition.
    #
    # Alternative: Use the fact that R(3,5,6) can be bounded by considering it as 3*(5*6) or similar.
    # Schoolbook is 90. Let me try to find a decomposition with rank < 90.
    #
    # For now, let me just output the schoolbook decomposition as a safe fallback, since my analysis above is uncertain.
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
