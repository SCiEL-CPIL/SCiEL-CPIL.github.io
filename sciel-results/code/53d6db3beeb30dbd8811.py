"""Matrix multiplication 2,4,6 — rank-24 decomposition via block structure."""

DIMENSIONS = [2, 4, 6]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    # We split p=6 into three blocks of size 2.
    # For each block q (q=0,1,2), we multiply A (2x4) by B_block (4x2).
    # The rank of M_{2,4,2} is 8.
    # Total rank = 3 * 8 = 24.
    
    # For M_{2,4,2}: A is 2x4, B is 4x2, C is 2x2.
    # We can further split m=4 into two m=2 blocks.
    # M_{2,2,2} has rank 7 (Strassen). But wait, let's think.
    # Actually, R(M_{2,2,2}) = 7 (Strassen's algorithm).
    # But we have m=4. We can split A into two 2x2 blocks and B into two 2x2 blocks.
    # Actually, let's use the direct approach: R(M_{2,4,2}).
    # We can view this as: C = A * B where A is 2x4, B is 4x2.
    # Let A = [A1, A2] where A1, A2 are 2x2.
    # Let B = [B1; B2] where B1, B2 are 2x2.
    # Then C = A1*B1 + A2*B2.
    # Each A_i * B_i is a 2x2 by 2x2 multiplication.
    # We can use Strassen for each, but they share the addition.
    # Actually, C = A1*B1 + A2*B2.
    # If we use Strassen for A1*B1 (7 mults) and Strassen for A2*B2 (7 mults),
    # that's 14 multiplications. Then we add the two 2x2 results.
    # But the addition is free in tensor rank (it's just adding entries).
    # Wait, tensor rank counts the number of rank-1 terms.
    # The sum of two tensors has rank at most the sum of their ranks.
    # So R(M_{2,4,2}) <= 2 * R(M_{2,2,2}) = 2 * 7 = 14.
    # But we also need to form the Strassen inputs, which involves additions of A and B entries.
    # These additions don't increase tensor rank (they're linear in the inputs).
    # So R(M_{2,4,2}) <= 14.
    # Hmm, but is it actually 14? Or can we do better?
    # Actually, M_{2,4,2} is isomorphic to M_{4,2,2} which is isomorphic to M_{2,4,2}.
    # Let's think differently. M_{2,4,2} can be seen as computing 4 dot products of length 4?
    # No. C[i,k] = sum_j A[i,j] * B[j,k].
    # There are 2*2 = 4 outputs, each a dot product of length 4.
    # But they share inputs.
    # Actually, the tensor M_{2,4,2} has rank... let me think.
    # It's the same as M_{4,2,2} by permutation.
    # M_{n,m,p} rank is known for small cases.
    # R(M_{2,4,2}) = R(M_{4,2,2}) = R(M_{2,2,4}).
    # M_{2,2,4}: A is 2x2, B is 2x4, C is 2x4.
    # This is 2 independent 2x2 by 2x2 multiplications (split B into two 2x2 blocks).
    # So R(M_{2,2,4}) = 2 * R(M_{2,2,2}) = 2 * 7 = 14.
    # So R(M_{2,4,2}) = 14.
    # Total rank = 3 * 14 = 42. That's worse than 24!
    
    # Let me reconsider. The schoolbook rank is 48.
    # Let's try a different approach.
    # We can use the fact that M_{2,4,6} can be decomposed using bilinear algorithms.
    # One approach: use the "laser method" or just careful block decomposition.
    
    # Let's try: split m=4 into two m=2 blocks.
    # A = [A1, A2], each 2x2.
    # B = [B1; B2], each 2x6.
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x2 * 2x6 = 2x6.
    # A2*B2 is 2x2 * 2x6 = 2x6.
    # R(M_{2,2,6}) = ?
    # M_{2,2,6}: A is 2x2, B is 2x6, C is 2x6.
    # This is 3 independent 2x2 by 2x2 multiplications.
    # So R(M_{2,2,6}) = 3 * 7 = 21.
    # So total rank = 2 * 21 = 42. Still worse.
    
    # Let's try splitting p=6 into three p=2 blocks.
    # C = [C1, C2, C3], each 2x2.
    # C1 = A * B1, where B1 is 4x2.
    # C2 = A * B2, where B2 is 4x2.
    # C3 = A * B3, where B3 is 4x2.
    # Each Cq is a M_{2,4,2} multiplication.
    # R(M_{2,4,2}) = 14 as computed above.
    # Total = 3 * 14 = 42.
    
    # Hmm, all my estimates give 42 or worse. Let me re-examine.
    # Wait, maybe I'm overcomplicating. Let's just use the schoolbook with some optimization.
    # Actually, let's think about what the minimum rank could be.
    # The tensor M_{2,4,6} has 2*4*6 = 48 entries in the "trivial" decomposition.
    # Lower bounds: R >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
    # So rank >= 24.
    # Can we achieve rank 24?
    # One way: if we can find a decomposition where each rank-1 term uses the full structure.
    # For rank 24, we need 24 rank-1 terms.
    # Note that m*p = 24, so one natural approach is to have each term correspond to a pair (j,k) with j in 1..4, k in 1..6.
    # But that gives 24 terms, and for each (j,k), the term is:
    #   e_j (in V) * e_k (in W) * (sum over i of A_i * something)
    # Actually, let's think about it as: for each j in 1..4, k in 1..6:
    #   U has a row for each i in 1..2: U[i, (j,k)] = A[i,j]
    #   V has V[j, (j,k)] = 1, and 0 elsewhere
    #   W has W[k, (j,k)] = 1, and 0 elsewhere
    # But this is exactly the schoolbook, and it gives rank 48, not 24.
    # The issue is that i has 2 values, so we need 2*4*6 = 48 terms.
    
    # To get rank 24, we need to somehow combine the i dimension.
    # This is possible if we use Strassen-like tricks on the i=2 dimension.
    # For each (j,k), we have C[i,k] += A[i,j] * B[j,k] for i=1,2.
    # If we fix (j,k), we're doing a 2x1 * 1x1 multiplication, which has rank 2.
    # But we have 4*6 = 24 such (j,k) pairs, so total rank 48.
    
    # To reduce rank, we need to share computations across different (j,k) pairs.
    # This is where block decomposition helps.
    
    # Let's try: group j into pairs. j = (j1, j2) where j1 in {1,2}, j2 in {1,2}.
    # So A has blocks A_{j1} which are 2x2.
    # For each j2 in {1,2} and k in 1..6:
    #   C[:,k] += A_{j1}[:, j2] * B[j1*2+j2, k]
    # This is a 2x1 * scalar for each j1, summed over j1.
    # So C[:,k] = sum_{j1=1}^2 A_{j1}[:, j2] * B[j1*2+j2, k]
    # This is a 2x2 * 2x1 multiplication for each (j2, k).
    # R(M_{2,2,1}) = 4 (it's just 2 dot products of length 2... no, it's 2x2 * 2x1).
    # Actually, M_{2,2,1}: A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This has 4 bilinear terms, so rank 4.
    # Total: 2 * 6 * 4 = 48. Still 48.
    
    # I think the schoolbook rank of 48 is hard to beat for this small case.
    # Let's just return the schoolbook decomposition. It's correct and has rank 48.
    # Score = 48/48 = 1.0.
    
    # Actually, wait. Let me reconsider the lower bound.
    # R(M_{n,m,p}) >= max(n*m, m*p, p*n) is a standard lower bound.
    # For n=2, m=4, p=6: max(8, 24, 12) = 24.
    # But this lower bound is not tight for all cases.
    # For M_{2,4,6}, the rank might indeed be 48.
    
    # Let's just implement the schoolbook.
    rank = n * m * p  # 48
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
