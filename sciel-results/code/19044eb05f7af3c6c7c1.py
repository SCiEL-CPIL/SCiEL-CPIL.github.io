"""Program for matrix multiplication 2x3 * 3x6 using Kronecker product structure."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We exploit 6 = 3 * 2, writing the 3x6 matrix as a tensor structure.
    # The multiplication (2x3) * (3x6) -> (2x6) can be seen as:
    #   Let p = 3 * 2. We decompose the k-index (size p=6) as k = (k1, k2) with k1 in [0,3), k2 in [0,2).
    #   Let the j-index (size m=3) stay as j in [0,3).
    #   Let the i-index (size n=2) stay as i in [0,2).
    #   
    # The tensor T[i,j,k] = delta[i,j,k] (multiplication tensor) is rank-1 by definition for 
    # the full multiplication. But we want to factor using the Kronecker structure.
    #
    # Actually, let me think more carefully. The matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has entries T[i, j, k] = 1 if this corresponds to a valid multiplication term.
    # In standard form, the multiplication C = A * B where C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,j,k] which is 1 for all i,j,k in range. Wait, no.
    #
    # Let me re-read the task. The tensor decomposition is for the multiplication itself.
    # U has shape (n*m, rank) = (6, rank), V has shape (m*p, rank) = (18, rank), W has shape (p*n, rank) = (12, rank).
    #
    # The convention is:
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # 
    # The tensor T for matrix multiplication is such that:
    # For each term c in the rank decomposition:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should reconstruct the multiplication.
    #
    # The multiplication tensor has T[i, j, k] = 1 for the "identity" structure... 
    # Actually, the matrix multiplication tensor is:
    # T[i, j, k] = 1 (for the standard basis), representing C[i,k] += A[i,j] * B[j,k].
    # 
    # So the tensor is T[i,j,k] = 1 for all valid i, j, k, and the contraction is:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    #
    # In a rank-r decomposition, we write:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has index (i,j), V has index (j,k), W has index (k,i).
    #
    # The schoolbook decomposition uses r = n*m*p terms, each being a rank-1 tensor.
    #
    # Now, with p = 6 = 3*2, let's write k = (a, b) where a in [0,3) and b in [0,2).
    # Then the multiplication becomes:
    # C[i, (a,b)] = sum_j A[i,j] * B[j, (a,b)]
    #
    # We can view this as two separate multiplications if we split appropriately,
    # but the structure is more like: for each (a,b), we're computing a 2x1 output
    # from a 2x3 matrix times a 3x1 column.
    #
    # Let me try a different approach. Since p=6=3*2, consider the 3x6 matrix B as
    # having entries B[j, (a,b)] where j in [0,3), a in [0,3), b in [0,2).
    # 
    # We can write B as a tensor B[j,a,b]. Then:
    # C[i,(a,b)] = sum_j A[i,j] * B[j,a,b]
    #
    # Now, if we could factor B[j,a,b] using a Kronecker structure... but B is just a matrix.
    #
    # Let me think about this differently. The key insight is that we can use the
    # bilinear structure. Consider splitting the problem:
    # 
    # Since p = 6 = 3 * 2, and m = 3, we have:
    # C[i, k] = sum_j A[i,j] B[j,k] where k ranges over 6 values.
    #
    # Write k = 3*b + a where a in [0,3), b in [0,2). Or k = a + 3*b.
    # Actually, the exact ordering matters for the W index. W[k*n + i, c].
    # 
    # Let me just try to construct a rank-12 decomposition directly using the structure.
    #
    # Idea: Split j into j' in [0,3) and use the fact that m=3 and p=6.
    # For each fixed b in [0,2), we have a 2x3 * 3x3 sub-problem (taking columns 3b..3b+2 of B).
    # Each 2x3 * 3x3 sub-problem has a known minimal rank. But we can't just add ranks because
    # the U matrix must be the same across all b's.
    #
    # Alternative: Use the Kronecker product structure explicitly.
    # Write the multiplication as a Kronecker product of two smaller multiplications.
    # 
    # Actually, let me reconsider. The rank of 2x3 * 3x6 multiplication:
    # We can write this as: (2x3 * 3x3) stacked twice in the column direction.
    # 
    # Let me try rank = 12 = 2 * 6 or 3 * 4 or 2 * 2 * 3.
    #
    # Let's try a direct construction. Consider:
    # C[i, 3*b + a] = sum_j A[i,j] B[j, 3*b+a] for i in [0,2), a in [0,3), b in [0,2).
    #
    # Define for each b, a 2x3 * 3x3 multiplication. If each uses rank 6, total would be 12
    # but we need to share the U factor.
    #
    # Let me try to construct it with rank 12.
    #
    # For b=0: C[i,a] = sum_j A[i,j] B[j,a] for a=0,1,2
    # For b=1: C[i,3+a] = sum_j A[i,j] B[j,3+a] for a=0,1,2
    #
    # I'll use a Kronecker-like structure. Let me define:
    # For each (j, a, b) combination, but that gives 3*3*2=18 which is worse.
    #
    # Let me try: rank = n * m * 2 = 2 * 3 * 2 = 12.
    # 
    # The idea: for each j in [0,3) and each "column group" b in [0,2), we create one term.
    # But we need to handle all a values too.
    #
    # Actually, let me just verify: can we do rank 12?
    # The tensor has 2*3*6 = 36 entries, all 1. We need:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    #
    # Let me try a structured decomposition.
    # 
    # Define for c = 0, ..., 11:
    #   Let c = 3*j + 2*b + a - a (this doesn't work).
    #
    # Let me try: c = j * 4 + b * 3 + a - ... no.
    #
    # Simpler: Let's use c = i * 6 + j for one set... no that's 12 but doesn't cover k properly.
    #
    # OK let me just brute-force a structured rank-12 decomposition.
    #
    # For c in range(12):
    #   c = 3*b + j  where b in [0,2), j in [0,3)
    #   Then U[i,m+j,c] = 1 for all i (since we sum over i separately... no)
    #
    # Let me think step by step. We need:
    # sum_c U[i,m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  for all i in {0,1}, j in {0,1,2}, k in {0,...,5}
    #
    # Try: U[i,m+j,c] = 1 if c corresponds to j (i.e., c mod 3 == j or c div 4 == j, etc.)
    #
    # Let me try: c = j * 4 + b where b in [0,4). Then for k = 3*b' + a where a in [0,3), b' in [0,2):
    #   We need V[j, 3*b'+a, c] * W[3*b'+a, i, c] to contribute.
    #
    # This is getting complex. Let me just code up a rank-12 attempt and verify.
    #
    # Strategy: For each j in {0,1,2} and each b in {0,1}, create a term.
    # c = 2*j + b, so j = c // 2, b = c % 2.
    # 
    # U[i*m + j, c]: set to 1 when j == c // 2, for all i. So U[i*3 + c//2, c] = 1.
    # 
    # Now we need V[j*p + k, c] * W[k*n + i, c] summed over c to give 1.
    # For fixed j and k = 3*b' + a:
    #   sum over c where c//2 == j of V[j*6+k, c] * W[k*2+i, c] = 1
    #   The c values are c = 2*j and c = 2*j+1, i.e., b = 0 and b = 1.
    #
    # So V[j*6+k, 2*j] * W[k*2+i, 2*j] + V[j*6+k, 2*j+1] * W[k*2+i, 2*j+1] = 1
    #
    # Let k = 3*b' + a. Set:
    #   V[j*6 + k, 2*j + b] = 1 if b == b', else 0
    #   W[k*2 + i, 2*j + b] = 1 for all i, j
    #
    # Then the sum becomes: V[j*6+k, 2*j+b'] * W[k*2+i, 2*j+b'] = 1 * 1 = 1. Good!
    # But wait, W[k*2+i, 2*j+b] = 1 for all i, j, b? That means W doesn't depend on i or j properly.
    #
    # Let's verify: W index is k*n + i = k*2 + i. For term c = 2*j+b:
    #   W[k*2 + i, c] should be 1 for all i in {0,1}, all k, all j, b.
    #   But c = 2*j+b determines j and b. So for a given c, j = c//2, b = c%2.
    #   W[k*2 + i, c] = 1 for all i, k.
    #
    # And V[j*6 + k, c] = 1 if (k // 3) == b where b = c % 2, else 0.
    #
    # Let me verify the full product:
    # sum_c U[i*3 + j, c] * V[j*6 + k, c] * W[k*2 + i, c]
    # = sum over c where U[i*3+j, c] = 1, which is c = 2*j and c = 2*j+1
    # = U[i*3+j, 2*j] * V[j*6+k, 2*j] * W[k*2+i, 2*j] + U[i*3+j, 2*j+1] * V[j*6+k, 2*j+1] * W[k*2+i, 2*j+1]
    # = 1 * V[j*6+k, 2*j] * 1 + 1 * V[j*6+k, 2*j+1] * 1
    # = V[j*6+k, 2*j] + V[j*6+k, 2*j+1]
    #
    # With k = 3*b' + a:
    #   V[j*6+k, 2*j] = 1 if b' == 0, else 0
    #   V[j*6+k, 2*j+1] = 1 if b' == 1, else 0
    # So the sum = 1 (exactly one of b' is 0 or 1). 
    #
    # This works! Rank 12.
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 18 x 12
    W = [[0] * rank for _ in range(p * n)]  # 12 x 12
    
    for c in range(rank):
        j = c // 2
        b = c % 2
        # U: U[i*m + j, c] = 1 for all i
        for i in range(n):
            U[i * m + j][c] = 1
        # V: V[j*p + k, c] = 1 if (k // 3) == b
        for k in range(p):
            if k // 3 == b:
                V[j * p + k][c] = 1
        # W: W[k*n + i, c] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
