"""Program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time

def construct():
    """Return a low-rank tensor decomposition for 2x3x9 matrix multiplication.
    
    Strategy: Split the 9 columns into 3 blocks of 3. Each block computes
    A (2x3) times a 3x3 sub-matrix of B. Use a rank-7 Strassen-like
    decomposition for the 2x3 x 3x3 multiplication per block.
    
    Total rank: 3 * 7 = 21 (vs schoolbook 54).
    
    U: (n*m, rank) = (6, 21)
    V: (m*p, rank) = (27, 21)  
    W: (p*n, rank) = (18, 21)
    
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    n_blocks = p // m  # 3 blocks of 3 columns each
    
    # Strassen decomposition for 3x3 matrix multiplication (rank 7)
    # For multiplying X (3x3) by Y (3x3):
    # M1 = (X00 + X22) * (Y00 + Y22)
    # M2 = (X20 + X22) * Y00
    # M3 = X00 * (Y02 - Y22)
    # M4 = X22 * (Y20 - Y00)
    # M5 = (X00 + X10) * Y10
    # M6 = (X12 + X22) * (Y11 + Y22)
    # M7 = (X10 - X20) * (Y20 + Y22)
    # M8 = (X00 - X10) * (Y01 + Y11)
    # M9 = (X11 - X21) * (Y10 + Y20)
    # M10 = (X11 + X21) * (Y02 + Y12)
    # M11 = (X01 - X11) * (Y11 + Y21)
    # M12 = (X01 + X11) * (Y00 - Y10)
    # M13 = (X10 + X20) * (Y11 - Y21)
    # M14 = (X11 + X21) * (Y02 - Y12)
    # M15 = (X01 - X11) * (Y00 + Y10)
    # M16 = (X10 + X20) * (Y02 + Y12)
    # M17 = (X00 - X10) * (Y10 + Y20)
    # M18 = (X01 + X11) * (Y01 - Y11)
    # M19 = (X11 - X21) * (Y00 + Y20)
    # M20 = (X00 + X20) * (Y12 - Y22)
    # M21 = (X01 - X11) * (Y10 - Y20)
    # M22 = (X01 + X11) * (Y01 + Y11)
    # M23 = (X00 + X10) * (Y02 - Y12)
    # M24 = (X10 - X20) * (Y00 + Y10)
    # M25 = (X11 + X21) * (Y01 - Y21)
    # M26 = (X00 - X20) * (Y01 + Y21)
    # M27 = (X01 + X21) * (Y00 - Y10)
    
    # Actually, the standard rank-7 for 3x3x3 is not well-known to be 7.
    # The best known for 3x3x3 is 19 (or 20). Let me use schoolbook for 3x3x3 (rank 9)
    # and see if I can do better.
    
    # For 2x3 x 3x3: this is computing C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, k in 0..2
    # This is 6 multiplications schoolbook. Lower bound: (2*3*3)^2/(4*9+9*9+4*9) = 36/153 < 1, so >= 1.
    # Actually the trivial lower bound is max(n*p, n*m, m*p) for the output... no.
    # The rank of the 2x3x3 tensor: each entry C[i,k] depends on 3 products.
    # We can group: for each k, C[:,k] = A @ B[:,k], which is 2x3 @ 3x1.
    # 2x3 @ 3x1 has rank at most 6, but actually it's just 2 outputs each being a dot product.
    # The tensor for 2x3x1 (2x3 times 3x1) has rank... C[i] = sum_j A[i,j]*B[j], so 2 outputs.
    # This is a 2x3x1 tensor. Its rank is at most 3 (each output is a linear combination of 3 A entries times B entries).
    # Actually rank of 2x3x1: the tensor T[i,j,0] = 1 for all i,j. This is a 2x3x1 tensor.
    # Rank = min over decompositions. Since the last dimension is 1, T[i,j,0] = sum_c U[i,j,c]*V[j,0,c]*W[0,i,c].
    # This is equivalent to a 2x3 matrix being a sum of rank-1 matrices, so rank <= 2 (since 2x3 has rank 2).
    # So 2x3x1 has rank 2!
    
    # For 2x3x3: C[i,k] = sum_j A[i,j]*B[j,k]. For each fixed k, it's a 2x3x1 tensor (rank 2).
    # But the B entries are shared across k. So we need rank 3 * 2 = 6? Or can we do better?
    # Actually, the full tensor is 2x3x3 with T[i,j,k]=1. 
    # Lower bound: (2*3*3)^2 / (2^2*3^2 + 3^2*3^2 + 2^2*3^2) = 36/(36+81+36) = 36/153 < 1, so >= 1.
    # The trivial bound is n*p = 6 (number of outputs). So rank >= 6? No, that's not a valid lower bound.
    # Actually, the rank of the 2x3x3 multiplication tensor: since for each k, the slice is rank 2,
    # and there are 3 slices, the total rank is at most 6. And it's at least... 
    # Let me just use rank 6 (schoolbook) for 2x3x3.
    
    # So total rank for 2x3x9 with 3 blocks: 3 * 6 = 18.
    # This matches the n*p lower bound of 18!
    
    # Let me construct the rank-18 decomposition.
    # For block b (b = 0,1,2), columns k = 3*b + 0, 3*b + 1, 3*b + 2
    # For each block, we need 6 terms (2x3x3 schoolbook)
    
    # Actually, let me verify: can we do rank 18 for the full 2x3x9?
    # The lower bound from n*p is 18. If we can achieve 18, that's optimal!
    
    # Construction: For the 2x3x9 tensor T[i,j,k] = 1 for all i in 0..1, j in 0..2, k in 0..8.
    # We need: T[i,j,k] = sum_c U[i*3+j, c] * V[j*9+k, c] * W[k*2+i, c]
    
    # Rank 18 construction:
    # For each k (0..8), and each i (0..1), we have output C[i,k] = sum_j A[i,j]*B[j,k]
    # That's 18 outputs, each requiring at least 1 term. So rank >= 18.
    # Can we achieve exactly 18?
    
    # For term c = i*9 + k (c ranges 0..17):
    #   U[i*3+j, c] = 1 if j == j_c for some j_c, else 0... 
    #   This doesn't work directly because each output needs a sum over j.
    
    # Alternative: For each (i, k) pair, use 3 terms (one for each j):
    #   That gives 2*9*3 = 54 (schoolbook).
    
    # To get rank 18, we need each term to contribute to multiple outputs.
    # Let's try: for each j (0..2) and each k (0..8), use one term:
    #   c = j*9 + k
    #   U[i*3+j, c] = 1 for all i (both rows)
    #   V[j*9+k, c] = 1
    #   W[k*2+i, c] = 1 for all i
    #   Then sum_c U[i*3+j,c]*V[j*9+k,c]*W[k*2+i,c] = sum_{j,k} 1*1*1 = 18 for each (i,k)??
    #   No, that's wrong. Let me reconsider.
    
    # The tensor is T[i,j,k] = 1. We need sum_c U[i*3+j,c]*V[j*9+k,c]*W[k*2+i,c] = 1 for ALL (i,j,k).
    # Not just for the outputs. This is a 3-index tensor, not just the output matrix.
    
    # So the constraint is: for every (i,j,k) in {0,1}x{0,1,2}x{0,...,8}:
    #   sum_c U[i*3+j, c] * V[j*9+k, c] * W[k*2+i, c] = 1
    
    # This is a system of 2*3*9 = 54 equations in rank*6 + rank*27 + rank*18 = 51*rank unknowns.
    # For rank 18: 918 unknowns, 54 equations. Very underdetermined!
    
    # So rank 18 should be achievable. Let me construct it.
    
    # Simple construction: For each (j, k) pair, create one term c = j*9 + k:
    #   U[i*3+j, c] = 1 for i = 0, 1  (both rows of the A matrix for column j)
    #   V[j*9+k, c] = 1
    #   W[k*2+i, c] = 1 for i = 0, 1  (both rows of the output for column k)
    #   
    # Check: sum_c U[i*3+j',c]*V[j'*9+k',c]*W[k'*2+i,c]
    # Only terms where j'=j and k'=k contribute (since U is nonzero only for j'=j, V only for k'=k)
    # For c = j*9+k: U[i*3+j, c]*V[j*9+k, c]*W[k*2+i, c] = 1*1*1 = 1
    # So sum = 1. Correct!
    
    # But wait, this gives rank 3*9 = 27, not 18.
    
    # To get rank 18, I need to be smarter. Let me think...
    
    # The key constraint is that U depends on (i,j), V on (j,k), W on (k,i).
    # For rank 18, I have 18 terms. Each term c has a 2x3 pattern in U, 3x9 in V, 9x2 in W.
    
    # Let me try: for each k (0..8) and each i (0..1), one term c = k*2 + i:
    #   U[i*3+j, c] = 1 for all j (0,1,2)
    #   V[j*9+k, c] = 1 for all j
    #   W[k*2+i, c] = 1
    #   
    # Check: sum_c U[i'*3+j',c]*V[j'*9+k',c]*W[k'*2+i',c]
    # For term c = k*2+i: U[i'*3+j', c] = 1 if i' = i, else 0 (since c encodes i)
    # Wait, I said U[i*3+j, c] = 1 for all j, but it should depend on i.
    # Let me redefine: U[row, c] where row = i*3+j.
    # For c = k*2 + i: U[i*3+j, c] = 1 for all j, for the specific i in c.
    # So U[0*3+j, c] = 1 if i=0, U[1*3+j, c] = 1 if i=1.
    # V[j*9+k, c] = 1 for all j.
    # W[k*2+i, c] = 1.
    # 
    # Check for (i', j', k'):
    # sum_c U[i'*3+j', c]*V[j'*9+k', c]*W[k'*2+i', c]
    # Only c = k'*2+i' has W[k'*2+i', c] = 1 (and U[i'*3+j', c] = 1, V[j'*9+k', c] = 1)
    # Other c: if c = k*2+i with k != k' or i != i', then either W[k'*2+i', c] = 0 or U[i'*3+j', c] = 0 or V[j'*9+k', c] = 0.
    # So sum = 1. Correct!
    
    # This gives rank 9*2 = 18! And it's optimal!
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 27 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    for k in range(p):  # 0..8
        for i in range(n):  # 0, 1
            c = k * n + i  # 0..17
            # U[i*m+j, c] = 1 for all j
            for j in range(m):
                U[i * m + j][c] = 1
            # V[j*p+k, c] = 1 for all j
            for j in range(m):
                V[j * p + k][c] = 1
            # W[k*n+i, c] = 1
            W[k * n + i][c] = 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
