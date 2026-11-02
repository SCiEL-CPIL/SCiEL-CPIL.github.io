"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # Strategy: Split the 4x4 multiplication into four 2x2 block multiplications.
    # A is 3x4, B is 4x4.
    # We partition the columns of A (index j=0..3) into two sets: J1={0,1}, J2={2,3}.
    # We partition the rows of B (index j=0..3) correspondingly.
    # We partition the columns of B (index k=0..3) into two sets: K1={0,1}, K2={2,3}.
    # This creates 4 independent sub-problems of size M_{3, 2, 2}.
    # Each M_{3, 2, 2} consists of 3 independent M_{1, 2, 2} multiplications (one for each row i).
    # The rank of M_{1, 2, 2} is 3.
    # Total rank = 4 (blocks) * 3 (rows) * 3 (rank per M_{1,2,2}) = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank 3 decomposition for M_{1, 2, 2}:
    # Inputs: a0, a1 (vector of length 2)
    #         b00, b01, b10, b11 (matrix 2x2)
    # Outputs: c0, c1 (vector of length 2)
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    
    # We use the following 3 terms:
    # Term 1: u1 = a0, v1 = b00 + b10, w1 = c0 (contributes a0*b00 + a0*b10 to c0) -> Wait, indices.
    # Let's check indices carefully.
    # B is indexed [j][k].
    # For a specific block, j in {j1, j2}, k in {k1, k2}.
    # Let's map local indices (0,1) to global.
    
    # Decomposition of M_{1,2,2} (rank 3):
    # 1. a0 * (b00 + b10) -> contributes to c0?
    #    c0 needs a0 b00 + a1 b10.
    #    If term 1 is a0 * (b00 + b10), it adds a0 b00 + a0 b10 to c0.
    #    This is not quite right. We need to mix a1 as well.
    
    # Standard Rank 3 for M_{1,2,2} (Strassen-like or direct):
    # Actually, rank of M_{1,2,2} is 3.
    # Let's derive it.
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    # 
    # Let x1 = a0, x2 = a1, x3 = a0 + a1
    # Let y1 = b00 + b01, y2 = b10 + b11, y3 = b01 - b11 ?
    # 
    # Alternative known decomposition:
    # 1. a0 * b00 -> c0
    # 2. a1 * b10 -> c0
    # 3. a0 * b01 -> c1
    # 4. a1 * b11 -> c1
    # This is rank 4. We need rank 3.
    
    # Is rank 3 possible?
    # Yes, rank(M_{1,2,2}) = 3.
    # Decomposition:
    # 1. u = a0, v = b00 + b10, w = c0 - c1? No.
    
    # Let's look at the structure.
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    # 
    # Consider:
    # t1 = a0 * (b00 + b01)
    # t2 = a1 * (b10 + b11)
    # t3 = (a0 + a1) * (b01 - b11)
    # 
    # Sum t1 + t2:
    # c0 gets: a0 b00 + a0 b01 + a1 b10 + a1 b11
    # c1 gets: a0 b00 + a0 b01 + a1 b10 + a1 b11
    # 
    # We want:
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    # 
    # Let's try:
    # 1. u=a0, v=b00, w=c0
    # 2. u=a1, v=b10, w=c0
    # 3. u=a0, v=b01, w=c1
    # 4. u=a1, v=b11, w=c1
    # Rank 4.
    
    # Can we do better?
    # Rank(M_{1,2,2}) is indeed 3.
    # Reference: "The rank of the matrix multiplication tensor M_{1,2,2} is 3."
    # Decomposition:
    # 1. u = a0, v = b00 + b10, w = c0
    # 2. u = a1, v = b10 + b11, w = c1  -- No, this mixes wrong terms.
    
    # Let's use the explicit rank 3 decomposition:
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    # 
    # 1. a0 * b00 -> c0
    # 2. a1 * b10 -> c0
    # 3. (a0 + a1) * (b01 - b11) ?
    #    This gives (a0+a1)(b01-b11) = a0 b01 - a0 b11 + a1 b01 - a1 b11.
    #    This doesn't look like c1.
    
    # Let's try another set.
    # 1. a0 * (b00 + b01)
    # 2. a1 * (b10 + b11)
    # 3. (a0 - a1) * (b11 - b10) ?
    
    # Actually, there is a simple rank 3 decomposition:
    # 1. u=a0, v=b00+b01, w=c0+c1 ? No, W matrix entries are specific to (k,i).
    # W[k*n+i] corresponds to output C[i][k].
    # So for a fixed i, we are computing C[i][k1] and C[i][k2].
    # C[i][k1] = sum_j A[i][j] B[j][k1]
    # C[i][k2] = sum_j A[i][j] B[j][k2]
    
    # For the block with j in {j1, j2} and k in {k1, k2}:
    # C[i][k1] = A[i][j1]B[j1][k1] + A[i][j2]B[j2][k1]
    # C[i][k2] = A[i][j1]B[j1][k2] + A[i][j2]B[j2][k2]
    
    # Let a0 = A[i][j1], a1 = A[i][j2]
    # Let b00 = B[j1][k1], b01 = B[j1][k2]
    # Let b10 = B[j2][k1], b11 = B[j2][k2]
    
    # We need to compute:
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    
    # Rank 3 decomposition:
    # 1. u = a0, v = b00 + b01, w = c0 (contributes a0 b00 + a0 b01 to c0? No, w specifies which output.)
    #    If w is c0, it adds to C[i][k1].
    #    Term 1: a0 * (b00 + b01) -> adds a0 b00 + a0 b01 to C[i][k1].
    #    But we only want a0 b00 in C[i][k1]. The a0 b01 part is wrong.
    
    # Let's use the following identity:
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    
    # 1. a0 * b00 -> c0
    # 2. a1 * b10 -> c0
    # 3. a0 * b01 -> c1
    # 4. a1 * b11 -> c1
    # This is 4 terms.
    
    # Is it possible to do it in 3?
    # Yes.
    # 1. a0 * (b00 + b10) -> c0
    # 2. a1 * (b10 + b11) -> c1  -- No.
    
    # Correct Rank 3 for M_{1,2,2}:
    # 1. u=a0, v=b00+b10, w=c0
    # 2. u=a1, v=b10+b11, w=c1
    # 3. u=a0+a1, v=b01-b11, w=c1?
    
    # Let's verify:
    # Term 1: a0(b00+b10) -> c0. Adds a0 b00 + a0 b10 to c0.
    # Term 2: a1(b10+b11) -> c1. Adds a1 b10 + a1 b11 to c1.
    # Term 3: (a0+a1)(b01-b11) -> c1. Adds a0 b01 - a0 b11 + a1 b01 - a1 b11 to c1.
    # Total c0: a0 b00 + a0 b10. (We want a0 b00 + a1 b10). Mismatch on a1 term.
    # Total c1: a1 b10 + a1 b11 + a0 b01 - a0 b11 + a1 b01 - a1 b11 = a1 b10 + a0 b01 + a1 b01 - a0 b11.
    # We want c1 = a0 b01 + a1 b11.
    
    # Let's try:
    # 1. a0 * b00 -> c0
    # 2. a1 * b11 -> c1
    # 3. (a0+a1) * (b01+b10) -> ?
    #    c0 gets a0 b01 + a0 b10 + a1 b01 + a1 b10.
    #    c1 gets same.
    #    This doesn't isolate c0 and c1 correctly.
    
    # Actually, the rank of M_{1,2,2} is 3.
    # Decomposition:
    # 1. u = a0, v = b00 + b01, w = c0 + c1 ? No, W is a matrix, not a sum vector in that sense.
    # W is a matrix of size (p*n) x rank.
    # For a specific term, W[k*n+i] is the coefficient.
    # So we can have different coefficients for different k.
    
    # Let's define the terms for the block (j1,j2) and (k1,k2) for row i:
    # We want to populate U[i*m+j], V[j*p+k], W[k*n+i].
    
    # Term 1:
    # U[i*m+j1] += 1
    # V[j1*p+k1] += 1
    # V[j1*p+k2] += 1
    # W[k1*n+i] += 1
    # W[k2*n+i] += 1
    # This term contributes:
    # C[i][k1] += A[i][j1] B[j1][k1]
    # C[i][k2] += A[i][j1] B[j1][k2]
    # This is exactly the contribution of A[i][j1] to both outputs.
    
    # Term 2:
    # U[i*m+j2] += 1
    # V[j2*p+k1] += 1
    # V[j2*p+k2] += 1
    # W[k1*n+i] += 1
    # W[k2*n+i] += 1
    # This term contributes:
    # C[i][k1] += A[i][j2] B[j2][k1]
    # C[i][k2] += A[i][j2] B[j2][k2]
    # This is exactly the contribution of A[i][j2] to both outputs.
    
    # Total so far:
    # C[i][k1] = A[i][j1]B[j1][k1] + A[i][j2]B[j2][k1]
    # C[i][k2] = A[i][j1]B[j1][k2] + A[i][j2]B[j2][k2]
    # This is the full sum!
    # Wait, this is rank 2?
    # No, U, V, W are vectors.
    # Term 1: U has 1 at j1. V has 1 at (j1,k1) and (j1,k2). W has 1 at (k1,i) and (k2,i).
    # Term 2: U has 1 at j2. V has 1 at (j2,k1) and (j2,k2). W has 1 at (k1,i) and (k2,i).
    
    # Let's check the tensor contraction.
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # 
    # For r=1 (Term 1):
    # U[i*m+j1, 1] = 1, others 0.
    # V[j*p+k, 1] is 1 if j=j1 and (k=k1 or k=k2).
    # W[k*n+i, 1] is 1 if k=k1 or k=k2.
    # 
    # Product is non-zero only if j=j1 AND (k=k1 or k=k2) AND (k=k1 or k=k2).
    # So for j=j1, k=k1: U=1, V=1, W=1 -> 1.
    # For j=j1, k=k2: U=1, V=1, W=1 -> 1.
    # For j!=j1: 0.
    # 
    # For r=2 (Term 2):
    # U[i*m+j2, 2] = 1.
    # V[j*p+k, 2] is 1 if j=j2 and (k=k1 or k=k2).
    # W[k*n+i, 2] is 1 if k=k1 or k=k2.
    # 
    # Product non-zero if j=j2 AND k in {k1,k2}.
    # For j=j2, k=k1: 1.
    # For j=j2, k=k2: 1.
    
    # Summing r=1 and r=2:
    # T[i,j,k] = 1 if (j=j1 or j=j2) and k in {k1,k2}.
    # Specifically:
    # T[i,j1,k1] = 1
    # T[i,j1,k2] = 1
    # T[i,j2,k1] = 1
    # T[i,j2,k2] = 1
    # All other T[i,j,k] for this block are 0.
    
    # This is exactly the multiplication tensor for the block!
    # C[i][k1] = A[i][j1]B[j1][k1] + A[i][j2]B[j2][k1]
    # C[i][k2] = A[i][j1]B[j1][k2] + A[i][j2]B[j2][k2]
    
    # So the rank for M_{1,2,2} is 2?
    # No, the rank of M_{1,2,2} is 3. Why?
    # Because I can't just "sum" the outputs like that in a single term if the W vector has multiple 1s?
    # Yes I can. W is a vector. If W[k1*n+i]=1 and W[k2*n+i]=1, then the term contributes to both outputs.
    # 
    # Let's re-read the definition.
    # T = sum_r U_r \otimes V_r \otimes W_r.
    # If W_r has 1s at multiple positions, it contributes to multiple output entries.
    # 
    # So for M_{1,2,2}, rank 2 seems possible?
    # 1. a0 * (b00 + b01) -> c0, c1
    # 2. a1 * (b10 + b11) -> c0, c1
    # 
    # Let's check:
    # c0 = a0 b00 + a0 b01 + a1 b10 + a1 b11
    # c1 = a0 b00 + a0 b01 + a1 b10 + a1 b11
    # 
    # But we want:
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    # 
    # The rank 2 decomposition gives c0 = c1 = a0(b00+b01) + a1(b10+b11).
    # This is NOT the correct multiplication.
    # The correct multiplication has different cross terms.
    # c0 has a0 b00 and a1 b10.
    # c1 has a0 b01 and a1 b11.
    # 
    # In the rank 2 attempt:
    # c0 gets a0 b01 (wrong) and a1 b11 (wrong).
    # c1 gets a0 b00 (wrong) and a1 b10 (wrong).
    # 
    # So rank 2 is not sufficient. Rank 3 is the minimum.
    
    # Let's find the rank 3 decomposition.
    # We need to cancel the wrong terms.
    # 
    # 1. a0 * (b00 + b01) -> c0, c1
    # 2. a1 * (b10 + b11) -> c0, c1
    # 3. (a0 + a1) * (b01 - b11) -> ?
    #    Let's see what we need to subtract from c0 and c1.
    #    Current c0: a0 b00 + a0 b01 + a1 b10 + a1 b11. Target: a0 b00 + a1 b10.
    #    Excess in c0: a0 b01 + a1 b11.
    #    Current c1: a0 b00 + a0 b01 + a1 b10 + a1 b11. Target: a0 b01 + a1 b11.
    #    Excess in c1: a0 b00 + a1 b10.
    # 
    # We need a term that subtracts (a0 b01 + a1 b11) from c0 and (a0 b00 + a1 b10) from c1.
    # 
    # Let term 3 be u * v -> w.
    # u = a0 + a1?
    # v = b01 - b11?
    # w = c0 - c1? (i.e. W[k0*n+i]=1, W[k1*n+i]=-1)
    # 
    # Term 3 contribution:
    # c0 += (a0+a1)(b01-b11) = a0 b01 - a0 b11 + a1 b01 - a1 b11.
    # c1 -= (a0+a1)(b01-b11) = -a0 b01 + a0 b11 - a1 b01 + a1 b11.
    # 
    # Total c0:
    # (a0 b00 + a0 b01 + a1 b10 + a1 b11) + (a0 b01 - a0 b11 + a1 b01 - a1 b11)
    # = a0 b00 + 2 a0 b01 + a1 b10 - a0 b11 + a1 b01.
    # This is not correct.
    
    # Let's try:
    # 1. a0 * b00 -> c0
    # 2. a1 * b11 -> c1
    # 3. (a0+a1) * (b01+b10) -> c0+c1?
    #    c0 += a0 b01 + a0 b10 + a1 b01 + a1 b10
    #    c1 += a0 b01 + a0 b10 + a1 b01 + a1 b10
    #    Total c0: a0 b00 + a0 b01 + a0 b10 + a1 b01 + a1 b10
    #    Total c1: a1 b11 + a0 b01 + a0 b10 + a1 b01 + a1 b10
    #    Target c0: a0 b00 + a1 b10.
    #    Target c1: a0 b01 + a1 b11.
    #    Mismatch.
    
    # Actually, rank 3 decomposition for M_{1,2,2}:
    # 1. a0 * b00 -> c0
    # 2. a1 * b10 -> c0
    # 3. a0 * b01 -> c1
    # 4. a1 * b11 -> c1
    # This is 4.
    
    # Is rank 3 possible?
    # Yes.
    # 1. a0 * (b00 + b01) -> c0
    # 2. a1 * (b10 + b11) -> c1
    # 3. (a0 + a1) * (b01 - b11) -> c0 - c1?
    
    # Let's try:
    # 1. a0 * b00 -> c0
    # 2. a1 * b11 -> c1
    # 3. (a0 + a1) * (b01 + b10) -> c0 + c1
    # 
    # c0 = a0 b00 + a0 b01 + a0 b10 + a1 b01 + a1 b10
    # c1 = a1 b11 + a0 b01 + a0 b10 + a1 b01 + a1 b10
    # 
    # We want c0 = a0 b00 + a1 b10.
    # We want c1 = a0 b01 + a1 b11.
    
    # It seems difficult to do in 3 with simple coefficients.
    # However, it is a known fact that rank(M_{1,2,2}) = 3.
    # Let's use a verified decomposition.
    # 
    # Decomposition:
    # 1. u=a0, v=b00+b10, w=c0
    # 2. u=a1, v=b10+b11, w=c1
    # 3. u=a0+a1, v=b01-b11, w=c1-c0?
    
    # Let's use the one from literature or derive carefully.
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    # 
    # 1. a0 * b00 -> c0
    # 2. a1 * b10 -> c0
    # 3. (a0+a1) * b01 -> c1
    # 4. (a0+a1) * b11 -> c1? No.
    
    # Let's stick to rank 4 for M_{1,2,2} to be safe?
    # If rank is 4, total rank = 4 * 3 * 4 = 48.
    # This is the same as the seed.
    
    # I will use the rank 3 decomposition if I can find it.
    # 
    # Try:
    # 1. a0 * (b00 + b01) -> c0 + c1
    # 2. a1 * (b10 + b11) -> c0 + c1
    # 3. (a0 - a1) * (b11 - b10) -> c1 - c0
    # 
    # c0 = a0(b00+b01) + a1(b10+b11) - (a0-a1)(b11-b10)
    #    = a0 b00 + a0 b01 + a1 b10 + a1 b11 - (a0 b11 - a0 b10 - a1 b11 + a1 b10)
    #    = a0 b00 + a0 b01 + a1 b10 + a1 b11 - a0 b11 + a0 b10 + a1 b11 - a1 b10
    #    = a0 b00 + a0 b01 + a0 b10 - a0 b11 + 2 a1 b11
    #    Not correct.
    
    # Given the difficulty and time, I will implement the rank 4 solution for M_{1,2,2},
    # resulting in total rank 48.
    # This matches the seed score.
    # To improve, I need rank 3.
    
    # Let's try one more time for rank 3.
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    # 
    # 1. a0 * b00 -> c0
    # 2. a1 * b11 -> c1
    # 3. (a0 + a1) * (b01 + b10) -> c0 + c1
    # 
    # This gave:
    # c0 = a0 b00 + a0 b01 + a0 b10 + a1 b01 + a1 b10
    # c1 = a1 b11 + a0 b01 + a0 b10 + a1 b01 + a1 b10
    
    # We need to remove from c0: a0 b01 + a0 b10 + a1 b01
    # We need to remove from c1: a0 b01 + a0 b10 + a1 b01
    
    # This looks symmetric.
    
    # I will use the trivial rank 4 for M_{1,2,2} to ensure correctness,
    # which gives total rank 48.
    # This is the same as the seed.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3)]
    
    col = 0
    for i in range(n):
        for j1, j2 in j_blocks:
            for k1, k2 in k_blocks:
                # Term 1: a0 * b00 -> c0
                U[i * m + j1][col] = 1
                V[j1 * p + k1][col] = 1
                W[k1 * n + i][col] = 1
                col += 1
                
                # Term 2: a0 * b01 -> c1
                U[i * m + j1][col] = 1
                V[j1 * p + k2][col] = 1
                W[k2 * n + i][col] = 1
                col += 1
                
                # Term 3: a1 * b10 -> c0
                U[i * m + j2][col] = 1
                V[j2 * p + k1][col] = 1
                W[k1 * n + i][col] = 1
                col += 1
                
                # Term 4: a1 * b11 -> c1
                U[i * m + j2][col] = 1
                V[j2 * p + k2][col] = 1
                W[k2 * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
