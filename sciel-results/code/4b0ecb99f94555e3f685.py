"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT over Z_2 x Z_7 for 2x4 x 4x7 -> 2x7 matrix multiplication.
    
    Strategy: Use the group G = Z_2 x Z_7 (order 14).
    - Embed A (2x4) into functions on Z_2 x Z_7 by zero-padding the 4-dim to 7-dim.
    - Embed B (4x7) into functions on Z_2 x Z_7.
    - The multiplication becomes a convolution, diagonalized by the Fourier transform.
    - Each Fourier mode (u,v) with u in {0,1}, v in {0,...,6} gives one rank-1 term.
    - Total rank = 14 (the order of the group), provided all characters are non-degenerate.
    
    For exact integer/half-integer entries, we use the fact that:
    - The Z_2 part contributes factors of +/-1.
    - The Z_7 part contributes 7th roots of unity.
    
    However, to get exact rational/half-integer entries, we need to handle the 
    complex roots carefully. The standard approach for WFT with prime p uses 
    the fact that the decomposition can be written with entries in Q(omega_p) 
    where omega_p is a primitive p-th root of unity. For exact decomposition 
    with rational entries, we typically need p to be such that the cyclotomic 
    field has degree 1 or 2, which is not the case for p=7 (degree 6).
    
    Alternative: Use a tensor decomposition that exploits the structure differently.
    
    Let's use a direct construction based on the observation that for n=2:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,...,6}
    
    We can use the following rank-28 decomposition (improving over 56):
    For each k in {0,...,6}, we have a 2x4 * 4x1 multiplication.
    This can be done with rank 4 (schoolbook) or potentially less.
    
    Actually, let's use a known efficient scheme. For 2xm x mxp, 
    the rank is at most 2*p if we use a certain Winograd-like scheme.
    
    Let me construct a rank-28 decomposition using the following idea:
    Split the 4 middle indices into pairs and use 2x2 Winograd (rank 5) 
    combined with the 7 output dimension.
    
    A = [A_col0 | A_col1 | A_col2 | A_col3] (each 2x1)
    B = [B_row0; B_row1; B_row2; B_row3] (each 1x7)
    
    C = A_col0 @ B_row0 + A_col1 @ B_row1 + A_col2 @ B_row2 + A_col3 @ B_row3
    
    Each outer product A_col_i @ B_row_i is rank 1, but we need to sum them.
    This doesn't directly give a tensor decomposition.
    
    Let's use a different approach: the standard rank-28 decomposition for 
    2x4 x 4x7 is achieved by treating it as two 2x2 x 2x7 multiplications 
    plus cross terms, or by using the fact that the border rank might be lower.
    
    For now, let's implement a clean rank-28 decomposition:
    For each output position (i,k) where i in {0,1}, k in {0,...,6},
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can decompose this as:
    For j in {0,1,2,3} and k in {0,...,6}:
    The term A[i,j]*B[j,k] contributes to C[i,k].
    
    A rank-28 decomposition:
    For each (j,k) pair (4*7=28 terms), define:
    U[i*m+j] gets the coefficient for A[i,j]
    V[j*p+k] gets the coefficient for B[j,k]  
    W[k*n+i] gets the coefficient for C[i,k]
    
    But this is just schoolbook with rank 28 (not 56) because we share the 
    (j,k) index across the two rows of A.
    
    Wait, schoolbook for 2x4 x 4x7 has rank 2*4*7 = 56.
    The rank-28 would require sharing terms.
    
    Let me think again. The multiplication tensor has indices (i,j,k) with 
    i in {0,1}, j in {0,1,2,3}, k in {0,...,6}.
    T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    
    Actually, the tensor is T_{(i,k),j} = 1 for all i,j,k, meaning 
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    A rank-1 decomposition term is u_i * v_j * w_k where:
    u is a vector in R^{n*m} = R^8
    v is a vector in R^{m*p} = R^{28}
    w is a vector in R^{p*n} = R^{14}
    
    The tensor entry T[(i,j), (j',k), (k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    Wait, that's not right. The tensor for matrix multiplication is:
    T[a,b,c] where a indexes A, b indexes B, c indexes C.
    T[a,b,c] = 1 if c = (i,k) and a = (i,j) and b = (j,k) for some consistent indices.
    
    More precisely, T[(i,j), (j,k), (i,k)] = 1, and 0 otherwise.
    So T is supported on triples where the j-index matches between A and B,
    and the i-index matches between A and C, and the k-index matches between B and C.
    
    A rank-1 decomposition: sum_r U[:,r] * V[:,r] * W[:,r] = T
    means T[(i,j),(j',k),(i',k')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    For schoolbook: for each (i,j,k), set U[i*m+j, r] = 1, V[j*p+k, r] = 1, 
    W[k*n+i, r] = 1 where r corresponds to (i,j,k). This gives rank 56.
    
    To get rank 28, we need to find a decomposition with 28 terms.
    One way: for each (j,k), define a term that handles both i=0 and i=1.
    
    Let's try: for each (j,k) in {0..3} x {0..6}, define term r = (j,k).
    U[0*m+j, r] = 1, U[1*m+j, r] = 1  (both rows of A use the same j)
    V[j*p+k, r] = 1
    W[k*n+0, r] = 1, W[k*n+1, r] = 1  (both columns of C use the same k)
    
    Then sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    = sum_{(j,k)} [delta_{j,j} * delta_{j',j} * delta_{k',k} * delta_{i',i}]
    Wait, this gives sum over all (j,k) where j=j' and k=k', so just one term,
    and the value is delta_{i,i'} (which is always 1 since we set both).
    
    That's not right. Let me be more careful.
    
    U[i*m+j, r_{j,k}] = 1 for all i, for each (j,k)
    V[j'*p+k, r_{j,k}] = delta_{j',j}
    W[k'*n+i', r_{j,k}] = delta_{k',k}
    
    Then the product is:
    sum_{(j,k)} 1 * delta_{j',j} * delta_{k',k} = 1 if j'=j and k'=k for some (j,k)
    But we sum over all (j,k), so if j' and k' are fixed, only the term (j'=j, k'=k) 
    contributes, giving 1.
    
    So T[(i,j),(j',k),(i',k')] = 1 for all i,i' when j=j' and k=k'.
    But we need T = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}.
    
    This gives 1 when j=j' and k=k', regardless of i,i'. That's wrong.
    
    I need to separate the i-index. Let me use:
    For each (i,j,k), that's 56 terms (schoolbook).
    
    To reduce rank, I need to find linear dependencies. The tensor has 
    a specific structure that might allow rank reduction.
    
    For 2x4 x 4x7, the known minimum rank is not well-established, but 
    it's at most 56 (schoolbook) and at least max(8,28,14) = 28.
    
    Let me just implement the schoolbook with rank 56 for now, as it's 
    guaranteed correct, and try to optimize later.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition with rank n*m*p = 56
    rank = n * m * p
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
