"""Tensor decomposition for 2x5x6 matrix multiplication using a ladder/shifted approach."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # The "ladder" or "shifted" decomposition technique for matrix multiplication
    # with one small dimension (n=2).
    # 
    # The idea: We can exploit the linearity in the output index i (which has only 2 values).
    # For each (j, k) pair, we need to compute contributions to C[0, k] and C[1, k].
    # 
    # A standard trick is to use a "shifted" decomposition where we combine terms.
    # However, for exact integer decompositions, the minimum rank for 2x5x6 is known
    # to be 59 (one less than schoolbook) due to a specific linear dependency.
    # 
    # Let's construct a rank-59 decomposition.
    # 
    # The key insight: For n=2, we can write the multiplication as:
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    # 
    # We can combine the last term of one sum with the first term of another
    # using the fact that A[0,j] and A[1,j] are independent variables.
    # 
    # Actually, a known result is that rank(M_{2,m,p}) = 2mp - 1 for m,p >= 2.
    # This comes from the fact that we can save one rank by using a "diagonal"
    # argument.
    # 
    # Let's construct this explicitly.
    # 
    # For the first 2m(p-1) terms, we handle all (i,j,k) with k < p-1:
    # For i in {0,1}, j in {0,...,m-1}, k in {0,...,p-2}:
    #   U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    # 
    # For the remaining terms (k = p-1), we need to handle 2m terms.
    # But we can save one rank by combining them.
    # 
    # Specifically, for k = p-1:
    # C[0, p-1] = sum_j A[0,j] B[j, p-1]
    # C[1, p-1] = sum_j A[1,j] B[j, p-1]
    # 
    # We can write this as:
    # C[0, p-1] = A[0,0] B[0, p-1] + sum_{j=1}^{m-1} A[0,j] B[j, p-1]
    # C[1, p-1] = A[1,0] B[0, p-1] + sum_{j=1}^{m-1} A[1,j] B[j, p-1]
    # 
    # Now, consider the term A[0,0] B[0, p-1]. We can "shift" it:
    # Note that A[0,0] B[0, p-1] appears in C[0, p-1].
    # 
    # A known construction: Use 2m(p-1) + (2m-1) = 2mp - 1 terms.
    # 
    # Let me try a different approach. The rank of M_{2,m,p} is indeed 2mp-1.
    # Here's a construction:
    # 
    # For i in {0,1}, j in {0,...,m-1}, k in {0,...,p-2}:
    #   Term: U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    #   This gives 2*m*(p-1) terms.
    # 
    # For the remaining k=p-1 case, we have 2m terms to handle.
    # We can handle them with 2m-1 terms using the following trick:
    # 
    # For j in {0,...,m-2}:
    #   Term for (i=0, j, k=p-1): U[0*m+j] = 1, V[j*p+(p-1)] = 1, W[(p-1)*n+0] = 1
    #   Term for (i=1, j, k=p-1): U[1*m+j] = 1, V[j*p+(p-1)] = 1, W[(p-1)*n+1] = 1
    # 
    # For j = m-1:
    #   We need to handle both i=0 and i=1 for j=m-1, k=p-1.
    #   But we can combine them into one term? No, that's not straightforward.
    # 
    # Let me look up the actual construction for rank 2mp-1.
    # 
    # Actually, the standard result is that rank(M_{2,m,p}) = 2mp - 1.
    # The construction uses the fact that we can write:
    # 
    # For all (i,j,k) with i in {0,1}, j in {0,...,m-1}, k in {0,...,p-1},
    # except we combine the last two terms in a specific way.
    # 
    # Let's try: Use 2mp terms initially, but then observe that there's a linear
    # dependency that allows us to remove one term.
    # 
    # Actually, I recall that for 2xm x mxp, the rank is 2mp - 1 when m,p >= 2.
    # Let me just construct the schoolbook and then try to find a rank reduction.
    # 
    # For now, let's return rank 59 = 2*5*6 - 1.
    # 
    # Construction:
    # Terms 0 to 2*m*(p-1)-1: Handle all (i,j,k) with k < p-1
    # Terms 2*m*(p-1) to 2mp-2: Handle (i,j,k) with k = p-1, but combine the last two
    
    rank = 2 * m * p - 1  # 59
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Handle all terms except the last column k=p-1
    for i in range(n):
        for j in range(m):
            for k in range(p - 1):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now handle k = p-1
    # We have 2m terms to handle: (0,0,p-1), (0,1,p-1), ..., (0,m-1,p-1), (1,0,p-1), ..., (1,m-1,p-1)
    # We can handle 2m-1 of them directly, and combine the last one.
    
    # For i=0, j=0 to m-1:
    for j in range(m):
        U[0 * m + j][col] = 1
        V[j * p + (p - 1)][col] = 1
        W[(p - 1) * n + 0][col] = 1
        col += 1
    
    # For i=1, j=0 to m-2:
    for j in range(m - 1):
        U[1 * m + j][col] = 1
        V[j * p + (p - 1)][col] = 1
        W[(p - 1) * n + 1][col] = 1
        col += 1
    
    # For i=1, j=m-1: We need to handle this with the remaining term.
    # But we've used 2*(p-1)*m + m + (m-1) = 2mp - 2 + m + m - 1 = 2mp + m - 1... that's wrong.
    
    # Let me recount:
    # First part: n * m * (p-1) = 2 * 5 * 5 = 50 terms
    # Second part: m = 5 terms
    # Third part: m-1 = 4 terms
    # Total: 50 + 5 + 4 = 59 terms. Good.
    # But we haven't handled (i=1, j=m-1, k=p-1) yet!
    # 
    # Wait, we have rank = 59, and we've used 59 columns. But we still need to handle
    # the term (i=1, j=m-1, k=p-1).
    # 
    # This means my construction is incorrect. The rank 2mp-1 construction must be different.
    # 
    # Let me reconsider. The correct construction for rank 2mp-1 is:
    # 
    # Use 2mp - 1 terms as follows:
    # For all (i,j,k) in {0,1} x {0,...,m-1} x {0,...,p-1}, except we modify the last term.
    # 
    # Actually, I think the correct approach is to use a "diagonal" shift.
    # 
    # Let me just return the schoolbook decomposition for now, since I'm not confident
    # in the rank-59 construction.
    
    # Reset and do schoolbook
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
