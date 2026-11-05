"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x6 matrix.
    
    We use the Kronecker product structure:
    The 3x3x6 multiplication tensor can be viewed as the tensor product of:
    - A 3x3x3 matrix multiplication tensor (rank 7 via Strassen-like)
    - A 1x3x2 row-vector extraction tensor (rank 2)
    
    However, a simpler approach: we can embed the 3x3x6 problem into
    larger known decompositions or use structured search.
    
    For 3x3x6, the schoolbook rank is 54. We try to find a better rank.
    
    Strategy: Use a structured bilinear form approach. The output C[i,k] = sum_j A[i,j]*B[j,k]
    for i in 0..2, j in 0..2, k in 0..5.
    
    We'll attempt to find a low-rank decomposition by exploiting the fact that
    the B matrix has 6 columns but only 3 rows, so we can group columns.
    
    Actually, let's think about this more carefully. The tensor T has indices (i,j,k)
    where i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4,5}, and T[i,j,k] = 1.
    
    We want to write T as a sum of rank-1 tensors:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank) with the
    trace convention W[k*n+i, r].
    
    Let's try rank 24 or so by grouping the k dimension.
    
    Actually, a key insight: we can use the fact that for fixed i, the row i of C
    is a 1x6 vector = A[i,:] (1x3) * B (3x6). This is a 1x3 times 3x6 multiplication.
    
    The rank of 1x3 times 3x6 is 3 (just dot product with each column). So for 3 rows,
    we might expect rank 9? But there's coupling through the shared A columns.
    
    Let me just do a direct structured construction.
    
    For each i (output row), C[i,k] = sum_j A[i,j] * B[j,k].
    
    We can write this as: for each i, define U_i[j,c] and V_j[k,c] such that
    sum_c U_i[j,c] V_j[k,c] = delta... no, that's not right.
    
    Let's use the standard approach: each term c contributes a rank-1 slice.
    Term c: U[:,c] is a vector in R^{n*m}, V[:,c] in R^{m*p}, W[:,c] in R^{p*n}.
    
    The contribution to T[i,j,k] from term c is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    One approach: for each i, we need a decomposition of the 3x6 matrix multiplication
    (A[i,:] * B) that shares structure across i.
    
    Let's try: for each i, use 3 terms (one per j), but that gives 9 terms total if independent.
    But we can share: if we use the same V and W patterns for all i, we might save.
    
    Actually, the cleanest low-rank approach: 
    For each j (0,1,2) and each i (0,1,2), we have a term that contributes to all k.
    That's 9 terms, each with:
      U[i*m+j, c] = 1
      V[j*p+k, c] = B[j,k]... no, V must be fixed.
    
    Wait, V and W are fixed matrices. Let me reconsider.
    
    Term c: pick a specific (i0, j0) for U and a specific k0 for W, then V[j0, k] = 1 for all k?
    No, V[j*p+k, c] is a single entry.
    
    Let me just use the schoolbook but try to compress. The schoolbook has 54 terms.
    Each term is identified by (i,j,k) with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    
    Can we merge terms? Two terms (i,j,k) and (i',j',k') can be merged if their
    U, V, W vectors are proportional. Since all entries are 0 or 1, we need exact matches.
    No merging possible in schoolbook.
    
    Let's try a different structure. Use 3x3 Strassen-like for the "core" and handle
    the extra 3 columns of B separately.
    
    Actually, 3x3x6 = 3x3x3 (for k=0,1,2) + 3x3x3 (for k=3,4,5). These are two independent
    3x3x3 multiplications. Each has rank at most 7 (Strassen). So total rank at most 14.
    
    But wait, they share the same A! So we can potentially share some terms.
    
    Let's construct: 
    For the first 3 columns (k=0,1,2): use a rank-7 decomposition of 3x3x3.
    For the last 3 columns (k=3,4,5): use another rank-7 decomposition.
    
    Total rank 14. The U matrix will have the same entries for both blocks (since A is shared),
    but V and W will be different.
    
    Actually, U[i*m+j, c] depends on i,j,c. For the first block, c=0..6, for second c=7..13.
    U[i*m+j, 0..6] comes from first 3x3x3, U[i*m+j, 7..13] from second.
    These can be different! So no sharing in U necessarily.
    
    But we CAN share if we design it that way. Let's just use rank 14 first.
    
    Standard 3x3x3 rank-7 decomposition (Strassen-like for 3x3):
    This is non-trivial. Let me use a known one or fall back to rank 9 (3x3 schoolbook is 9,
    but Strassen gives 7 for 2x2x2, not directly for 3x3x3).
    
    Actually, the minimal rank for 3x3x3 is known to be between 7 and 9. It's an open problem
    whether it's 7 or 8 or 9. Let's just use rank 9 (schoolbook) for each 3x3x3 block,
    giving total rank 18.
    
    Hmm, that's better than 54 but not great. Let me think if we can do better.
    
    Actually, for 3x3x3, the schoolbook rank is 9. We can't easily get below 9 without
    a known decomposition. Let's use 18 total.
    
    Wait, I can do better. For 3x3x6, note that the "B" part is 3x6. We can think of this
    as computing A * [B1 | B2] where B1, B2 are 3x3. The result is [A*B1 | A*B2].
    
    If we use a rank-r decomposition for 3x3x3, we get rank 2r for 3x3x6.
    With r=9, that's 18.
    
    Let me just implement rank 18 using two independent 3x3x3 schoolbook decompositions.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Use two 3x3x3 schoolbook decompositions, one for k=0,1,2 and one for k=3,4,5
    # Total rank = 9 + 9 = 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    # First block: k=0,1,2
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Second block: k=3,4,5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
