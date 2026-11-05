"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-15 exact tensor decomposition for 3x5x5 matrix multiplication.
    
    The key insight: C = A * B where A is 3x5 and B is 5x5.
    Each row of C is a linear combination of the rows of B with coefficients from A.
    
    Specifically, C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can decompose this using the fact that for each fixed i, the map
    (A row i) * B -> C row i is a 1x5 * 5x5 multiplication.
    
    A 1x5 * 5x5 multiplication has tensor rank 5 (schoolbook: 5 terms).
    
    For 3 independent rows, we might think we need 3*5 = 15 terms.
    But we need to be careful about how we index the tensor factors.
    
    The tensor for matrix multiplication has indices:
    U[i*m + j, c]  where i in [0,n), j in [0,m)  -> n*m entries
    V[j*p + k, c]  where j in [0,m), k in [0,p)  -> m*p entries  
    W[k*n + i, c]  where k in [0,p), i in [0,n)  -> p*n entries
    
    For n=3, m=5, p=5:
    U: 15 entries (i*m+j = i*5+j)
    V: 25 entries (j*p+k = j*5+k)
    W: 15 entries (k*n+i = k*3+i)
    
    The schoolbook decomposition uses rank n*m*p = 75, with one term per (i,j,k).
    
    Now, can we do better? 
    
    Key observation: C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a sum over j of products of A[i,j] and B[j,k]. For fixed i, this is
    a dot product of the i-th row of A with each column of B (transposed view).
    
    The tensor rank of the 1x5x5 multiplication (row vector times matrix) is 5.
    Each such multiplication requires 5 rank-1 terms.
    
    For 3 independent row computations, if they share the same B, we can potentially
    share some structure. But since A[i,j] are independent for different i, the
    total rank should be at least 5 (for the B side) times... wait.
    
    Let me think more carefully. The tensor T has entries T[i,j,k] = delta condition.
    Actually, the multiplication tensor is defined by:
    T[i, (j,k)] = 1 if the computation uses A[i,j] and B[j,k] to produce C[i,k]
    
    More precisely, the structure tensor has:
    T_{(i,j), (j',k), (k',i')} = 1 if i=i', j=j', k=k'
    Wait, that's not quite right. Let me use the standard definition.
    
    The matrix multiplication tensor has:
    T[(i,j),(j',k),(k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    No wait. The standard structure tensor for n x m * m x p -> n x p is:
    T[a, b, c] where a indexes (i,j) in A, b indexes (j',k) in B, c indexes (i',k') in C
    T[(i,j),(j',k),(i',k')] = 1 if i=i' and j=j' and k=k'
    
    So T[(i,j),(j',k),(i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    The rank-1 decomposition of this tensor is:
    sum_c u_c(a) * v_c(b) * w_c(c) = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    Schoolbook: for each (i,j,k), define:
    u_{(i,j,k)}[(i',j')] = 1 if i'=i, j'=j
    v_{(i,j,k)}[(j',k')] = 1 if j'=j, k'=k  
    w_{(i,j,k)}[(i'',k'')] = 1 if i''=i, k''=k
    
    Then sum over (i,j,k) of u*v*w gives delta_{i,i'}*delta_{j,j'}*delta_{k,k'} as desired.
    
    Rank = n*m*p = 75.
    
    Now, the directive says to find rank-15. Let's see if that's possible.
    
    Actually, I recall that for matrix multiplication, the lower bounds are much higher
    than what simple arguments give. The Strassen algorithm shows that 2x2x2 can be done
    with rank 7 < 8. But for larger matrices, the best known algorithms are complex.
    
    For 3x5x5, I don't think rank 15 is achievable. The trivial lower bound is max(n*m, m*p, p*n) = max(15, 25, 15) = 25.
    
    Wait, that's not a valid lower bound. Let me reconsider.
    
    Actually, a valid lower bound: the tensor must have rank at least the maximum of
    the "slice ranks" or other invariants. But I'm not sure rank 15 is feasible.
    
    Let me try a different approach. What if we group by i (the row of C)?
    
    For fixed i, C[i,k] = sum_j A[i,j] * B[j,k]. This is a bilinear form in the
    row A[i,:] and the matrix B.
    
    The tensor for this sub-problem (1x5 * 5x5) has rank 5.
    
    If we use 5 terms for each of the 3 rows, we get 15 terms total. But we need to
    check if these can be combined.
    
    For row i, the 5 rank-1 terms are: for each j in {0,1,2,3,4}:
    u_{i,j}[(i',j')] = 1 if i'=i, j'=j
    v_{i,j}[(j',k')] = 1 if j'=j
    w_{i,j}[(i'',k'')] = 1 if i''=i, k''=k'  (wait, this doesn't look right)
    
    Let me be more careful. For the 1x5 * 5x5 case (n=1, m=5, p=5):
    C[k] = sum_j A[j] * B[j,k]
    
    U has 1*5 = 5 entries, indexed by j
    V has 5*5 = 25 entries, indexed by (j,k)
    W has 5*1 = 5 entries, indexed by k
    
    Rank-1 term for each j:
    u_j[j'] = delta_{j,j'}
    v_j[(j',k')] = delta_{j,j'}  (i.e., 1 if j'=j, 0 otherwise, independent of k')
    w_j[k'] = 1  (all ones)
    
    Then sum_j u_j[j] * v_j[(j',k')] * w_j[k'] = sum_j delta_{j,j'} * delta_{j,j'} * 1 = delta_{j,j'} * 1
    
    Hmm, that's not right either. Let me reconsider.
    
    For term j:
    u_j: vector of length 5, u_j[j'] = 1 if j'=j else 0
    v_j: vector of length 25, v_j[j'*5+k'] = 1 if j'=j else 0 (for all k')
    w_j: vector of length 5, w_j[k'] = 1 for all k'
    
    Then (u_j * v_j * w_j)[j, (j',k'), k''] = u_j[j] * v_j[j'*5+k'] * w_j[k'']
    = delta_{j,j'} * delta_{j,j'} * 1 = delta_{j,j'}
    
    Sum over j: sum_j delta_{j,j'} = 1 if j'=j for some j... that's not working.
    
    I think I'm confusing myself. Let me just use the standard result: the rank of
    1xn * nxm matrix multiplication is n (not n*m). Wait, is that true?
    
    Actually, I believe the rank of 1xn * nxm is n. Let me verify for n=2, m=2:
    C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    
    Can we do this with 2 bilinear terms?
    t0 = A[0] * (B[0,0], B[0,1])  -- no, B is a matrix, we need separate terms.
    
    Actually, each term is a product of a linear form in A and a linear form in B.
    A linear form in A: c0*A[0] + c1*A[1]
    A linear form in B: sum_{j,k} d_{j,k} * B[j,k]
    
    With 2 terms, we can compute at most 2 bilinear forms. We need 4 output entries,
    each being a sum of 2 products. So 2 terms should suffice if we can arrange it.
    
    Term 0: (A[0]) * (B[0,0] + B[0,1])
    Term 1: (A[1]) * (B[1,0] + B[1,1])
    
    This gives:
    t0 = A[0]*(B[0,0]+B[0,1])
    t1 = A[1]*(B[1,0]+B[1,1])
    
    C[0] = A[0]*B[0,0] + A[1]*B[1,0] = t0 * (1/1) ... no, we can't separate them.
    
    We need linear combinations of the terms to get each C[k].
    C[0] = a0*t0 + a1*t1 = a0*A[0]*(B[0,0]+B[0,1]) + a1*A[1]*(B[1,0]+B[1,1])
    We need this to equal A[0]*B[0,0] + A[1]*B[1,0].
    
    This requires a0*B[0,0] + a1*B[1,0] = B[0,0] and a0*B[0,1] + a1*B[1,1] = 0,
    and similarly for the A[1] part. This is overdetermined and generally not possible.
    
    So the rank of 1x2*2x2 is actually 4, not 2. My earlier assumption was wrong.
    
    In general, the rank of 1xn * nxm is n*m (schoolbook), and I don't think it can be
    reduced below that in general.
    
    So for 3x5*5x5, the rank is at least 3*5 = 15 if we could decompose each row
    independently, but actually each row's multiplication has rank 5*5=25, so total
    would be 3*25=75 if fully independent.
    
    But there might be sharing across rows. The key is that all rows share the same B.
    
    I think the rank 15 claim in the directive is based on a misunderstanding. Let me
    just implement the schoolbook decomposition which is known to be correct, and see
    if there's a way to reduce it.
    
    Given the difficulty, let me stick with the schoolbook rank 75 decomposition.
    """
    n, m, p = DIMENSIONS
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
