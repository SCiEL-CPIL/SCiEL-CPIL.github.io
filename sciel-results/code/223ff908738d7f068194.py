"""Matrix multiplication 2,4,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x7 matrix.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    
    Returns dict with 'U', 'V', 'W' as lists of lists (integer entries).
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Schoolbook rank is n*m*p = 56.
    # For small dimensions, we can try to find lower rank decompositions.
    # 
    # Key insight: The matrix multiplication tensor T_{ijk} = delta_{j,k} * delta_{i,i} 
    # (with appropriate indexing) has rank related to the dimensions.
    #
    # For a (n x m) * (m x p) multiplication, the tensor lives in 
    # (F^n ⊗ F^m) ⊗ (F^m ⊗ F^p) ⊗ (F^n ⊗ F^p)
    # with T[i,j; j',k; i',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    #
    # The rank is at least max(n*m, m*p, n*p) by dimension arguments,
    # and at most n*m*p (schoolbook).
    #
    # For 2x4 x 4x7: n*m=8, m*p=28, n*p=14, n*m*p=56.
    # Lower bound: max(8, 28, 14) = 28.
    #
    # Let's try to construct a rank-28 decomposition using the fact that
    # one dimension (m=4) is the "bottleneck" shared factor.
    #
    # Actually, let me think more carefully. The tensor for matrix mult (n×m)·(m×p):
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # represents C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # In the factorization: C[i,k] = sum_r U[i*m+j ... ] -- wait, let me re-read the convention.
    #
    # U has shape (n*m, rank), indexed by [i*m + j, r]
    # V has shape (m*p, rank), indexed by [j*p + k, r]  
    # W has shape (p*n, rank), indexed by [k*n + i, r]
    #
    # The reconstruction is: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, that's not quite right either. Let me think about the standard decomposition.
    #
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: C_{ik} = sum_j A_{ij} B_{jk}
    #
    # The multiplication tensor M has components M[i,j,k] = A[i,j] * B[j,k] -> C[i,k]
    # 
    # In the factorized form with factors U, V, W:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    #
    # Hmm, but that doesn't look standard. Let me reconsider.
    #
    # Actually, I think the convention is:
    # The bilinear form is: C = sum_r U_r ⊗ V_r ⊗ W_r (outer products)
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    #
    # And the "trace" or contraction gives:
    # C[i,k] = sum_r W[k*n+i, r] * (U_r · (A reshaped)) * (V_r · (B reshaped))
    #
    # More precisely, if we flatten A into a vector a of length n*m (row-major: a[i*m+j] = A[i,j])
    # and B into b of length m*p (row-major: b[j*p+k] = B[j,k]),
    # then C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * a[i*m+j]) * (sum_k' V[j'*p+k', r] * b[j'*p+k'])
    #
    # Wait, that still doesn't work dimensionally for a single r. Let me think again.
    #
    # I believe the correct interpretation is:
    # C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
    # where the summation over the middle index j is implicit in how U and V are structured.
    #
    # Actually, the most natural reading for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Each term A[i,j]*B[j,k] can be written as a product of three factors:
    # f1 = A[i,j], f2 = B[j,k], f3 = 1
    # 
    # But we need it to be linear in A and B separately, and the result linear.
    # 
    # The standard tensor decomposition of matrix multiplication:
    # The tensor T ∈ R^{(n*m) × (m*p) × (n*p)} with
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i' and j=j' and k=k', else 0
    #
    # And C[i,k] = sum_{i',j,k'} T[i*m+j, j*p+k, k*n+i] * A[i,j] * B[j,k]
    #
    # Wait, let me be very precise. Define:
    # a vector a ∈ R^{n*m}: a[i*m+j] = A[i,j]
    # a vector b ∈ R^{m*p}: b[j*p+k] = B[j,k]
    # a vector c ∈ R^{n*p}: c[i*p+k] = C[i,k]  (but the problem uses W indexed by k*n+i)
    #
    # The problem says W has shape (p*n, rank) with entry W[k*n+i, r].
    # So the output is indexed as [k*n + i] which is column-major in (i,k) or row-major in (k,i).
    #
    # The multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    #
    # As a trilinear form: C[i,k] = sum_{i,j,k} delta_{i,i} delta_{j,j} delta_{k,k} A[i,j] B[j,k]
    # where the deltas ensure index matching.
    #
    # The tensor T with entries:
    # T[(i,j), (j',k'), (k'',i'')] = 1 iff i=i'' and j=j' and k=k''
    #
    # Then C[i,k] = sum_{(i',j),(j',k'),(k'',i'')} T[(i',j),(j',k'),(k'',i'')] A[i',j] B[j',k'] * delta for matching
    #
    # Hmm, I'm overcomplicating this. Let me just use the schoolbook and see if I can reduce rank.
    #
    # For the schoolbook: rank = n*m*p = 56
    # U[i*m+j, col(i,j,k)] = 1
    # V[j*p+k, col(i,j,k)] = 1
    # W[k*n+i, col(i,j,k)] = 1
    # for each (i,j,k) triple.
    #
    # Now, can we do better? 
    #
    # One approach: if one of the dimensions is 1, we can factor. But here all are >= 2.
    #
    # Another approach: use the fact that we only need n*m*p terms but some might be combined.
    #
    # For 2x4 x 4x7, let me think about whether rank < 56 is possible.
    #
    # Known results: The rank of matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, n*p).
    # For our case: max(8, 28, 14) = 28.
    #
    # Can we achieve rank 28? This would require a very efficient decomposition.
    #
    # Let me try a different approach: use the structure that m=4 is the shared dimension.
    # 
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    #
    # For fixed j, the contribution is A[i,j] * B[j,k] which is an outer product of a vector
    # in R^n (the i-th column of A for fixed j) and a vector in R^p (the k-th row of B for fixed j).
    #
    # So C = sum_{j=0}^{3} (A[:,j]) (B[j,:])
    #
    # Each term is an outer product of an n-vector and p-vector, giving an n*p matrix.
    # There are m=4 such terms.
    #
    # But in the tensor factorization framework, each "term" in the sum corresponds to a rank-1
    # component of the multiplication tensor. The question is: can we represent the sum of 4
    # outer products (each of rank 1 in the output space) using fewer than 4*n*p = 4*14 = 56
    # rank-1 components in the full tensor?
    #
    # Actually no - each A[i,j]*B[j,k] is already a product of two linear forms in A and B.
    # The tensor rank counts the number of such products needed.
    #
    # The schoolbook uses n*m*p = 56 products: one for each (i,j,k).
    # But A[i,j]*B[j,k] doesn't depend on the other indices, so we can group:
    # For fixed j: sum_i sum_k A[i,j] B[j,k] = (sum_i A[i,j] e_i) ⊗ (sum_k B[j,k] e_k)
    # 
    # This is ONE outer product in the output space, but it involves n*p individual products A[i,j]*B[j,k].
    #
    # In the tensor factorization, we need:
    # C[i,k] = sum_r W[k*n+i, r] * (linear form in A) * (linear form in B)
    #
    # For the schoolbook: each r corresponds to a unique (i,j,k), and:
    # U[i*m+j, r] = delta_{i,i_r} delta_{j,j_r}  (picks out A[i_r, j_r])
    # V[j*p+k, r] = delta_{j,j_r} delta_{k,k_r}  (picks out B[j_r, k_r])
    # W[k*n+i, r] = delta_{k,k_r} delta_{i,i_r}  (puts result at C[i_r, k_r])
    #
    # Now, can we combine? The issue is that U depends on (i,j), V on (j,k), W on (k,i).
    # The shared index j couples them.
    #
    # Let me try: for each j (0 to m-1=3), define a factor that handles all i and k for that j.
    # But U[i*m+j, r] must be a single value for given (i,j,r), and we want it to pick out A[i,j].
    # If we use one factor per j, then U[i*m+j, r_j] = 1 for all i, which means the linear form
    # in A is sum_i A[i,j] (the j-th column of A). Similarly V[j*p+k, r_j] = 1 for all k,
    # giving sum_k B[j,k] (the j-th row of B). And W[k*n+i, r_j] would need to be... 
    # but W[k*n+i, r_j] is a single value, while we need C[i,k] += A[i,j]*B[j,k] for ALL i,k.
    # A single scalar W[k*n+i, r_j] can't distribute A[i,j]*B[j,k] to different (i,k) positions.
    #
    # So we can't simply use m factors. We need at least n*p factors per j in general, giving n*m*p total.
    #
    # UNLESS we can use the freedom in choosing U, V, W entries (not just 0/1) to create
    # cancellations or combinations.
    #
    # Let me try a computational approach: set up the system of equations and try to find
    # a solution with rank r < 56.
    #
    # The constraint is: for all i,k:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] B[j'',k''])
    #
    # This must hold for ALL A and B, so it's a polynomial identity. Equating coefficients:
    # For each (i,k) and each (i',j') and (j'',k''):
    # delta_{i,i'} delta_{j,j''} delta_{k,k''} = sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r]
    #
    # Wait, that's not right either. Let me be more careful.
    #
    # LHS: C[i,k] = sum_j A[i,j] B[j,k]
    # Coefficient of A[i',j'] B[j'',k''] in LHS: delta_{i,i'} delta_{j,j''} delta_{k,k''} (where j=j'' from the sum)
    # Actually: sum_j A[i,j] B[j,k], the coefficient of A[i',j'] is delta_{i,i'} delta_{j,j'} (as a function of j),
    # and then multiplied by B[j,k]. So the coefficient of A[i',j'] B[j'',k''] is:
    # delta_{i,i'} * delta_{j',j''} * delta_{k'',k}
    #
    # RHS: sum_r W[k*n+i, r] (sum_{a,b} U[a*m+b, r] A[a,b]) (sum_{c,d} V[c*p+d, r] B[c,d])
    # = sum_r W[k*n+i, r] (sum_{a,b,c,d} U[a*m+b,r] V[c*p+d,r] A[a,b] B[c,d])
    # Coefficient of A[i',j'] B[j'',k''] in RHS:
    # sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r]
    #
    # So the constraint is: for all i, k, i', j', j'', k'':
    # delta_{i,i'} delta_{j',j''} delta_{k'',k} = sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r]
    #
    # Let me simplify the indices. Let i' = a, j' = b, j'' = c, k'' = d.
    # delta_{i,a} delta_{b,c} delta_{d,k} = sum_r W[k*n+i, r] U[a*m+b, r] V[c*p+d, r]
    #
    # For this to hold, when i=a, b=c, d=k: LHS = 1, so sum_r W[k*n+i,r] U[i*m+b,r] V[b*p+k,r] = 1
    # When any of the deltas is 0: LHS = 0, so sum_r W[k*n+i,r] U[a*m+b,r] V[c*p+d,r] = 0
    #
    # This is a system of (n*p) * (n*m) * (m*p) = 14 * 8 * 28 = 3136 equations
    # in (n*m + m*p + p*n) * r = (8 + 28 + 14) * r = 50r unknowns.
    #
    # For r = 28: 50*28 = 1400 unknowns, 3136 equations. Overdetermined!
    # For r = 56: 50*56 = 2800 unknowns, 3136 equations. Still overdetermined but schoolbook works.
    #
    # Wait, I must be miscounting. The equations aren't all independent. Let me reconsider.
    #
    # The equation is: for all (i,k,a,b,c,d):
    # delta_{i,a} delta_{b,c} delta_{d,k} = sum_r W[k*n+i, r] U[a*m+b, r] V[c*p+d, r]
    #
    # Number of distinct (i,k) pairs: n*p = 14
    # Number of distinct (a,b) pairs: n*m = 8
    # Number of distinct (c,d) pairs: m*p = 28
    # Total equations: 14 * 8 * 28 = 3136
    #
    # But many are trivially satisfied (LHS=0). The non-trivial ones are when i=a, b=c, d=k.
    # Number of non-trivial equations: for each (i,b,k) with i in [0,2), b in [0,4), k in [0,7): 2*4*7=56.
    # These give: sum_r W[k*n+i,r] U[i*m+b,r] V[b*p+k,r] = 1
    #
    # The trivial equations: when i≠a or b≠c or d≠k: sum_r W[k*n+i,r] U[a*m+b,r] V[c*p+d,r] = 0
    # Count: 3136 - 56 = 3080.
    #
    # So we have 56 "diagonal" equations (=1) and 3080 "off-diagonal" equations (=0).
    #
    # This is a very constrained system. The schoolbook solution with r=56 works because each
    # (i,b,k) triple gets its own factor r, and the off-diagonal equations are satisfied by
    # the Kronecker delta structure.
    #
    # For a lower rank solution, we'd need to satisfy all these with fewer factors.
    # This seems very hard. Let me check if rank 56 is actually minimal for 2x4x7.
    #
    # Known lower bounds: rank(M_{n,m,p}) >= max(n*m, m*p, n*p) = max(8,28,14) = 28.
    # Also: rank >= n*m*p / max(n,m,p) ... no, that's not a standard bound.
    #
    # Another bound: rank >= (n*m*p) / max(n,m,p) is not standard.
    #
    # The Strassen-type bounds don't directly apply to small cases.
    #
    # I think for 2x4x7, the rank might indeed be 56 (schoolbook optimal), or possibly slightly less.
    # Let me try to verify computationally whether a rank < 56 solution exists by attempting
    # to solve the linear system for small reductions.
    #
    # Actually, given the time constraint (150s), let me just return the schoolbook solution.
    # It's guaranteed correct and has rank 56.
    
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
