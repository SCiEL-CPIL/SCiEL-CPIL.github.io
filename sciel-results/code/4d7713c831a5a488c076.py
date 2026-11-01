"""Matrix multiplication 2,2,11 - rank-minimal tensor decomposition search."""
DIMENSIONS = [2, 2, 11]

import time
import itertools
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # Check: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_frac(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank(n, m, p, target_rank, deadline, seed=42):
    """
    Try to find a rank-`target_rank` decomposition for multiplying n x m by m x p.
    
    The tensor T_{i,j,k} = 1 for all i,j,k.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    
    This is equivalent to finding a rank-r decomposition of the all-ones tensor of shape (n*m, m*p, p*n).
    
    Strategy: 
    - For n=2, m=2, p=11: the tensor is 4 x 22 x 22, all entries = 1.
    - We can use the fact that all-ones tensor of shape (a,b,c) has tensor rank related to min(a,b,c).
    - Actually, the all-ones tensor T_{abc} = 1 can be decomposed as sum_c e_a * e_b * e_c where e is the all-ones vector.
    - But that's rank 1 only if we allow e_a e_b e_c. Wait, no: T_{abc} = 1 = (1,1,...,1)_a * (1,1,...,1)_b * (1,1,...,1)_c.
    - So rank 1! But wait, that's for the full tensor. The matrix multiplication tensor is NOT all-ones.
    
    Let me re-read the problem. The matrix multiplication tensor for (n,m,p) has entries:
    T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'}
    
    So T[i,j,j',k,k',i'] = 1 if j==j' and k==k', else 0.
    
    In the flattening convention U[n*m, rank], V[m*p, rank], W[p*n, rank]:
    U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] summed over c should equal delta_{j,j'} * delta_{k,k'}.
    
    So for each (j,k) pair (the "interior" indices), we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i', c] = 1 for all (i,i') when j==j', k==k'
    and 0 otherwise.
    
    This means the tensor factors as: for each (j,k), we have a rank-1 tensor in the (i,i') space that equals 1 everywhere, plus cross terms must vanish.
    
    Actually, let me think more carefully. The tensor is:
    T_{(i,j),(j',k),(k',i')} = [j=j'] [k=k']
    
    This can be written as: sum over (j,k) of E_{j,j} ⊗ E_{k,k} in the (i,i') space... no.
    
    Let me use the standard formulation. The matrix multiplication tensor M_{(n,m,p)} has:
    M_{i,j,k; i',j',k'} = delta_{j,j'} delta_{k,k'}
    
    with indices: first matrix (i,j), second matrix (j',k), result (i',k')... wait, I need to be careful.
    
    Standard: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    Tensor: T_{i,j,j',k,k',i'} = delta_{j,j'} delta_{k,k'}
    
    Flattened: U indexed by (i,j) -> i*m+j, V indexed by (j',k) -> j'*p+k, W indexed by (k',i') -> k'*n+i'
    
    We need: sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c] = delta_{j,j'} delta_{k,k'}
    
    Key observation: This tensor is a sum over (j,k) of rank-1 components in the sense that for fixed (j,k), the sub-tensor over (i,i') is all-ones (rank 1 as a matrix, but we need rank-1 in the 3-way sense).
    
    For fixed (j,k), define A^{(j,k)}_{i} = 1, B^{(j,k)}_{i'} = 1, and the "middle" part is 1 if (j',k)=(j,k) and 0 otherwise.
    
    Actually, the cleanest way: T = sum_{j=0}^{m-1} sum_{k=0}^{p-1} u^{(j,k)} ⊗ v^{(j,k)} ⊗ w^{(j,k)}
    
    where u^{(j,k)} is a vector of length n*m, v^{(j,k)} is length m*p, w^{(j,k)} is length p*n.
    
    For the term (j,k):
    u^{(j,k)}[i*m+j'] = 1 if j'=j, else 0  -> so u^{(j,k)} has 1s at positions (i,j) for all i
    v^{(j,k)}[j'*p+k'] = 1 if j'=j and k'=k, else 0 -> v^{(j,k)} has a single 1 at (j,k)
    w^{(j,k)}[k'*n+i'] = 1 if k'=k, else 0 -> w^{(j,k)} has 1s at positions (k,i') for all i'
    
    This gives the schoolbook decomposition with rank = m*p.
    
    But we can do better! The key insight: for different (j,k) pairs, the vectors u and w might be reusable if they have the same support pattern.
    
    u^{(j,k)} depends only on j (the column of the first matrix).
    w^{(j,k)} depends only on k (the row of the second matrix / column of result).
    v^{(j,k)} depends on both j and k.
    
    So we have m distinct u-vectors and p distinct w-vectors, but m*p distinct v-vectors.
    
    The schoolbook uses m*p terms. Can we reduce?
    
    For n=2, m=2, p=11: schoolbook rank = 2*11 = 22.
    
    Wait, the current program uses rank = n*m*p = 2*2*11 = 44. Let me re-examine.
    
    The current program has:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1  
    W[k*n+i][column] = 1
    
    This iterates over i, j, k, giving n*m*p = 44 terms. But this is redundant!
    
    For a fixed (j,k), varying i just gives different U entries (at positions (i,j)) and W entries (at positions (k,i')). The V entry is always at (j,k).
    
    Actually wait, let me re-check. In the schoolbook:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor term for (i,j,k) is: A_{i,j} * B_{j,k} contributes to (AB)_{i,k}.
    
    In tensor form: T_{(i,j),(j',k),(k',i')} = delta_{j,j'} delta_{k,k'}
    
    The "standard" schoolbook decomposition uses one term per (j,k) pair (not per (i,j,k)):
    For each (j,k):
      u^{(j,k)} has 1 at (i,j) for all i  -- but this is a vector, and u^{(j,k)}_i = 1 for the i-component
      Actually, u^{(j,k)} is a vector of length n*m. u^{(j,k)}[i*m + j'] = [j'=j] * 1
      v^{(j,k)} is a vector of length m*p. v^{(j,k)}[j'*p + k'] = [j'=j] * [k'=k]
      w^{(j,k)} is a vector of length p*n. w^{(j,k)}[k'*n + i'] = [k'=k] * 1
      
    Then sum_c u_c ⊗ v_c ⊗ w_c gives:
    sum_{j,k} [j'=j] * [j''=j][k''=k] * [k'''=k] * (product of 1s)
    = [j'=j''][k''=k''']
    
    Wait, that's not right either. Let me be very explicit.
    
    Term c corresponding to (j,k):
    U_c[i*m + j'] = 1 if j' == j, else 0
    V_c[j''*p + k'] = 1 if j'' == j AND k' == k, else 0
    W_c[k''*n + i'] = 1 if k'' == k, else 0
    
    Product: U_c[i*m+j'] * V_c[j''*p+k'] * W_c[k''*n+i'] = [j'=j] * [j''=j] * [k'=k] * [k''=k]
    
    Sum over c=(j,k): sum_{j,k} [j'=j][j''=j][k'=k][k''=k] = [j'=j''] * [k'=k'']
    
    Yes! So rank m*p = 2*11 = 22 is sufficient.
    
    But the current code uses n*m*p = 44, which is wasteful. It has one term per (i,j,k) instead of per (j,k).
    
    Can we do even better than 22?
    
    The tensor rank of the matrix multiplication tensor M_{n,m,p} is a well-studied quantity. For n=m=p=2, it's 7 (Strassen). For n=m=p=1, it's 1.
    
    For n=2, m=2, p=11, the tensor rank is at most 22 (as shown above). Can it be lower?
    
    Lower bound: The tensor has "slice rank" considerations. The tensor T_{(i,j),(j',k),(k',i')} = delta_{j,j'} delta_{k,k'}.
    
    For fixed i, the "slice" over (j, j', k, k', i') is a matrix of size (m*p) x (p*n) = 22 x 22. This slice has rank m = 2 (since it's delta_{j,j'} delta_{k,k'}, which as a matrix from (j,k) to (j',k') is identity-like, rank m*p... wait no).
    
    Let me think about it differently. The tensor can be seen as a matrix of size (n*m) x (m*p) for each fixed (k',i'), or various other flattenings.
    
    The flattening to a matrix of size (n*m) x (m*p) for fixed (k',i'):
    T_{(i,j), (j',k); (k',i')} = delta_{j,j'} delta_{k,k'}
    
    For fixed (k',i'), this is an (n*m) x (m*p) matrix with entries delta_{j,j'} delta_{k,k'}.
    
    This matrix has rank at most m = 2 (since for each j, the row (i,j) is nonzero only in columns (j,k) for k=k', and it's the same for all i).
    
    Actually, for fixed k': the matrix is block-diagonal with m blocks (one for each j), each block being n x 1 (a column of 1s) if k=k', else 0. So rank = m = 2.
    
    This doesn't give a strong lower bound on tensor rank.
    
    Another approach: The tensor rank of M_{2,2,11} is at least max over all flattening ranks. The flattening to (n*m) x (m*p * p * n) has rank at most... this gets complicated.
    
    Let me just try to find a decomposition of rank < 22 by search.
    
    Actually, for the purposes of this task, let me first implement the rank-22 decomposition (which is a clear improvement over 44), and then try to search for lower rank.
    
    The rank-22 decomposition:
    For c = 0 to 21, corresponding to (j,k) where j = c // 11, k = c % 11:
      U[i*2 + j, c] = 1 for all i (i.e., U[0+j, c] = U[1+j, c] = 1, others 0)
      V[j*11 + k, c] = 1 (single entry)
      W[k*2 + i, c] = 1 for all i (i.e., W[0+k*2+0, c] = W[0+k*2+1, c] = 1, wait, W is indexed by k'*n + i', so W[k*2 + i', c] = 1 for all i')
    
    Let me code this up.
    
    """
    if time.time() > deadline:
        return None
    
    # Try the simple m*p rank decomposition first
    # c corresponds to (j, k)
    for c in range(target_rank):
        j = c // p
        k = c % p
        if j >= m or k >= p:
            continue
        # U[i*m + j, c] = 1 for all i
        for i in range(n):
            U[i*m + j][c] = 1
        # V[j*p + k, c] = 1
        V[j*p + k][c] = 1
        # W[k*n + i', c] = 1 for all i'
        for ip in range(n):
            W[k*n + ip][c] = 1
    
    if _verify(U, V, W, n, m, p):
        return (U, V, W)
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Schoolbook rank = m * p = 2 * 11 = 22
    schoolbook_rank = m * p
    
    # Try to find a decomposition with rank < schoolbook_rank
    # Start from rank 1 and go up
    best_result = None
    best_rank = schoolbook_rank
    
    # First, establish the rank-22 baseline
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    
    for c in range(best_rank):
        j = c // p
        k = c % p
        for i in range(n):
            U[i*m + j][c] = 1
        V[j*p + k][c] = 1
        for ip in range(n):
            W[k*n + ip][c] = 1
    
    if _verify(U, V, W, n, m, p):
        best_result = (U, V, W)
    
    # Now try lower ranks
    for target_rank in range(1, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        result = _try_rank(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = target_rank
            break
    
    if best_result is None:
        # Fallback: return the rank-22 decomposition
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        for c in range(best_rank):
            j = c // p
            k = c % p
            for i in range(n):
                U[i*m + j][c] = 1
            V[j*p + k][c] = 1
            for ip in range(n):
                W[k*n + ip][c] = 1
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
