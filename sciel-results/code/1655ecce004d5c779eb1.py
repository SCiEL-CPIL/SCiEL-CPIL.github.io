"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The matrix multiplication tensor T has T[a, b, c] = 1 iff a = i*m + j, b = j*p + k, c = k*n + i
    for some i in [0, n), j in [0, m), k in [0, p), and 0 otherwise.
    Here n=2, m=3, p=9.
    
    We use a decomposition of rank m*p = 27.
    For each pair (j, k) in [0, m) x [0, p), we create one rank-1 term.
    The term index r = j*p + k.
    
    For term r = j*p + k:
      U[i*m + j, r] = 1 for i in [0, n)
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1 for i in [0, n)
    
    Let's verify this gives the correct tensor:
    (U @ V @ W^T)[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    Let a = i*m + j, b = j'*p + k', c = k''*n + i''.
    
    U[a, r] is nonzero only if r = j*p + k' for some k' (since U[i*m+j, r]=1 iff r corresponds to row j of U, i.e., r = j*p + k for any k).
    Actually, U[i*m + j, r] = 1 if and only if the second index of a (which is j) matches the j in r = j*p + k. So U[a, r] = 1 iff r = j*p + k for some k in [0, p).
    
    V[b, r] is nonzero only if r = j'*p + k' (since b = j'*p + k', and V[b, r] = 1 iff r = b).
    
    So for U[a,r] and V[b,r] to both be nonzero, we need r = j*p + k AND r = j'*p + k', which implies j = j' and k = k'. So r must equal j*p + k'.
    
    Now check W[c, r] where c = k''*n + i'' and r = j*p + k'.
    W[k''*n + i'', r] = 1 iff r = j*p + k AND the i in W's row matches i''. Specifically, W[k''*n + i'', r] = 1 iff r = j*p + k'' AND i'' = i where... wait.
    
    Let me redefine more carefully. For term r = j*p + k:
      U[i*m + j, r] = 1 for all i in [0, n). So U[a, r] = 1 iff a's second index (a mod m) equals j.
      V[j*p + k, r] = 1. So V[b, r] = 1 iff b = j*p + k.
      W[k*n + i, r] = 1 for all i in [0, n). So W[c, r] = 1 iff c's first index (c // n) equals k.
    
    So (U V W^T)[a, b, c] = sum_r [a mod m == j(r)] * [b == r] * [c // n == k(r)]
    where j(r) and k(r) are defined by r = j*p + k.
    
    = [b mod p == k(b//p * p + (b mod p)) ... ] 
    
    Let b = j_b * p + k_b. Then the only r with V[b, r] = 1 is r = b = j_b * p + k_b.
    For this r, j(r) = j_b and k(r) = k_b.
    So the sum is [a mod m == j_b] * [c // n == k_b].
    
    But we need the result to be 1 iff there exist i, j, k such that a = i*m + j, b = j*p + k, c = k*n + i.
    Given a, b, c, let j = a mod m, k = b mod p (since b = j*p + k implies k = b mod p and j = b // p).
    Wait, b = j*p + k, so j = b // p and k = b % p.
    And a = i*m + j, so j = a % m and i = a // m.
    And c = k*n + i, so i = c % n and k = c // n.
    
    So the conditions are:
    - j from a: a % m
    - j from b: b // p
    - k from b: b % p
    - k from c: c // n
    - i from a: a // m
    - i from c: c % n
    
    For consistency, we need:
    a % m == b // p
    b % p == c // n
    a // m == c % n
    
    Now, our formula gives [a mod m == b // p] * [c // n == b % p].
    This does NOT include the condition a // m == c % n.
    
    So this rank-27 decomposition is incorrect. Let me reconsider.
    
    The schoolbook has rank n*m*p = 54. Can we do better?
    
    Let me try a different approach. What if we fix i and use rank m*p for each i? No, that would still be n*m*p.
    
    Actually, let me reconsider the tensor structure. T[a, b, c] = 1 iff:
    a = i*m + j
    b = j*p + k  
    c = k*n + i
    for some i, j, k.
    
    This is a standard matrix multiplication tensor. The rank of the matrix multiplication tensor <n, m, p> is not always n*m*p. For example, <2,2,2> has rank 7, not 8.
    
    For 2x3*3x9, let's see if we can find a better decomposition.
    
    Actually, let me just go with a correct decomposition. The rank-54 schoolbook is correct. But maybe there's a rank-27 or rank-18 decomposition.
    
    Let me think about this differently. The matrix multiplication A (n x m) times B (m x p) gives C (n x p) where C[i, k] = sum_j A[i, j] B[j, k].
    
    The tensor is T[i, j, k; i', j', k'] ... no, it's simpler. T[a, b, c] where a = (i, j), b = (j, k), c = (k, i) in some sense.
    
    Actually, I think for this specific case, rank 27 might work if I structure it correctly. Let me try:
    
    For each j in [0, m) and k in [0, p), create a term r = j*p + k.
    U[i*m + j, r] = 1 for all i
    V[j*p + k, r] = 1
    W[k*n + i, r] = 1 for all i
    
    As I showed, this gives [a%m == b//p] * [c//n == b%p], which is missing the condition a//m == c%n.
    
    So this doesn't work. Let me just use the schoolbook decomposition of rank 54, which is correct.
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
