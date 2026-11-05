"""Matrix multiplication 3x5x7: rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Strategy: Use the Kronecker product structure.
    The 3x5x7 multiplication tensor T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}
    can be decomposed using the fact that:
    - A 3x5 matrix multiplication can be seen as 3 independent 1x5 multiplications
    - A 5x7 matrix multiplication can be seen as 5 independent 1x7 multiplications
    
    Actually, let's use a more direct approach. The standard result is:
    rank(MM_{m,n,p}) <= rank(MM_{m,n,p'}) * something... 
    
    Better approach: Use the known decomposition.
    For 3x5x7, we can use:
    - Decompose 5 = 2+3, so 3x5x7 = 3x2x7 + 3x3x7
    - Or use rank bounds: rank(MM_{n,m,p}) <= n*m*p but can be lower.
    
    Known small cases:
    - rank(MM_{2,2,2}) = 7 (Strassen)
    - rank(MM_{1,2,2}) = 3 (Laderman-like)
    - rank(MM_{2,2,1}) = 3
    
    For 3x5x7:
    We can write 3x5x7 as sum of independent sub-multiplications.
    
    Approach: Use the fact that rank(MM_{n,m,p}) is sub-multiplicative in some sense.
    
    Let me think about this differently. The tensor for multiplying (3x5) by (5x7) is:
    T_{i,j,k} where i in {0,1,2}, j in {0,...,4}, k in {0,...,6}
    T_{i,j,k} = 1 if the multiplication produces C[i,k] += A[i,j]*B[j,k]
    
    So T is a 3x5x7 tensor where T[i,j,k] = 1 for all valid (i,j,k) - it's the all-ones tensor!
    Wait, no. The multiplication tensor has T[i,j,k] = 1 iff the term A[i,j]*B[j,k] contributes to C[i,k].
    Yes, for standard matrix multiplication, T[i,j,k] = 1 for all (i,j,k) in {0..2}x{0..4}x{0..6}.
    
    So we need to decompose the all-ones 3x5x7 tensor.
    
    The rank of the all-ones tensor of size n x m x p is min(n*m, m*p, n*p) = min(15, 35, 21) = 15.
    
    Wait, is that right? The all-ones tensor J_{n,m,p} has J_{i,j,k} = 1 for all i,j,k.
    We can write J_{i,j,k} = sum_{r=1}^{R} u_{i,r} * v_{j,r} * w_{k,r}
    
    If we set R = min(n*m, m*p, n*p):
    For R = n*m = 15: 
    u_{i,r} = 1 if r = i*m + j' for some j', else 0... 
    
    Actually, let's think more carefully.
    
    J_{i,j,k} = 1 = sum_r u_{i,r} v_{j,r} w_{k,r}
    
    One simple decomposition: R = n*m*p (schoolbook), but we can do better.
    
    Key insight: J_{i,j,k} = (sum_i' delta_{i,i'})(sum_j' delta_{j,j'})(sum_k' delta_{k,k'})
    
    Better: Use the fact that 1 = 1*1*1, and we can factor through intermediate indices.
    
    Actually, the simplest non-trivial decomposition:
    J_{i,j,k} = sum_{r=1}^{min(n,m,p)} ... no, that doesn't work directly.
    
    Let me reconsider. The rank of the all-ones tensor J_{n,m,p}:
    
    Lower bound: max(n, m, p) (from slice ranks)
    Upper bound: min(n*m, m*p, n*p)
    
    For n=3, m=5, p=7:
    Lower bound: max(3,5,7) = 7
    Upper bound: min(15, 35, 21) = 15
    
    Can we achieve rank 15? 
    
    J_{i,j,k} = sum_{r=1}^{15} u_{i,r} v_{j,r} w_{k,r}
    
    Let r index pairs (a,b) where a in {0..2}, b in {0..4} (so 15 values).
    u_{i,r} = delta_{i,a}
    v_{j,r} = delta_{j,b}  
    w_{k,r} = 1
    
    Then sum_r u_{i,r} v_{j,r} w_{k,r} = sum_{a,b} delta_{i,a} delta_{j,b} * 1 = 1 if there exists (a,b) with a=i, b=j, which is always true.
    Wait, that gives sum = 1 for all (i,j,k). Yes!
    
    So rank <= 15.
    
    But wait, the task says the output factor follows W[k*n+i, r]. So the W matrix has indices (k*n+i, r) where k is the column index of C and i is the row index.
    
    Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    
    So for C = A*B where A is n x m and B is m x p, C is n x p.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has indices (i, j, k) with T[i,j,k] = 1.
    U has shape (n*m, rank): U[i*m+j, r]
    V has shape (m*p, rank): V[j*p+k, r]
    W has shape (p*n, rank): W[k*n+i, r]
    
    And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the all-ones tensor, we want: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Using the decomposition above with R = n*m = 15:
    Let r index (a, b) with a in {0,1,2}, b in {0,1,2,3,4}
    U[i*m+j, r] = 1 if (i,j) = (a,b), else 0
    V[j*p+k, r] = 1 (always)
    W[k*n+i, r] = 1 (always)
    
    Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = sum_r U[i*m+j,r] * 1 * 1
    = 1 if there exists r such that (i,j) = (a_r, b_r), which is exactly 1.
    
    So this works with rank 15!
    
    But can we do better? The lower bound is 7. Let me see if rank 7, 8, ..., 14 is possible.
    
    Actually, for the all-ones tensor, it's known that rank(J_{n,m,p}) = min(n*m, m*p, n*p) when n,m,p are all >= 2? No, that's not right either.
    
    Let me just go with rank 15 which is provably correct, and see if we can improve.
    
    Actually, I realize I should double-check my indexing. Let me re-derive.
    
    n=3 (rows of A), m=5 (cols of A = rows of B), p=7 (cols of B)
    U: (n*m, rank) = (15, rank), indexed by i*m+j
    V: (m*p, rank) = (35, rank), indexed by j*p+k
    W: (p*n, rank) = (21, rank), indexed by k*n+i
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For all-ones: T[i,j,k] = 1 for all i,j,k.
    
    My decomposition: R = 15, r = a*5+b where a in {0,1,2}, b in {0,1,2,3,4}
    U[i*5+j, r] = 1 if i==a and j==b, else 0
    V[j*7+k, r] = 1
    W[k*3+i, r] = 1
    
    Check: sum_r U[i*5+j, r] * V[j*7+k, r] * W[k*3+i, r]
    = sum_{a,b} [i==a and j==b] * 1 * 1
    = 1 (exactly one (a,b) equals (i,j))
    
    Yes! This works. Rank = 15.
    
    Can we do better? Let's try to see if rank < 15 is possible.
    
    Actually, I recall that for the all-ones tensor, the rank is indeed min(n*m, m*p, n*p) for n,m,p >= 2. This is because:
    - The tensor has a certain "border rank" and "rank" structure.
    
    Let me just go with 15 for now, which is a big improvement from 105 (3*5*7=105).
    
    Wait, 3*5*7 = 105, not 99. The current program has rank = n*m*p = 105.
    
    So going from 105 to 15 is a huge improvement.
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS
    
    # Use the all-ones tensor decomposition with rank = min(n*m, m*p, n*p)
    # For 3x5x7: min(15, 35, 21) = 15
    rank = min(n * m, m * p, n * p)
    
    # We'll use the decomposition where r indexes (i, j) pairs if n*m is the min
    # General approach: choose the smallest of the three products
    if n * m <= m * p and n * m <= n * p:
        # Use U indexed by (i,j), V and W are all ones
        R = n * m
        U = [[0] * R for _ in range(n * m)]
        V = [[1] * R for _ in range(m * p)]
        W = [[1] * R for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                r = i * m + j
                U[i * m + j][r] = 1
    elif m * p <= n * m and m * p <= n * p:
        # Use V indexed by (j,k), U and W are all ones
        R = m * p
        U = [[1] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[1] * R for _ in range(p * n)]
        for j in range(m):
            for k in range(p):
                r = j * p + k
                V[j * p + k][r] = 1
    else:
        # Use W indexed by (k,i), U and V are all ones
        R = n * p
        U = [[1] * R for _ in range(n * m)]
        V = [[1] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                r = k * n + i
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
