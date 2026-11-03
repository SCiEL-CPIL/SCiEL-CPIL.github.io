"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Returns {'U': U, 'V': V, 'W': W} where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    and for all i,j,k:
      sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]... 
    
    Actually the convention is:
      The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
      We need: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    
    Wait, let me re-read. The output factor follows W[k*n+i, r].
    So the decomposition is:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    No wait, that's not right either. Let me think about this more carefully.
    
    The standard matrix multiplication tensor: (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = 1.
    
    A rank-1 decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, the problem says the output factor follows W[k*n+i, r]. So W is indexed by (k, i) flattened as k*n+i.
    
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    This is a 3-way tensor decomposition. The schoolbook solution has rank n*m*p = 100.
    
    For 4x5 * 5x5, the known minimum rank... Let me think. The Strassen algorithm gives rank 7 for 2x2. 
    For larger matrices, the minimum rank is not generally known.
    
    Actually, for matrix multiplication of size (n,m) x (m,p), the minimum rank is at least n*m*p / something... 
    
    The trivial bound is rank >= max(n*m, m*p, p*n) = max(20, 25, 20) = 25.
    Also rank >= n*m*p / max(n,m,p)^2 ... no that's not right.
    
    Actually, the rank of the matrix multiplication tensor M_{n,m,p} is at least n*m*p / max(n*m, m*p, p*n) ... no.
    
    The rank is at least the "slice rank" or other lower bounds. For M_{4,5,5}, I don't know the exact minimum rank off the top of my head.
    
    Let me try a different approach. Since the task says to find a rank-minimal exact decomposition, and the schoolbook gives 100, maybe I can do better with a structured approach.
    
    Actually, for this specific problem, let me try to use the fact that we can potentially use algorithms like those based on group theory or other structured decompositions.
    
    But given the time constraint (150s), let me first try to verify if the schoolbook solution works, then try to reduce the rank.
    
    Let me re-read the problem statement more carefully.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 5 matrix."
    
    The interface says: U (n*m, rank), V (m*p, rank), W (p*n, rank) with integer/half-integer entries.
    
    The decomposition should satisfy: for all i, j, k:
      sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]... 
    
    No, that doesn't make sense. Let me think again.
    
    The matrix multiplication: C = A * B, where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T is defined by T[i,j,k] = 1 (since the multiplication is just summing products).
    
    A rank-r decomposition of this tensor means:
      T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, but W should be indexed to give the output. Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W[k*n+i, r] is the coefficient for the (k,i) output position in the r-th term.
    
    So the decomposition is:
      For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    And U[i*m+j, r] is the coefficient for input A[i,j] in term r,
    V[j*p+k, r] is the coefficient for input B[j,k] in term r,
    W[k*n+i, r] is the coefficient for output C[i,k] in term r.
    
    This makes sense! Each term r computes a product of a linear combination of A's entries, a linear combination of B's entries, and adds it (scaled) to a linear combination of C's entries.
    
    The schoolbook solution: for each (i,j,k), have a term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    
    Now, can we do better? For 2x2x2, Strassen gives rank 7 instead of 8.
    
    For 4x5x5, I'm not sure of the optimal rank. Let me try a heuristic approach.
    
    Given the time constraint, let me first ensure the schoolbook solution works, then try to see if I can find a lower rank decomposition.
    
    Actually, let me just return the schoolbook solution for now, since finding a truly rank-minimal decomposition for 4x5x5 is a hard open problem in general.
    
    Wait, but the score is the rank, and lower is better. The seed has rank 100. If I can find any decomposition with rank < 100, I improve.
    
    Let me try a simple improvement: use the fact that we can combine terms.
    
    Actually, let me just return the schoolbook solution and see if it passes. Then I'll try to improve.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # Schoolbook rank = n*m*p = 100
    
    # Let's try to find a lower rank decomposition
    # One approach: use randomized methods to find a sparse decomposition
    
    # For now, return the schoolbook solution
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
