"""Matrix multiplication 3x3 by 3x7: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The schoolbook rank is 3*3*7 = 63.
    We try to find a lower rank using a structured approach.
    
    Strategy:
    1. Start with the known lower bound and try to construct a decomposition.
    2. Use the fact that for small matrices, we can exploit linearity and sparsity.
    3. Try rank values from a reasonable lower bound up to 63, and check if a valid
       decomposition exists using a constraint-solving approach.
    
    The tensor for matrix multiplication C = A * B where A is n x m and B is m x p:
    T[i, k, j] = 1 if the output element (i,k) depends on input A[i,j] * B[j,k]
    
    Actually, let's use the standard formulation:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    So the multiplication tensor has entries T[i,k,j] = 1 for all i,j,k,
    where i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition means:
    (A*B)[i,k] = sum_{c=0}^{r-1} U[i*m+j_c, c] * V[j_c*p+k_c, c] * W[k*n+i, c]
    
    Wait, let me re-read the problem. The output factor follows W[k*n+i, r].
    
    The factor matrices are:
    U: shape (n*m, rank) - for the first matrix A (n x m)
    V: shape (m*p, rank) - for the second matrix B (m x p)
    W: shape (p*n, rank) - for the output C (n x p), indexed as W[k*n + i, c]
    
    The decomposition is:
    C[i,k] = sum_{c=0}^{rank-1} sum_{j=0}^{m-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that's not quite right either. Let me think about this more carefully.
    
    In a tensor decomposition of matrix multiplication, we have:
    (A*B)[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j'*p+k, c] * B[j',k]) * W[k*n+i, c]
    
    No, that's not standard. Let me reconsider.
    
    The standard formulation is:
    We want to compute C = A * B where A is n x m, B is m x p.
    
    The multiplication tensor T has indices (i, j, k) where:
    - i: row of A and C (0..n-1)
    - j: column of A / row of B (0..m-1)  
    - k: column of B and C (0..p-1)
    
    T[i,j,k] = 1, and the product is sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i]... 
    
    Actually, the cleaner way: the bilinear map (A, B) -> A*B has tensor:
    M[i,k; j] = 1, meaning C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank-r decomposition gives:
    C[i,k] = sum_{c=0}^{r-1} f_c(A) * g_c(B) * h_c(i,k)
    
    where f_c(A) = sum_{i,j} U[i*m+j, c] * A[i,j]
    g_c(B) = sum_{j,k} V[j*p+k, c] * B[j,k]
    h_c(i,k) = W[k*n+i, c]
    
    So: C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', c] * B[j'',k''])
    
    For this to equal sum_j A[i,j] * B[j,k] for all A, B, we need:
    sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k'', c] = 1 if (i',j',j'',k'') = (i,j,j,k) else 0
    
    i.e., sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = delta_{i',i} * delta_{j',j} * delta_{j'',j} * delta_{k'',k}
    
    Wait, that's not right. Let me be more careful.
    
    C[i,k] = sum_c W[k*n+i, c] * (sum_{a,b} U[a*m+b, c] * A[a,b]) * (sum_{c2,d} V[c2*p+d, c] * B[c2,d])
    
    = sum_c W[k*n+i, c] * sum_{a,b,c2,d} U[a*m+b, c] * V[c2*p+d, c] * A[a,b] * B[c2,d]
    
    For this to equal sum_j A[i,j] * B[j,k], we need:
    sum_c W[k*n+i, c] * U[a*m+b, c] * V[c2*p+d, c] = 1 if (a,b,c2,d) = (i,j,j,k) else 0
    
    So the constraint is: for all i, k, a, b, c2, d:
    sum_c W[k*n+i, c] * U[a*m+b, c] * V[c2*p+d, c] = delta_{a,i} * delta_{b,j} * delta_{c2,j} * delta_{d,k}
    where j = b = c2 (the shared index)
    
    More precisely: the RHS is 1 if a=i, b=j, c2=j, d=k for some j, and 0 otherwise.
    But b and c2 are the same index j, so the condition is a=i, b=c2=j, d=k.
    
    So: sum_c W[k*n+i, c] * U[a*m+b, c] * V[b*p+d, c] = delta_{a,i} * delta_{d,k} * (1 if b is valid, i.e., always)
    
    Wait, I think I'm overcomplicating. The key constraint is:
    
    For the decomposition to be valid, for all i, k:
    sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) = sum_j A[i,j] * B[j,k]
    
    This must hold for ALL A and B, so by bilinearity:
    sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1, j2} * delta_{i,i} * delta_{k,k}
    
    No wait. Let me expand:
    LHS = sum_c W[k*n+i, c] * sum_{j1} U[i*m+j1, c] * A[i,j1] * sum_{j2} V[j2*p+k, c] * B[j2,k]
    = sum_{j1,j2} A[i,j1] * B[j2,k] * (sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c])
    
    RHS = sum_j A[i,j] * B[j,k]
    
    So we need: sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1,j2}
    
    for all i, k, j1, j2.
    
    This is the key system of equations!
    
    n=3, m=3, p=7.
    Variables: U (9 x r), V (21 x r), W (21 x r) where r is the rank.
    Equations: for each (i, k, j1, j2) with i in [0,3), k in [0,7), j1,j2 in [0,3):
    sum_c W[k*3+i, c] * U[i*3+j1, c] * V[j2*7+k, c] = delta_{j1,j2}
    
    That's 3*7*3*3 = 189 equations.
    
    The schoolbook solution has r=63 with U, V, W having single 1s.
    
    Let's try to find a lower rank. Known results:
    - For 3x3 x 3x3, the rank is at most 23 (Strassen-like), and it's known to be 23.
    - For 3x3 x 3x7, we can potentially do better than 63.
    
    Let me try a constructive approach. We can fix some structure and solve for the rest.
    
    Given the time constraint, let me try a practical approach:
    1. Try small ranks and use a randomized/structured initialization + Newton's method or least squares.
    2. Verify exactness with rational arithmetic.
    
    Actually, for exactness with integer/half-integer entries, let me try to construct a known low-rank decomposition.
    
    For 3x3 matrix multiplication, the Strassen decomposition has rank 7.
    We can extend this to 3x3 * 3x7 by using 7 copies of the 3x3x3 Strassen decomposition, giving rank 49.
    
    But we can do better. Let me think...
    
    Actually, let me just try to find a decomposition with rank less than 63 using a numerical approach and then verify.
    
    Given the complexity, let me start with a simpler improvement: use the fact that we can share computations.
    
    For now, let me return the schoolbook decomposition as a baseline, but try to find a lower rank.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Try to find a low-rank decomposition
    best_result = None
    best_rank = n * m * p  # 63
    
    # Schoolbook as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    sb = schoolbook()
    best_result = {"U": sb["U"], "V": sb["V"], "W": sb["W"]}
    best_rank = sb["rank"]
    
    # Try to find lower rank using a structured approach
    # For 3x3 x 3x7, try ranks from 40 to 62
    for r in range(40, best_rank):
        if time.time() > deadline - 5:
            break
        if _try_rank(n, m, p, r):
            res = _construct_rank(n, m, p, r)
            if res is not None:
                best_result = res
                best_rank = r
                break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

def _try_rank(n, m, p, r):
    """Placeholder - returns False for now."""
    return False

def _construct_rank(n, m, p, r):
    """Placeholder - returns None for now."""
    return None

# The above is a stub. Let me implement a real search.
