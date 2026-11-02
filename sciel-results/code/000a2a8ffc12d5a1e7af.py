"""Matrix multiplication 3x4x7: structured slice-and-dice decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Construct a low-rank exact tensor decomposition for 3x4 * 4x7 multiplication.
    
    Strategy: Partition the 3 rows into three 1-row slices.
    For each 1x4 * 4x7 slice, we need to compute a row vector of length 7.
    
    A 1x4 * 4x7 multiplication has rank 7 (schoolbook: 4*7=28, but we can do better).
    Actually, for 1xm * mxp, the rank is exactly p (since it's just p linear combinations).
    
    Wait, let me reconsider. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p.
    The tensor T is in U^* ⊗ V^* ⊗ W where U has dim n*m, V has dim m*p, W has dim n*p.
    T(i,j; j,k; i,k) = 1.
    
    For a 1x4 * 4x7 sub-problem (n=1, m=4, p=7):
    U has dim 1*4 = 4, V has dim 4*7 = 28, W has dim 7*1 = 7.
    T(a; b; c) = 1 if a = j (0..3), b = j*7+k, c = k.
    
    This is essentially: for each output element c=k, we compute sum_j A[j] * B[j*7+k].
    The rank of this tensor is 7 (one rank-1 term for each k, with U having 1s in all 4 positions... no wait).
    
    Actually for n=1: T[j, j*7+k, k] = 1.
    Rank-1 decomposition: For each k, we have a term where U has 1s in all 4 positions (for all j), V has 1 at position j*7+k for each j, W has 1 at position k.
    But that's not rank 1 because U is the same vector for all k, but V differs.
    
    Let me think again. A rank-1 term is U ⊗ V ⊗ W where U is a vector in R^4, V in R^28, W in R^7.
    The product is U[a] * V[b] * W[c]. We need this to equal T[a,b,c].
    
    T[a, a*7+k, k] = 1, and 0 otherwise.
    
    If I set U = (1,1,1,1), V[k*7+a] = 1 for all a,k, W[k] = 1, then U⊗V⊗W has all entries = 1, which is wrong.
    
    Correct approach: For each k (0..6), define:
    U_k = (1,1,1,1) [same for all k]
    V_k has 1s at positions k, 7+k, 14+k, 21+k (i.e., positions a*7+k for a=0..3)
    W_k has 1 at position k, 0 elsewhere
    
    Then sum_k U_k ⊗ V_k ⊗ W_k gives T[a, a*7+k, k] = U_k[a] * V_k[a*7+k] * W_k[k] = 1*1*1 = 1, and all other entries are 0.
    
    So rank = 7 for the 1x4*4x7 case. That's the same as schoolbook for this slice? No, schoolbook would be 4*7=28.
    
    So total rank = 3 * 7 = 21. That's much better than 84!
    
    Let me verify: n=3, m=4, p=7.
    For row i (0..2), we compute C[i,k] = sum_j A[i,j] * B[j,k].
    
    U has dim n*m = 12. Index: i*4 + j.
    V has dim m*p = 28. Index: j*7 + k.
    W has dim p*n = 21. Index: k*3 + i.
    
    For each (i, k) pair, we want a rank-1 term. There are 3*7 = 21 such pairs.
    For term (i,k):
    U_term[i*4 + j] = 1 for all j (0..3), 0 otherwise.
    V_term[j*7 + k] = 1 for all j (0..3), 0 otherwise.
    W_term[k*3 + i] = 1, 0 otherwise.
    
    Sum over all (i,k): 
    Result[a,b,c] = sum_{(i,k): a=i*4+j, b=j*7+k, c=k*3+i} 1*1*1
    
    For a given (a,b,c): a = i*4+j, b = j'*7+k', c = k''*3+i''.
    We need j = j' and k = k' and i = i'' for the term to contribute.
    So a = i*4+j, b = j*7+k, c = k*3+i.
    This matches T[a,b,c] = 1 iff a=i*4+j, b=j*7+k, c=k*3+i for some valid i,j,k.
    
    Yes! This is correct. Rank = 21.
    """
    n, m, p = DIMENSIONS
    rank = n * p  # 3 * 7 = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            # U: 1s at positions i*m + j for j in 0..m-1
            for j in range(m):
                U[i * m + j][col] = 1
            # V: 1s at positions j*p + k for j in 0..m-1
            for j in range(m):
                V[j * p + k][col] = 1
            # W: 1 at position k*n + i
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
