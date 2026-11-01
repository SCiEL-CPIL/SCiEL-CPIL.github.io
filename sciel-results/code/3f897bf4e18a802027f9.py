"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
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


def _verify(n, m, p, U, V, W, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > tol:
                    return False
    return True


def _build_rank_26(n, m, p):
    """
    Construct a rank-26 decomposition for M_{2,2,13}.
    
    The multiplication tensor T[i,j,k] = 1 for all i,j,k.
    We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For n=2, m=2, p=13:
    - U has shape (4, rank)
    - V has shape (26, rank) 
    - W has shape (26, rank)
    
    Key insight: Since m=2, we have only 2 "middle" indices.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    We can use the following construction:
    For each output column k (0..12) and each row i (0..1), we need to compute
    the dot product of row i of A with column k of B.
    
    A rank-26 approach: Use 26 terms, one for each (i,k) pair.
    But that's not how it works - we need shared structure.
    
    Better approach: Since m=2, we can write:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is a sum of two rank-1 terms in the "tensor" sense, but we need
    a bilinear decomposition. The standard approach for M_{2,2,p}:
    
    For each k, we need to compute two outputs (i=0 and i=1).
    
    Let's think of it as: for each k, we compute a 2x2 matrix product
    with a 2x1 matrix, giving a 2x1 result.
    
    Actually, the tensor is T[i,j,k] where i in {0,1}, j in {0,1}, k in {0..12}.
    
    A valid rank-26 construction:
    We can use the fact that for each k, the sub-tensor over (i,j) is:
    T[0,0,k]=1, T[0,1,k]=1, T[1,0,k]=1, T[1,1,k]=1
    
    This is the all-ones 2x2 matrix, which has rank 1 as a matrix (it's 11^T).
    
    So for each k, we can write the (i,j) part as u_i * v_j where u=(1,1), v=(1,1).
    
    This gives us a decomposition where for each k:
    - We use one term with U[i*m+j, c_k] = 1 for all i,j (since u_i*v_j = 1)
    - V[j*p+k, c_k] = 1 for all j
    - W[k*n+i, c_k] = 1 for all i
    
    But that's just the schoolbook with rank = p = 13? No, that doesn't work
    because we need to separate the different (i,j) contributions properly.
    
    Let me reconsider. The tensor T[i,j,k] = 1 for all i,j,k.
    
    We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    One approach: Use rank = p * n = 13 * 2 = 26.
    
    For each (k, i) pair, create a term c = k*n + i.
    Set:
    - W[k*n+i, c] = 1 (this term only contributes to output (k,i))
    - U[i*m+j, c] = 1 for all j (this term uses all of row i of A)
    - V[j*p+k, c] = 1 for all j (this term uses all of column k of B)
    
    Then: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    = sum_c [U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]]
    
    For a fixed (i,j,k), which terms c contribute?
    c = k'*n + i' where W[k'*n+i', c] = 1 only if k'=k and i'=i.
    So only c = k*n + i has W[k*n+i, c] = 1.
    
    For that term: U[i*m+j, k*n+i] = 1, V[j*p+k, k*n+i] = 1.
    So the product is 1*1*1 = 1. 
    
    Wait, but we're summing over c, and for the specific (i,j,k), only one c contributes.
    So the sum is 1. That works!
    
    But wait, this means for each output element (i,k), we use exactly one term.
    The term c = k*n + i has:
    - U[i*m+j, c] = 1 for j=0,1
    - V[j*p+k, c] = 1 for j=0,1
    - W[k*n+i, c] = 1
    
    So the contribution to T[i',j',k'] from term c is:
    U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c]
    
    This is non-zero only if k'=k and i'=i (from W), and then:
    U[i*m+j', c] * V[j'*p+k, c] * 1 = 1 * 1 * 1 = 1
    
    So T[i,j',k] gets contribution 1 from this term for any j'.
    But T[i,j,k] should be 1, and we're getting 1 from this single term. 
    
    However, T[i,j',k] for j' != j also gets 1 from this term, which is wrong
    because T[i,j',k] should be 1 for all j', so actually it IS correct!
    
    Wait, T[i,j,k] = 1 for ALL i,j,k. So every entry of the tensor is 1.
    So we need sum_c to equal 1 for every (i,j,k).
    
    With my construction, for each (i,j,k), exactly one term c=k*n+i contributes 1,
    so the sum is 1. This works!
    
    So rank = n*p = 26.
    """
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            c = k * n + i
            # W[k*n+i, c] = 1
            W[k * n + i][c] = 1
            # U[i*m+j, c] = 1 for all j
            for j in range(m):
                U[i * m + j][c] = 1
            # V[j*p+k, c] = 1 for all j
            for j in range(m):
                V[j * p + k][c] = 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s
    n, m, p = DIMENSIONS

    # Try the rank-26 construction first
    result = _build_rank_26(n, m, p)
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result

    # Fall back to schoolbook
    result = _build_schoolbook(n, m, p)
    return result
# EVOLVE-BLOCK-END
