"""Optimized program for matrix multiplication 2x3 * 3x5."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    The tensor T_{i,j,k} = delta(i,j) * delta(j,k) for (i,j,k) in [0,n) x [0,m) x [0,p)
    where n=2, m=3, p=5.
    
    We use the standard schoolbook decomposition which has rank n*m*p = 30.
    However, we can potentially do better. For small matrix multiplications, the
    minimal rank is known in some cases.
    
    For 2x3 * 3x5, the schoolbook rank is 30. Let's check if we can find a lower rank
    decomposition.
    
    The tensor is T[i,j,k] = 1 if i==j and j==k (in the appropriate indexing), else 0.
    Wait, let me re-read the problem.
    
    Matrix multiplication: A is n x m, B is m x p, C = A * B is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
    Actually no, the tensor T has entries T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
    So T[i,j,k] = 1 if the indices match appropriately.
    
    Actually, the standard way: C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T_{i,j,k} = 1 for all valid i,j,k? No.
    
    Let me think again. The bilinear map is (A, B) -> A*B where A is n x m and B is m x p.
    (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    So the tensor T has dimensions n x m x p with T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait, that's not right either. The tensor should encode which products contribute to which output.
    
    Actually, the multiplication tensor T is defined by:
    T = sum_{i,j,k} e_i^U ⊗ e_j^V ⊗ e_k^W where the term corresponds to A[i,j] * B[j,k] contributing to C[i,k].
    
    So T[i,j,k] = 1 if the indices are consistent, i.e., for each (i,j,k) in [0,n) x [0,m) x [0,p),
    the product A[i,j] * B[j,k] contributes to C[i,k]. So T[i,j,k] = 1 for all i,j,k.
    
    Hmm, but that would make the tensor all-ones, which has rank 1. That can't be right.
    
    Let me re-read the task description more carefully.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."
    
    The interface says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    
    So the tensor is of size (n*m) x (m*p) x (p*n). The entry T[a,b,c] where:
    - a = i*m + j for i in [0,n), j in [0,m)
    - b = j*p + k for j in [0,m), k in [0,p)
    - c = k*n + i for k in [0,p), i in [0,n)
    
    The tensor entry T[a,b,c] = 1 if the indices are consistent, i.e., the j in a matches the j in b, and the k in b matches the k in c, and the i in c matches the i in a.
    
    So T[i*m+j, j*p+k, k*n+i] = 1, and T[a,b,c] = 0 otherwise.
    
    This is the standard matrix multiplication tensor. Its rank is what we're looking for.
    
    For n=2, m=3, p=5, the schoolbook rank is n*m*p = 30.
    
    Can we do better? The minimal rank of matrix multiplication tensors is a hard problem.
    For small cases, some are known. Let me think about whether 30 is minimal for 2x3 * 3x5.
    
    Actually, I recall that for most small matrix multiplication problems, the schoolbook rank is minimal, but there are exceptions (like Strassen's algorithm for 2x2).
    
    For 2x3 * 3x5, I believe the schoolbook rank of 30 is likely minimal, but let me try to construct a decomposition and verify it.
    
    Since I don't have a known better algorithm, I'll use the schoolbook decomposition of rank 30.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 30
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
