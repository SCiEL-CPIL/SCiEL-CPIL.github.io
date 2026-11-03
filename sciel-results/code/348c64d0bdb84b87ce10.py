"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    Approach: Model the tensor decomposition as a linear algebra problem.
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[i,k,j] = 1 for all valid (i,j,k).
    
    A rank-r decomposition means:
    T[i,k,j] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We can fix W to be the identity matrix (or a simple structure) and solve for U and V.
    Specifically, if we set W[k*n+i, c] = delta_{k*n+i, c} (identity-like), then:
    T[i,k,j] = U[i*m+j, k*n+i] * V[j*p+k, k*n+i]
    
    This means for each (i,k), we need:
    sum_j U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all j where T[i,k,j]=1
    
    Actually, let's use a cleaner approach. The matrix multiplication tensor can be 
    viewed as a matrix M of size (n*m) x (m*p) when we fix the output index.
    
    For each output position (i,k), we have a matrix M_{i,k} of size n*m x m*p where:
    M_{i,k}[i*m+j, j*p+k] = 1 for all j, and 0 otherwise.
    
    Wait, that's not quite right. Let me think more carefully.
    
    The standard approach: The rank of the matrix multiplication tensor M_{n,m,p}
    is equal to the minimum number of terms needed to express it as a sum of 
    rank-1 tensors.
    
    For small dimensions, we can use the following approach:
    1. Construct the "slice" matrices for each output index
    2. Compute the rank of the combined system
    
    Actually, for 2x4 x 4x6, let's use a known result or compute directly.
    
    The key insight: we can fix W to be an identity matrix of size p*n x p*n,
    and then the problem reduces to finding U and V such that:
    For each (i,k): sum_j U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
    
    This is a system of linear equations. Let's set it up properly.
    
    If we fix W[k*n+i, c] = 1 if c == k*n+i, else 0, then:
    T[i,k,j] = U[i*m+j, k*n+i] * V[j*p+k, k*n+i]
    
    For this to equal 1 for all j, we need:
    U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all j
    
    This means U[i*m+j, k*n+i] = 1 and V[j*p+k, k*n+i] = 1 for all j.
    
    So the rank would be p*n = 12, but that's not right because we're missing the 
    coupling between different j's.
    
    Let me reconsider. The correct approach is:
    
    The tensor T[i,k,j] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition: T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    If we fix W to be the identity (W[k*n+i, c] = delta_{k*n+i, c}), then:
    T[i,k,j] = U[i*m+j, k*n+i] * V[j*p+k, k*n+i]
    
    For this to work, we need U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all i,j,k.
    
    This is only possible if U[i*m+j, k*n+i] = 1 and V[j*p+k, k*n+i] = 1 for all i,j,k,
    which gives rank = p*n = 12. But this doesn't account for the fact that the same
    U and V entries are shared across different k values.
    
    Actually, I think the issue is that fixing W to identity is too restrictive.
    
    Let's use a different approach: use the fact that the rank of the matrix 
    multiplication tensor can be computed via the "Strassen" method or by 
    solving a linear system.
    
    For practical purposes with 2x4 x 4x6, let's use the schoolbook decomposition
    which has rank 48, and see if we can improve it.
    
    Actually, let's just use the schoolbook decomposition as it's guaranteed to work
    and is simple to implement. The minimal rank might be lower, but finding it 
    requires more sophisticated algorithms.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    rank = rank_schoolbook
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
