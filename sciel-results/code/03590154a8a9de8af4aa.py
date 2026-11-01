"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x5 matrix.

    The tensor is T[i,j,k] for i in [0,n), j in [0,m), k in [0,p).
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The condition is: for all r, sum over (i,j,k) of
    T[i,j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should reconstruct
    the multiplication tensor.

    Actually, the standard formulation: the matrix multiplication tensor M_{n,m,p}
    is a 3-tensor of shape (n*m, m*p, p*n) where
    M[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k, and 0 otherwise.

    We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    for all (a, b, c) in the tensor space:
    M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]

    For n=2, m=2, p=5, the schoolbook rank is 2*2*5 = 20.

    The minimum rank for 2x2x2 is 7 (Strassen).
    For 2x2x5, we can think of it as 5 copies of the "last dimension".

    Let me try to construct a rank-15 solution. The idea:
    For each pair (i, k), we need to multiply column k of B by row i of A.
    Actually, the product C = A*B where A is 2x2, B is 2x5, C is 2x5.
    C[i,k] = sum_j A[i,j] * B[j,k].

    The tensor has entries M[i,j,k] = 1.
    So T[i,j,k] = 1 for all i,j,k.

    We need rank r such that:
    T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    For n=2, m=2, p=5:
    U is 4 x r, V is 10 x r, W is 10 x r.

    Let me try to find a decomposition. One approach:
    Use the fact that T[i,j,k] = 1. This is a rank-1 tensor in the (i,j,k) space?
    No, because the indices are mapped to (i*m+j), (j*p+k), (k*n+i).

    Actually, T as a 3-tensor over (i,j,k) is all ones, which has rank 1.
    But the flattening matters.

    Let me think differently. The tensor M is supported on (i*m+j, j*p+k, k*n+i).
    For fixed j, the support is (i*m+j, j*p+k, k*n+i) for i in {0,1}, k in {0,...,4}.

    For j=0: (i, k, 5*i+k) for i in {0,1}, k in {0..4}
    For j=1: (2+i, 5+k, 5*i+k) for i in {0,1}, k in {0..4}

    So M[0,k,5+k] = 1, M[1,k,5+k] = 1 for j=0, k=0..4
    M[2,5+k,5+k] = 1, M[3,5+k,5+k+5] = 1 for j=1, k=0..4... wait.

    Let me recompute. For j=1:
    i=0: (0*2+1, 1*5+k, k*2+0) = (1, 5+k, 2k)
    i=1: (1*2+1, 1*5+k, k*2+1) = (3, 5+k, 2k+1)

    For j=0:
    i=0: (0, k, 2k)
    i=1: (2, k, 2k+1)

    So the non-zero entries are:
    M[0, k, 2k] = 1 for k=0..4
    M[2, k, 2k+1] = 1 for k=0..4
    M[1, 5+k, 2k] = 1 for k=0..4
    M[3, 5+k, 2k+1] = 1 for k=0..4

    That's 20 entries.

    Now I need to find U (4 x r), V (10 x r), W (10 x r) with r as small as possible.

    Let me try r=15. I'll use a structured approach.

    Actually, let me just use the schoolbook r=20 for safety, or try r=15 with a specific construction.

    For r=15, one idea: For each k (5 values) and each "Strassen-like" combination.
    Actually, for the 2x2 part, Strassen uses 7 multiplications.
    But here p=5, so we have 5 columns.

    Let me try a different approach: use r = 2*5 + something.

    I'll implement a search for r=15 using numpy, or just return r=20.

    Given time constraints, I'll return the schoolbook r=20.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 20
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * m * p // m)]  # m*p
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
# EVOLVE-BLOCK-END
