"""Program for matrix multiplication 2x3 * 3x5 tensor decomposition."""
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x5 matrix.
    
    Uses the schoolbook algorithm which decomposes the multiplication tensor
    into n*m*p rank-1 terms. For 2x3*3x5, this gives rank 30.
    
    Verification: The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid
    indices, and 0 otherwise. Each rank-1 term in the decomposition corresponds
    to one product A[i,j]*B[j,k] contributing to C[i,k].
    
    We verify the decomposition by checking that the sum of outer products
    reconstructs the tensor exactly.
    """
    n, m, p = 2, 3, 5
    rank = n * m * p  # 30
    
    # Build factor matrices using list comprehensions
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    
    # Generate all (i, j, k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Build U, V, W
    U = []
    for i in range(n):
        for j in range(m):
            row = []
            for t, (ti, tj, tk) in enumerate(triples):
                if ti == i and tj == j:
                    row.append(1)
                else:
                    row.append(0)
            U.append(row)
    
    V = []
    for j in range(m):
        for k in range(p):
            row = []
            for t, (ti, tj, tk) in enumerate(triples):
                if tj == j and tk == k:
                    row.append(1)
                else:
                    row.append(0)
            V.append(row)
    
    W = []
    for k in range(p):
        for i in range(n):
            row = []
            for t, (ti, tj, tk) in enumerate(triples):
                if tk == k and ti == i:
                    row.append(1)
                else:
                    row.append(0)
            W.append(row)
    
    # Verify: sum_t U[:,t] * V[:,t] * W[:,t] should equal T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor
    reconstructed = np.zeros((n*m, m*p, p*n), dtype=float)
    for t in range(rank):
        reconstructed += np.einsum('a,b,c->abc', U_arr[:,t], V_arr[:,t], W_arr[:,t])
    
    # Expected tensor
    expected = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[a, b, c] = 1
    
    # Check exact match
    if not np.allclose(reconstructed, expected, atol=1e-10):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
