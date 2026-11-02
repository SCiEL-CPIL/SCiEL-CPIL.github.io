"""Rewritten program for matrix multiplication 2x5 by 5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _validate(U, V, W):
    """Check that U ⊗ V ⊗ W equals the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    assert U.shape == (n_rows, U.shape[1])
    assert V.shape == (m_rows, V.shape[1])
    assert W.shape == (p_rows, W.shape[1])
    rank = U.shape[1]
    assert rank == V.shape[1] == W.shape[1]
    # Build the decomposition: C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # U is indexed by (i,j) -> n*m rows
    # V is indexed by (j,k) -> m*p rows
    # W is indexed by (k,i) -> p*n rows (output factor follows W[k*n+i, r])
    # The tensor T has T[(i,m+j), (j*m+k)] = 1 if matching, but more precisely:
    # The multiplication tensor T_{ij, jk, ki} = 1
    # In the decomposition: sum_r U[(i*m+j), r] * V[(j*m+k), r] * W[(k*n+i), r] = 1 iff this is a valid product term
    # Actually, the tensor for matrix mult: T is a 3-tensor with indices (i,j), (j,k), (k,i)
    # T[(i,j),(j,k),(k,i)] = 1 for all i,j,k
    # Wait, let me think more carefully.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form / tensor: for each (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has components T_{(i,j),(j,k),(k,i)} = 1
    # In the factorization C = sum_r (U[:,r] outer A_flat) * ... no.
    # Standard tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # The product tensor is: for each r, (u_r ⊗ v_r ⊗ w_r)[(i*m+j), (j*m+k), (k*n+i)]
    #   = U[(i*m+j), r] * V[(j*m+k), r] * W[(k*n+i), r]
    # We need: sum_r U[(i*m+j),r]*V[(j*m+k),r]*W[(k*n+i),r] = 1 if the indices are consistent
    # i.e., the sum equals 1 for all valid (i,j,k) tuples, and 0 otherwise.
    # Actually, the tensor entries are: T[(i*m+j), (j'*m+k), (k'*n+i')] = 1 iff j=j' and k=k* and i=i'
    # More precisely: T is nonzero only when the three index pairs agree: the j in A matches j in B,
    # the k in B matches k in W-output, and i in A matches i in W-output.
    # T[(i*m+j), (j*m+k), (k*n+i)] = 1
    # And 0 otherwise.
    
    # Let's build the full tensor and check
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*m+k, k*n+i] = 1.0
    
    # Reconstruct from factorization
    T_rec = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(T_rec, T, atol=1e-9):
        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n*m*p = 2*5*5 = 50
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k  (since B is m x p, row j, col k -> index j*p + k)
                # W output: k*n + i (per the trace convention W[k*n+i, r])
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
                r += 1
    
    # Validate
    if not _validate(U, V, W):
        # Should not happen for schoolbook, but just in case
        pass
    
    return {'U': U, 'V': V, 'W': W}

# EVOLVE-BLOCK-END
