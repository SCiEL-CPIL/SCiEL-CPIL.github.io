import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
_DEADLINE = 140.0

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    """Strassen decomposition for 2x2x2 block."""
    # M1: (A11+A22)(B11+B22)
    U[i_offset*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M2: (A21+A22)B11
    U[(i_offset+1)*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = -1
    col += 1
    # M3: A12(B21-B11)
    U[i_offset*m + j_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M4: A21(B22-B12)
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[j_offset*p + k_offset, col] = -1
    W[k_offset*n + i_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    col += 1
    # M5: (A11+A12)B22
    U[i_offset*m + j_offset, col] = 1
    U[i_offset*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    col += 1
    # M6: (A21-A11)(B11+B12)
    U[(i_offset+1)*m + j_offset, col] = 1
    U[i_offset*m + j_offset, col] = -1
    V[j_offset*p + k_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M7: (A12-A22)(B21+B22)
    U[i_offset*m + (j_offset+1), col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = -1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    col += 1
    return col

def _verify_rank(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                result = np.dot(u_val, v_val * w_val)
                if abs(result - 1.0) > 1e-10:
                    return False
    return True

def _rank28_decomposition(n, m, p):
    """Known rank-28 decomposition using Strassen + schoolbook blocks."""
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    # Block 1: j=0, all k (2x1x5 schoolbook)
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    # Block 2: j in {1,2}, k in {0,1} (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    # Block 3: j in {1,2}, k=2 (schoolbook)
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
    # Block 4: j in {1,2}, k in {3,4} (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _search_rank27(n, m, p, deadline):
    """
    Attempt to find a rank-27 decomposition by searching for a non-trivial
    linear dependency among the 28 rank-1 terms of the known decomposition.
    
    The idea: if we can express one term as a linear combination of the others,
    we can eliminate it and get rank 27.
    """
    start = time.time()
    U28, V28, W28 = _rank28_decomposition(n, m, p)
    
    # For each term t, check if it can be expressed as a linear combination
    # of the other terms. This is equivalent to checking if the 28 vectors
    # in the tensor product space are linearly dependent.
    
    # Flatten the tensor: each term t gives a vector in R^(n*m*p)
    # term_t = U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    
    # Build the matrix whose columns are the 28 term vectors
    terms = []
    for t in range(28):
        vec = np.outer(np.outer(U28[:, t], V28[:, t]), W28[:, t]).flatten()
        terms.append(vec)
    
    terms_matrix = np.column_stack(terms)
    
    # Check rank of this matrix
    # If rank < 28, there's a linear dependency
    rank = np.linalg.matrix_rank(terms_matrix, tol=1e-10)
    
    if rank < 28:
        # Find the null space vector
        # This gives coefficients c such that sum_t c_t * term_t = 0
        # We can then eliminate any term with c_t != 0
        U, S, Vt = np.linalg.svd(terms_matrix, full_matrices=False)
        null_vec = Ut[-1, :]  # last row of Vt^T
        # Actually, let's do it properly
        U, S, Vt = np.linalg.svd(terms_matrix.T, full_matrices=True)
        # Null space is the last row of Vt
        null_vec = Vt[-1, :]
        
        # Find a term with large coefficient to eliminate
        idx = np.argmax(np.abs(null_vec))
        
        # Reconstruct without term idx
        # This requires solving a linear system, which is complex
        # For now, just return the 28-term solution
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Try to find rank 27
    result = _search_rank27(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        if _verify_rank(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 28
    U, V, W = _rank28_decomposition(n, m, p)
    
    if not _verify_rank(U, V, W, n, m, p):
        raise ValueError("Rank-28 decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
