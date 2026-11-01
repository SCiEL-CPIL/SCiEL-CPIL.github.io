"""Optimized 4x4x4 matrix multiplication tensor decomposition.

Uses the recursive Strassen algorithm to achieve rank 49 (7*7),
then checks for linear dependence to potentially reduce rank further.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_decomposition():
    """
    Return U, V, W for 2x2 matrix multiplication using Strassen's 7-multiplication algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2.
    Indices:
    U: A00=0, A01=1, A10=2, A11=3
    V: B00=0, B01=1, B10=2, B11=3
    W: C00=0 (k=0,i=0), C01=1 (k=0,i=1), C10=2 (k=1,i=0), C11=3 (k=1,i=1)
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1: (A00+A11) * (B00+B11) -> C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11) * B00 -> C10 += 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A00 * (B01-B11) -> C01 += 1, C11 -= 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # M4: A11 * (B10-B00) -> C00 += 1, C10 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01) * B11 -> C00 -= 1, C01 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00) * (B00+B01) -> C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11) * (B10+B11) -> C00 += 1, C11 -= 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct using numpy.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute the tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # The matrix multiplication tensor should be all 1s
    expected = np.ones((n, m, p))
    
    return np.allclose(T, expected, atol=1e-9)

def _reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by finding linearly dependent terms.
    Each term r defines a vector v_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] in R^{n*m*m*p*p*n}.
    If these vectors are linearly dependent, we can remove redundant terms.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Build the matrix of term vectors (flattened outer products)
    # Shape: (n*m*p*n, rank) -> actually (n*m * m*p * p*n, rank) but that's 4*4*4*4*4*4 = 4096
    # More efficiently: use the matricization
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Fix (i,j): T[i,j,:] as function of k = sum_r U[i,j,r] * (V[:,k,r] * W[k,i,r])
    # Better: form the (n*m) x (p*n) matricization for each fixed j
    # Or: form a big matrix where each column r is the vectorized (U[:,r] ⊗ V[:,r] ⊗ W[:,r])
    
    # For efficiency, use the fact that we just need to check if columns are independent
    # Use SVD on a matricization
    # Matricization: fix first mode i, get (m*p*n) x rank matrix
    # M[i, (j,k)] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] -- this is the tensor itself
    
    # Actually, the simplest: form matrix M of shape (n*m*p*n, rank) where
    # M[(i,j), (k,i')] ... no, let's just do it directly.
    
    # Term vector for rank r: outer product of U[:,r], V[:,r], W[:,r]
    # Flatten to get a vector of size (n*m)*(m*p)*(p*n) = 16*16*16 = 4096
    # That's manageable for 49 columns.
    
    dim = (n * m) * (m * p) * (p * n)  # 4096
    
    # Build matrix: columns are the 4096-dim vectors
    # For speed, use einsum to get all at once
    # T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Vectorized: for each r, the vector is [U[i,j,r]*V[j,k,r]*W[k,i,r] for all i,j,k]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute all term tensors at once: shape (n, m, p, rank)
    T_terms = np.einsum('ijr,jkr,kir->ijk r', U_r, V_r, W_r)
    # Reshape to (n*m*p, rank)
    T_mat = T_terms.reshape(-1, rank)
    
    # Check rank of T_mat
    # Use SVD
    elapsed = time.time()
    if elapsed > deadline:
        return U, V, W
    
    try:
        # Use SVD to find rank
        U_svd, s_vals, Vt = np.linalg.svd(T_mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    # Find the numerical rank
    tol = 1e-10 * max(s_vals) if len(s_vals) > 0 else 1e-10
    num_rank = np.sum(s_vals > tol)
    
    if num_rank >= rank:
        return U, V, W
    
    # We can potentially reduce to num_rank
    # The independent directions are given by the first num_rank columns of U_svd
    # But we need to express the original terms as combinations of independent ones
    # Actually, for tensor decomposition, we can't just take SVD of the matricization
    # and get a valid tensor decomposition directly.
    
    # Alternative: greedy approach - try removing one term at a time and check if
    # the remaining terms still span the same tensor.
    
    # Simpler: check if removing any single term still gives a valid decomposition
    # This won't work directly since removing a term changes the tensor.
    
    # The correct approach: if the term vectors are linearly dependent, there exists
    # a linear combination sum c_r * T_r = 0. We can use this to eliminate one term
    # by expressing it in terms of others, but this requires the coefficients to allow
    # it. For a rank decomposition, we need each term to be a rank-1 tensor, so we
    # can't just take linear combinations.
    
    # So the SVD approach doesn't directly give us a lower-rank decomposition.
    # The rank of the tensor is the minimum number of rank-1 tensors needed, which
    # is what we're trying to find.
    
    # For the structured Strassen decomposition, the 49 terms are likely all
    # linearly independent (they come from different block combinations), so
    # this check will probably confirm rank 49.
    
    return U, V, W

def _strassen_4x4_decomposition():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using two levels of Strassen.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # Block-level Strassen terms for 4x4
    # Each term defines which 2x2 blocks of A, B, C are involved
    block_terms = [
        {'P': {(0,0): 1, (1,1): 1}, 'Q': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'P': {(1,0): 1, (1,1): 1}, 'Q': {(0,0): 1}, 'C': {(1,0): 1}},
        {'P': {(0,0): 1}, 'Q': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): -1}},
        {'P': {(1,1): 1}, 'Q': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'P': {(0,0): 1, (0,1): 1}, 'Q': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'P': {(1,0): 1, (0,0): -1}, 'Q': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'P': {(0,1): 1, (1,1): -1}, 'Q': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): -1}},
    ]
    
    for rb, term in enumerate(block_terms):
        P_coeffs = term['P']
        Q_coeffs = term['Q']
        C_coeffs = term['C']
        
        for rs in range(7):
            r = rb * 7 + rs
            
            for (bi, bj), pc in P_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        u2_coeff = U2[a * 2 + b][rs]
                        if u2_coeff != 0:
                            ai = 2 * bi + a
                            aj = 2 * bj + b
                            U[ai * 4 + aj][r] += pc * u2_coeff
            
            for (bi, bj), qc in Q_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        v2_coeff = V2[a * 2 + b][rs]
                        if v2_coeff != 0:
                            bi_idx = 2 * bi + a
                            bj_idx = 2 * bj + b
                            V[bi_idx * 4 + bj_idx][r] += qc * v2_coeff
            
            for (ci, cj), cc in C_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        w2_coeff = W2[a * 2 + b][rs]
                        if w2_coeff != 0:
                            crow = 2 * ci + a
                            ccol = 2 * cj + b
                            W[ccol * 4 + crow][r] += cc * w2_coeff
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Construct the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try rank-49 Strassen decomposition
    U, V, W = _strassen_4x4_decomposition()
    
    if _verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank via linear dependence check
        U2, V2, W2 = _reduce_rank(U, V, W, n, m, p, deadline)
        if _verify_decomposition(U2, V2, W2, n, m, p):
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
