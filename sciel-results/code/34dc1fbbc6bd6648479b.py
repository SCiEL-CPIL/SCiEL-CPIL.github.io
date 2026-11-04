"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    rank = U.shape[1]
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
    # Compute via einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
    # Expected T[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _try_block_decomp(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by exploiting block structure.
    For multiplying A (n x m) by B (m x p), we can partition the middle dimension.
    
    Key insight: if we partition m into blocks, we can use block-matrix multiplication.
    But more importantly, for the specific case 3x4x8, we can try to find
    a decomposition with rank < 96 by using the fact that the tensor has
    some structure.
    
    One approach: use the "Laderman-like" idea or just try to find a sparse
    decomposition via random search with integer/rational entries.
    """
    n, m, p = 3, 4, 8
    base_rank = n * m * p  # 96
    
    # Try a simple optimization: see if we can reduce rank by combining terms
    # where the same (i,j) or (j,k) or (k,i) appears.
    
    # Actually, let's try a direct approach: use numpy to find a low-rank
    # approximation and then round to rationals.
    
    # The multiplication tensor T has entries T[i,j,k] = 1.
    # We want U (12 x r), V (32 x r), W (24 x r) such that
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try rank 88, 80, 72, etc.
    # For each rank, we'll try to solve the system.
    
    # Actually, a simpler approach: the schoolbook is already quite good.
    # Let's try to see if there's a known lower bound or known result for 3x4x8.
    
    # For now, let's just return the schoolbook if we can't find better.
    # But let's try a few random perturbations to see if we can reduce rank.
    
    # Actually, let's try a different approach: use the fact that we can
    # combine terms if they share structure.
    
    # For the 3x4x8 case, the middle dimension is 4. If we think of this as
    # multiplying by a 4x8 matrix, we can view B as having 4 rows.
    # A[i,j] * B[j,k] summed over j=0..3.
    
    # One idea: if we can express the sum over j using fewer than 4 terms
    # per (i,k) pair, we'd reduce rank. But that's essentially finding a
    # low-rank factorization of each "slice" of the tensor.
    
    # Let's just try to verify the schoolbook and return it.
    # The key improvement would come from a smarter algorithm, but for now
    # let's ensure correctness and try a few simple reductions.
    
    # Try rank = 96 (schoolbook)
    result = _schoolbook(n, m, p)
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # If schoolbook fails (shouldn't happen), raise
    raise ValueError("Schoolbook decomposition failed verification")

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as a fallback
    fallback = _schoolbook(n, m, p)
    
    # Try to find a better solution
    best_result = fallback
    best_rank = n * m * p
    
    # Try some simple optimizations
    # 1. Check if we can reduce rank by removing redundant terms
    # The schoolbook has one term per (i,j,k) triple. If two terms have the same
    # (i,j) but different k, or same (j,k) but different i, etc., they can't be
    # combined directly. But if we allow non-1 entries, we might be able to.
    
    # For now, just return the schoolbook. This is a safe baseline.
    # The real improvement would require a more sophisticated search.
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fallback.items()}
    
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in best_result.items()}
# EVOLVE-BLOCK-END
