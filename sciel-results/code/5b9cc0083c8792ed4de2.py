"""
Program for matrix multiplication 2,2,7.
Uses a rank-25 decomposition based on Strassen's algorithm for 2x2 blocks.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i==i, j==j, k==k."""
    # T is a tensor of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    T = _build_target_tensor(n, m, p)
    # Reconstruct tensor from U, V, W
    # T_recon[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.max(np.abs(T_recon - T)) > tol:
        return False
    return True

def _strassen_block_decomposition(col_start, k_start, U, V, W, n, m, p):
    """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
    # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
    U[0, col_start+0] += 1
    U[3, col_start+0] += 1
    V[7*0 + k_start, col_start+0] += 1
    V[7*1 + (k_start+1), col_start+0] += 1
    W[2*k_start + 0, col_start+0] += 1
    W[2*(k_start+1) + 1, col_start+0] += 1

    # M2 = (A10 + A11) * B0,ks
    U[2, col_start+1] += 1
    U[3, col_start+1] += 1
    V[7*0 + k_start, col_start+1] += 1
    W[2*k_start + 1, col_start+1] += 1
    W[2*(k_start+1) + 1, col_start+1] -= 1

    # M3 = A00 * (B0,ks+1 - B1,ks+1)
    U[0, col_start+2] += 1
    V[7*0 + (k_start+1), col_start+2] += 1
    V[7*1 + (k_start+1), col_start+2] -= 1
    W[2*(k_start+1) + 0, col_start+2] += 1
    W[2*(k_start+1) + 1, col_start+2] += 1

    # M4 = A11 * (B1,ks - B0,ks)
    U[3, col_start+3] += 1
    V[7*1 + k_start, col_start+3] += 1
    V[7*0 + k_start, col_start+3] -= 1
    W[2*k_start + 0, col_start+3] += 1
    W[2*k_start + 1, col_start+3] += 1

    # M5 = (A00 + A01) * B1,ks+1
    U[0, col_start+4] += 1
    U[1, col_start+4] += 1
    V[7*1 + (k_start+1), col_start+4] += 1
    W[2*k_start + 0, col_start+4] -= 1
    W[2*(k_start+1) + 0, col_start+4] += 1

    # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
    U[2, col_start+5] += 1
    U[0, col_start+5] -= 1
    V[7*0 + k_start, col_start+5] += 1
    V[7*0 + (k_start+1), col_start+5] += 1
    W[2*(k_start+1) + 1, col_start+5] += 1

    # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
    U[1, col_start+6] += 1
    U[3, col_start+6] -= 1
    V[7*1 + k_start, col_start+6] += 1
    V[7*1 + (k_start+1), col_start+6] += 1
    W[2*k_start + 0, col_start+6] += 1

def _schoolbook_decomposition(col_start, k_idx, U, V, W, n, m, p):
    """Add schoolbook rank-4 terms for a single column k_idx."""
    # A00 * B0,k -> C0,k
    U[0, col_start+0] += 1
    V[7*0 + k_idx, col_start+0] += 1
    W[2*k_idx + 0, col_start+0] += 1
    
    # A01 * B1,k -> C0,k
    U[1, col_start+1] += 1
    V[7*1 + k_idx, col_start+1] += 1
    W[2*k_idx + 0, col_start+1] += 1
    
    # A10 * B0,k -> C1,k
    U[2, col_start+2] += 1
    V[7*0 + k_idx, col_start+2] += 1
    W[2*k_idx + 1, col_start+2] += 1
    
    # A11 * B1,k -> C1,k
    U[3, col_start+3] += 1
    V[7*1 + k_idx, col_start+3] += 1
    W[2*k_idx + 1, col_start+3] += 1

def _build_rank25_decomposition(n, m, p):
    """Build the standard rank-25 decomposition."""
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen for columns 0,1
    _strassen_block_decomposition(0, 0, U, V, W, n, m, p)
    # Strassen for columns 2,3
    _strassen_block_decomposition(7, 2, U, V, W, n, m, p)
    # Strassen for columns 4,5
    _strassen_block_decomposition(14, 4, U, V, W, n, m, p)
    # Schoolbook for column 6
    _schoolbook_decomposition(21, 6, U, V, W, n, m, p)
    
    return U, V, W

def _search_rank(n, m, p, target_rank, max_time=10.0):
    """
    Attempt to find a decomposition of the given target rank.
    Uses random local search starting from the rank-25 solution.
    Returns (U, V, W) if found, else None.
    """
    if target_rank >= 25:
        U, V, W = _build_rank25_decomposition(n, m, p)
        if target_rank == 25:
            return U, V, W
        else:
            # Truncate to target_rank
            return U[:, :target_rank], V[:, :target_rank], W[:, :target_rank]
    
    # For target_rank < 25, we need a smarter search.
    # This is a hard problem. We'll try a simple random perturbation approach
    # but it's unlikely to succeed quickly for rank 24.
    
    # Start from rank-25 solution
    U0, V0, W0 = _build_rank25_decomposition(n, m, p)
    
    # We need to find U, V, W of shape (4, target_rank), (14, target_rank), (14, target_rank)
    # such that sum_r U[:,r] V[:,r]^T W[:,r]^T = T
    
    # This is a system of polynomial equations.
    # Let's try to use the fact that the rank-25 solution has a specific structure.
    # If we can merge two terms, we might get rank 24.
    
    # For now, return None to indicate failure.
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    start_time = time.time()
    
    # First, try to find a better solution if time permits
    # But given the difficulty, we'll just return the known rank-25 solution
    U, V, W = _build_rank25_decomposition(n, m, p)
    
    # Verify the solution
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
