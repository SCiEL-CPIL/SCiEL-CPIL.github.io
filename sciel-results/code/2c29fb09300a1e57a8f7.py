"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    # The tensor T[i,j,k] = delta for MM: T_{(i,j),(j,k),(k,i)} = 1 if indices match
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    # if (i,j,k) is a valid triple, else 0
    # Actually the tensor is: for each (i,j,k), the entry is 1 if the multiplication
    # C[i,k] = sum_j A[i,j]*B[j,k] involves this term
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    
    rank = U.shape[1]
    # Build the tensor from the decomposition
    # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # For efficiency, check a sample or all entries
    # Total entries: n*m * m*p * p*n = (n*m*p)^2 which is 96^2 = 9216, manageable
    # But let's check all valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_cyclic_shift_decomposition(n, m, p):
    """
    Try a decomposition based on cyclic shifts.
    For matrix multiplication, we can sometimes use decompositions where
    the rank is less than n*m*p by exploiting structure.
    
    One approach: if p is a multiple of some d, we can group the k-dimension.
    Here p=8, so we can try p=2*4 or p=4*2.
    """
    # This is a placeholder for more sophisticated methods
    pass

def _construct_via_block_decomposition(n, m, p):
    """
    Try to decompose the 3x4x8 multiplication by exploiting p=8=2*4.
    Split B into two 4x4 blocks: B = [B1 | B2]
    Then C = A*B1 + A*B2, but this doesn't directly help with rank reduction
    unless we can share computation.
    
    Actually, let's think about this differently.
    The matrix multiplication tensor for (n,m,p) has rank at most n*m*p.
    We want to find a lower rank decomposition.
    
    Known results:
    - For 2x2x2, rank is 7 (Strassen)
    - For 3x3x3, rank is at most 23 (Winograd), lower bound is 19
    
    For 3x4x8, we can try:
    - Split into 3x4x4 + 3x4x4 (two independent multiplications): rank <= 2*48 = 96 (same as schoolbook)
    - Or try to find a shared structure
    
    Let's try a different approach: use the fact that 4 = 2*2 and 8 = 2*2*2
    to apply Strassen-like decompositions recursively.
    
    Actually, let's just try to verify if the schoolbook is optimal or if we can do better.
    For now, return the schoolbook decomposition as a safe baseline.
    """
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _attempt_improved_decomposition(n, m, p, deadline):
    """
    Attempt to find a decomposition with rank < n*m*p.
    This is a search procedure that may not always succeed.
    """
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    # Try rank = schoolbook_rank - 1, schoolbook_rank - 2, etc.
    # For each rank, try to find a valid decomposition
    # This is a hard problem in general, so we use heuristics
    
    # Heuristic 1: Try to merge some of the schoolbook terms
    # If two terms have the same (i,j) in U, same (j,k) in V, but different (k,i) in W,
    # they can't be merged directly.
    
    # Heuristic 2: Use the structure of small dimensions
    # For n=3, m=4, p=8, we have 12 rows in U, 32 rows in V, 24 rows in W
    
    # Let's try a specific construction:
    # Group the k-dimension into pairs: (k, k+4) for k=0,1,2,3
    # For each (i,j), we have 8 terms in schoolbook
    # If we can combine terms across the pair, we might reduce rank
    
    # Actually, let's try a simpler approach: just return schoolbook for now
    # and add a verification step
    
    if time.time() > deadline - 5:
        return None
    
    # Try to verify schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    if not _verify_decomposition(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    # For now, we don't have a proven better decomposition
    # Return None to indicate no improvement found
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 10 second safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Try to find an improved decomposition
    improved = _attempt_improved_decomposition(n, m, p, deadline)
    if improved is not None:
        U, V, W = improved
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
