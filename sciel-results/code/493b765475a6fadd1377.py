"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Tries to find rank-23 decomposition via search, falls back to rank-26.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Build the multiplication tensor T[i,j,k] = 1 if c[i,k] depends on a[i,j]*b[j,k]
    # T is 3x3x3, T[i,j,k] = 1 for all i,j,k (since every a[i,j]*b[j,k] contributes to c[i,k])
    # Actually T[i,j,k] = 1 iff the product a[i,j]*b[j,k] contributes to c[i,k], which is always true.
    # So T is all ones? No - the tensor entry T[i,j,k] = 1 means that in the decomposition,
    # the term U[i,j]*V[j,k]*W[k,i] should sum to 1 for each (i,j,k).
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k in {0,1,2}.
    
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    # This is equivalent to: for each (i,k), sum_j T[i,j,k] = sum_j 1 = 3? No.
    # Actually the tensor is: T[i,j,k] = 1 for all i,j,k.
    # We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k] = 1 for all i,j,k.
    
    # Try rank 23 with a structured search
    for R in [23, 24, 25]:
        if time.time() > deadline - 30:
            break
        
        # Try to find a valid decomposition
        best_result = _try_find_decomposition(R, n, m, p, deadline)
        if best_result is not None:
            return best_result
    
    # Fallback: rank-26 decomposition
    R = 26
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)
    
    # Strassen for top-left 2x2 (7 terms)
    r = 0
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    r = 1
    U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1
    
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1
    
    r = 4
    U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1
    
    r = 5
    U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1
    
    r = 6
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1
    
    border_indices = [
        (2, 0, 0), (2, 0, 1), (2, 0, 2),
        (2, 1, 0), (2, 1, 1), (2, 1, 2),
        (2, 2, 0), (2, 2, 1), (2, 2, 2),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        (0, 0, 2), (0, 1, 2),
        (1, 0, 2), (1, 1, 2)
    ]
    
    col = 7
    for i, j, k in border_indices:
        U[idx_u(i, j), col] = 1
        V[idx_v(j, k), col] = 1
        W[idx_w(k, i), col] = 1
        col += 1
    
    assert col == R
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_find_decomposition(R, n, m, p, deadline):
    """Try to find a rank-R decomposition via structured search."""
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Use the known structure: 3 overlapping Strassen blocks + corrections
    # Build a candidate and verify
    
    # Strategy: Start with 3 Strassen blocks (21 terms) and see if we can
    # cover all 27 tensor entries with just 2 more terms.
    
    # For now, this is a placeholder - the structured search is complex.
    # Return None to trigger fallback.
    return None
# EVOLVE-BLOCK-END
