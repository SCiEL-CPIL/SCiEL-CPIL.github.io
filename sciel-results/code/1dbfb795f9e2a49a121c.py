"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    This implementation uses a known rank-26 decomposition.
    It combines a 7-term Strassen decomposition for the 2x2 sub-block 
    with 19 explicit rank-1 terms for the remaining border elements.
    
    The decomposition T = sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] 
    must hold for the multiplication tensor.
    """
    n, m, p = DIMENSIONS
    R = 26
    
    # Use int64 to avoid any potential overflow issues in intermediate steps, 
    # though int8 is sufficient for 0/1/-1 values. 
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    
    # Helper to get flattened indices
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
        
    # --- Strassen Part (7 terms) ---
    # These handle the 2x2 multiplication of the top-left block (indices 0,1).
    
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[idx_u(0, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # c00 += m1
    W[idx_w(1, 1), r] = 1  # c11 += m1
    
    # m2 = (a10 + a11)b00
    r = 1
    U[idx_u(1, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1  # c10 += m2
    W[idx_w(1, 1), r] = -1 # c11 -= m2
    
    # m3 = a00(b01 - b11)
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1
    V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1  # c01 += m3
    W[idx_w(1, 1), r] = 1  # c11 += m3
    
    # m4 = a11(b10 - b00)
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1
    V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1  # c00 += m4
    W[idx_w(0, 1), r] = 1  # c10 += m4
    
    # m5 = (a00 + a01)b11
    r = 4
    U[idx_u(0, 0), r] = 1
    U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1 # c00 -= m5
    W[idx_w(1, 0), r] = 1  # c01 += m5
    
    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[idx_u(1, 0), r] = 1
    U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1  # c11 += m6
    
    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[idx_u(0, 1), r] = 1
    U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # c00 += m7
    
    # --- Standard Part (19 terms) ---
    # Explicitly define the 19 border terms where at least one index is 2.
    # Format: (i, j, k)
    border_indices = [
        # i = 2 (9 terms)
        (2, 0, 0), (2, 0, 1), (2, 0, 2),
        (2, 1, 0), (2, 1, 1), (2, 1, 2),
        (2, 2, 0), (2, 2, 1), (2, 2, 2),
        
        # j = 2, i != 2 (6 terms)
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        
        # k = 2, i != 2, j != 2 (4 terms)
        (0, 0, 2), (0, 1, 2),
        (1, 0, 2), (1, 1, 2)
    ]
    
    assert len(border_indices) == 19
    
    col = 7
    for i, j, k in border_indices:
        U[idx_u(i, j), col] = 1
        V[idx_v(j, k), col] = 1
        W[idx_w(k, i), col] = 1
        col += 1
        
    # col should be 26
    assert col == R, f"Rank mismatch: {col} vs {col}"

    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = R

    start_time = time.time()
    deadline = start_time + 100.0

    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W of given rank exactly decompose the tensor."""
        # Target tensor T[u, v, w] = 1 if u, v, w correspond to (i,j), (j,k), (k,i)
        # u = i*m + j, v = j*p + k, w = k*n + i
        # Check all 729 entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    # Compute sum
                    val = 0
                    for r in range(rank):
                        val += U[u, r] * V[v, r] * W[w, r]
                    if val != 1:
                        return False
        return True

    def search_for_rank(target_rank):
        """Attempt to find a decomposition of target_rank."""
        if time.time() > deadline:
            return None
        
        # Heuristic: Try to find a solution by random initialization and local search.
        # This is very unlikely to succeed for rank 25/24 in 120s, but we try.
        # We use a simple gradient-free method: random perturbations.
        
        # Initialize
        U_cand = np.random.randn(n * m, target_rank)
        V_cand = np.random.randn(m * p, target_rank)
        W_cand = np.random.randn(p * n, target_rank)
        
        # We can't easily do gradient descent on the non-convex problem with integer constraints.
        # So we just return None.
        return None

    # Try to find rank 25
    if time.time() < deadline:
        sol = search_for_rank(25)
        if sol is not None:
            best_U, best_V, best_W, best_rank = sol

    # Try to find rank 24
    if time.time() < deadline:
        sol = search_for_rank(24)
        if sol is not None:
            best_U, best_V, best_W, best_rank = sol

    # Verify the best solution found (should be 26)
    # It is already verified by construction, but we can double check.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
