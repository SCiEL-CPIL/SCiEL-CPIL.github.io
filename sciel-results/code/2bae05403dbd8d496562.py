"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
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

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if this is a valid (i,j,k) product term, 0 otherwise.
    # Wait, the tensor for matrix multiplication C = A * B where A is n x m and B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form has terms A[i,j] B[j,k] contributing to C[i,k].
    # So the tensor T is of order 3, with indices (i, j, k) corresponding to (A, B, C).
    # T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # In our case, C has dimensions n x p.
    # So for a fixed j, the term A[i,j] B[j,k] contributes to C[i,k].
    # Thus, the non-zero entries of T are at (i, j, k) for all i in [0,n), j in [0,m), k in [0,p).
    # There are exactly n*m*p such entries, all equal to 1.
    # The total size of T is (n*m) * (m*p) * (p*n).
    # We compute the reconstructed tensor and check if it matches the target.
    
    # Efficiently: for each r, U[:,r] (n*m), V[:,r] (m*p), W[:,r] (p*n).
    # Contribution to T is outer(U[:,r], outer(V[:,r], W[:,r]))
    # Reshape U_r to (n, m), V_r to (m, p), W_r to (p, n) -> wait, W index is k*n+i.
    # So W[:,r] reshaped to (p, n) gives W_kn.
    # Let's just use a loop over r and accumulate, or use einsum if fast enough.
    # Since rank is small (up to 72), a loop is fine.
    
    # Target tensor T_target
    # T_target[i*m+j, j*p+k, k*n+i] = 1
    # It's easier to check specific entries.
    
    # Let's build the full tensor T_reconstructed of shape (n*m, m*p, p*n)
    # This might be large: 9 * 24 * 24 = 5184. That's small.
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        # u_r is (n*m), v_r is (m*p), w_r is (p*n)
        # We want to add outer(u_r, outer(v_r, w_r)) to T_recon
        T_recon += np.einsum('a,b,c->abc', u_r, v_r, w_r)
    
    # Now check if T_recon matches the target.
    # Target: T_target[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # All other entries should be 0.
    
    # Check the 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = T_recon[i*m+j, j*p+k, k*n+i]
                if abs(val - 1.0) > tol:
                    return False
    # Check a sample of 0s to ensure no spurious entries
    # Total entries: n*m * m*p * p*n
    # Non-zero entries: n*m*p
    # We can just check the whole thing if it's small.
    # 5184 entries is tiny.
    # Create target mask
    # Actually, just sum of squares of (T_recon - T_target) should be ~0
    # But building T_target explicitly is easy.
    T_target = np.zeros_like(T_recon)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _try_rank(r, max_iters=200, deadline=None):
    """
    Try to find a rank-r decomposition using iterative rounding / local search.
    This is a heuristic.
    """
    n, m, p = DIMENSIONS
    if deadline and time.time() > deadline - 5:
        return None
    
    # Initialize U, V, W with small random integers
    np.random.seed(42)
    U = np.random.randint(-1, 2, size=(n*m, r))
    V = np.random.randint(-1, 2, size=(m*p, r))
    W = np.random.randint(-1, 2, size=(p*n, r))
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline - 10:
            break
            
        # Compute current tensor
        T_recon = np.zeros((n*m, m*p, p*n))
        for rr in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
        
        err = np.max(np.abs(T_recon - T_target))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-6:
                # Verify exactly
                if _verify(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Perturb: pick a random entry and flip/sign
        matrix = np.random.choice([0, 1, 2])
        if matrix == 0:
            idx = np.random.randint(n*m)
            col = np.random.randint(r)
            U[idx, col] *= -1
        elif matrix == 1:
            idx = np.random.randint(m*p)
            col = np.random.randint(r)
            V[idx, col] *= -1
        else:
            idx = np.random.randint(p*n)
            col = np.random.randint(r)
            W[idx, col] *= -1
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try ranks in increasing order of difficulty (lower rank is harder)
    # Known lower bound: rank >= max(nm, mp, pn) ? No, rank >= max(n, m, p) is weak.
    # Rank >= n*m + m*p + p*n - (something)?
    # For 3x3x3, min rank is 7 (Laderman).
    # For 3x3x8, schoolbook is 72.
    # A trivial bound: rank >= max(n*m, m*p, p*n) is NOT true.
    # Actually, rank >= max(n, m, p) * max(n, m, p) ? No.
    # Let's just try a few reasonable ranks.
    
    # If we can't find anything better, return schoolbook.
    # Schoolbook rank = 72.
    
    # Can we do better than 72?
    # The problem is bilinear.
    # A simple upper bound is n*m*p = 72.
    # Can we share multiplications?
    # For 3x3, we can do 7 instead of 9.
    # For 3x3x8, maybe we can use Laderman-like tricks for each 3x3 block?
    # But the B matrix is 3x8, so it has 8 columns.
    # Each output C[i,k] is a dot product of A[i,:] and B[:,k].
    # There are 3*8=24 outputs.
    # Each output requires 3 multiplications (schoolbook).
    # Total 72.
    # If we can share, we reduce it.
    
    # Let's try rank 24 (1 multiplication per output element? No, that's impossible usually).
    # Rank 48? (2 per output?)
    # Rank 60?
    
    # Given the time constraint and complexity of finding exact integer decompositions,
    # a safe bet is to return the schoolbook solution if we don't find a quick improvement.
    # But the prompt asks for a "stronger algorithm".
    # The previous attempt tried to search for rank 24 and failed/was rejected.
    # Let's try a structured search for a slightly reduced rank, e.g., by combining terms.
    
    # However, without a guaranteed algorithm to find lower rank decompositions for this specific size,
    # the most robust "improvement" might be to ensure the schoolbook is returned quickly and correctly,
    # or to attempt a very limited search for rank < 72.
    
    # Let's attempt rank 60, 54, 48, 36, 24 with very short timeouts.
    # If any succeed, great. If not, schoolbook.
    
    ranks_to_try = [24, 36, 48, 60]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        result = _try_rank(r, max_iters=50, deadline=deadline)
        if result:
            return result
            
    return _schoolbook()
# EVOLVE-BLOCK-END
