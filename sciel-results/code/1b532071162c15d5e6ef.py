"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target(n, m, p):
    """Build the target matrix multiplication tensor."""
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    return target

def _compute_tensor(U, V, W):
    """Compute the decomposed tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    rank = U.shape[1]
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=float)
    chunk = max(1, min(rank, 200))
    for s in range(0, rank, chunk):
        e = min(s + chunk, rank)
        T += np.einsum('ar,br,cr->abc', U[:, s:e], V[:, s:e], W[:, s:e])
    return T

def _verify(U, V, W, target, tol=1e-6):
    """Verify decomposition matches target."""
    T = _compute_tensor(U, V, W)
    return np.max(np.abs(T - target)) < tol

def _try_strassen_blocks(n, m, p):
    """
    Try to use Strassen's algorithm on 2x2 blocks to reduce rank.
    For 4x5x6: pad to 4x6x6, use 2x2 Strassen blocks.
    The 4x6x6 matrix mult can be done with:
    - 4 blocks of 2x2 mult (using Strassen: 7 instead of 8)
    - But we need to be careful about the rectangular nature.
    
    Actually, let's think about this more carefully.
    A[i,j] * B[j,k] where A is 4x5, B is 5x6.
    
    We can't directly use 2x2 Strassen because the middle dimension is 5.
    
    Alternative: Use the fact that we can decompose the 5-dimension.
    Write j = 2*block + offset, where block in {0,1,2} (but 5 = 2*2 + 1)
    
    Let's try a different approach: use the known Strassen-based algorithm
    for rectangular matrices by padding and then using block decomposition.
    
    For simplicity, let's just try to find a valid decomposition with
    half-integer entries using a more systematic search.
    """
    pass

def _find_decomposition(n, m, p, target, max_rank=None):
    """
    Search for a minimal rank decomposition with half-integer entries.
    
    Strategy: Start from schoolbook (rank n*m*p) and try to reduce rank
    by finding linear dependencies or using known algorithms.
    
    For now, use a simple approach: try ranks from a lower bound up,
    and use random restarts with gradient-based optimization.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Lower bound for rank: max(n*m, m*p, p*n) / something
    # Actually, a better lower bound: the tensor rank is at least 
    # max(n*m, m*p, p*n) / min(n,m,p) in some cases, but let's just use
    # a reasonable lower bound.
    
    # Known results: rank(4x4x4) <= 23 (Strassen), rank(2x2x2) = 7
    # For 4x5x6, let's start searching from rank 24 upward.
    
    start_rank = max(24, min(n*m, m*p, p*n))  # rough lower bound
    
    best_U, best_V, best_W = None, None, None
    best_rank = max_rank + 1
    
    # First, get schoolbook as fallback
    U_sb, V_sb, W_sb = _make_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    rng = np.random.default_rng(42)
    
    # Try ranks from start_rank up to best_rank
    for rank in range(start_rank, best_rank + 1):
        # For each rank, try a few random restarts
        for restart in range(3):
            # Initialize with random values
            U = rng.standard_normal((nm, rank)) * 0.1
            V = rng.standard_normal((mp, rank)) * 0.1
            W = rng.standard_normal((pn, rank)) * 0.1
            
            # Try a few gradient steps
            for step in range(50):
                T = _compute_tensor(U, V, W)
                diff = T - target
                
                # Compute gradients
                dU = np.einsum('abc,br,cr->ar', diff, V, W)
                dV = np.einsum('abc,ar,cr->br', diff, U, W)
                dW = np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Learning rate (adaptive)
                lr = 0.01 / (1 + step * 0.01)
                
                # Gradient descent
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check if we've converged
                if np.max(np.abs(diff)) < 1e-8:
                    break
            
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify(U_snap, V_snap, W_snap, target, tol=1e-5):
                return U_snap, V_snap, W_snap, rank
    
    # If we couldn't find anything better, return schoolbook
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Build target
    target = _build_target(n, m, p)
    
    # Try to find a good decomposition
    U, V, W, rank = _find_decomposition(n, m, p, target)
    
    # Convert to lists with half-integer rounding
    def to_list(mat):
        rounded = np.round(np.array(mat) * 2) / 2
        return rounded.tolist()
    
    U_out = to_list(U)
    V_out = to_list(V)
    W_out = to_list(W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
