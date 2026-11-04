"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _build_decomposition(n, m, p, indices):
    """Build U, V, W from a list of (i,j,k) tuples."""
    rank = len(indices)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c, (i, j, k) in enumerate(indices):
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    # Compute tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    
    # Expected: T[i,j,k] = 1 if i,j,k form a valid multiplication term, else 0
    # For matrix multiplication (n x m) * (m x p) -> (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] should be 1 for all valid (i,j,k) combinations
    # Actually, the standard decomposition has T[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-10)

def _try_rank_reduction(n, m, p, target_rank, time_limit=100.0):
    """Attempt to find a decomposition of the given rank."""
    start_time = time.time()
    full_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    full_rank = len(full_indices)
    
    if target_rank >= full_rank:
        return _build_decomposition(n, m, p, full_indices)
    
    # Try to find a subset of indices that still works
    # This is a combinatorial problem, so we'll use a greedy approach
    # Start with all indices and try to remove some
    
    best_result = None
    best_rank = full_rank
    
    # Simple heuristic: try removing indices that might be redundant
    # For matrix multiplication, we can sometimes reduce rank by exploiting structure
    
    # Try various rank values from target_rank up to full_rank
    for r in range(target_rank, full_rank + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find r indices that work
        # Use a simple random sampling approach
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            if r == full_rank:
                indices = full_indices
            else:
                # Random subset
                indices = full_indices[:r]
            
            U, V, W = _build_decomposition(n, m, p, indices)
            if _verify_decomposition(U, V, W, n, m, p):
                if r < best_rank:
                    best_rank = r
                    best_result = (U, V, W)
                return U, V, W
    
    if best_result is not None:
        return best_result
    return _build_decomposition(n, m, p, full_indices)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start with the known schoolbook decomposition (rank = n*m*p = 72)
    full_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    U, V, W = _build_decomposition(n, m, p, full_indices)
    
    # Try to find a lower rank decomposition
    # For 3x3x8, known lower bounds suggest rank might be reducible
    # Let's try a few different target ranks
    
    target_ranks = [60, 55, 50, 45, 40]
    for target in target_ranks:
        if time.time() > deadline:
            break
        
        try:
            U_new, V_new, W_new = _try_rank_reduction(n, m, p, target, time_limit=10.0)
            if U_new.shape[1] < U.shape[1]:
                U, V, W = U_new, V_new, W_new
        except Exception:
            continue
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
