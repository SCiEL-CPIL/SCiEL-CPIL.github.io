"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i, j, :], V_r[j, k, :]) * 1.0  # This isn't right
                # Actually: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=100.0):
    """
    Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
    We look for a subset of columns that spans the same tensor.
    """
    start_time = time.time()
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Strategy: Try to find a smaller set of columns that still spans the tensor.
    # We can use a greedy approach: start with all columns, try removing one at a time.
    # If removal doesn't break the decomposition, keep it removed.
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    active_cols = list(range(rank))
    
    # Precompute for faster verification
    # For a given set of active columns, check if the tensor is still 1 everywhere
    
    def check_tensor(U_c, V_c, W_c, cols, n, m, p, tol=1e-8):
        """Check if the decomposition with only columns in 'cols' is valid."""
        if not cols:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in cols:
                        val += U_c[i*m+j, c] * V_c[j*p+k, c] * W_c[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Greedy: try to remove each column one at a time
    # This is O(rank^2 * n*m*p) which is 75^2 * 75 = ~421875 ops, fast enough
    for _pass in range(3):  # Multiple passes to catch cascading removals
        if time.time() - start_time > max_time - 10:
            break
        removed_any = False
        for c in list(active_cols):
            if time.time() - start_time > max_time - 10:
                break
            test_cols = [x for x in active_cols if x != c]
            if check_tensor(current_U, current_V, current_W, test_cols, n, m, p):
                active_cols = test_cols
                removed_any = True
                best_rank = len(active_cols)
                best_U = current_U[:, active_cols].copy()
                best_V = current_V[:, active_cols].copy()
                best_W = current_W[:, active_cols].copy()
        if not removed_any:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Start with the trivial decomposition
    U, V, W = _trivial_decomposition(n, m, p)
    current_rank = U.shape[1]
    
    # Verify the trivial decomposition is valid
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Trivial decomposition failed verification")
    
    # Try to reduce rank if we have time
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 20.0:
        U, V, W, new_rank = _try_reduce_rank(U, V, W, n, m, p, max_time=min(remaining_time, 120.0))
        if new_rank < current_rank:
            current_rank = new_rank
    
    # Final safety check
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return the trivial decomposition as a fallback
        U, V, W = _trivial_decomposition(n, m, p)
    
    # Convert to lists for the required interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
