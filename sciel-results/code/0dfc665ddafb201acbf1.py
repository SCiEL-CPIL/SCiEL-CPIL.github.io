"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def _local_search(U, V, W, n, m, p, deadline, max_iter=500):
    """
    Local search: try to reduce rank by merging two terms into one.
    Two terms c1, c2 can be merged if there exist scalars such that
    the sum U1*V1*W1 + U2*V2*W2 can be replaced by a single term.
    This is generally not possible, but we can try perturbations.
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = best_U.shape[1]
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        current_rank = best_U.shape[1]
        if current_rank <= 1:
            break
            
        # Try to remove a random column and see if we can adjust others
        col_to_try = np.random.randint(0, current_rank)
        
        # Create a copy without this column
        U_test = np.delete(best_U, col_to_try, axis=1)
        V_test = np.delete(best_V, col_to_try, axis=1)
        W_test = np.delete(best_W, col_to_try, axis=1)
        
        # Check validity
        if _is_valid(U_test, V_test, W_test, n, m, p):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank -= 1
            continue
        
        # Try a perturbation: modify a random entry in a random column
        # and check if validity is maintained
        if current_rank > 1:
            col_mod = np.random.randint(0, current_rank)
            row_U = np.random.randint(0, n * m)
            row_V = np.random.randint(0, m * p)
            row_W = np.random.randint(0, p * n)
            
            U_test = best_U.copy()
            V_test = best_V.copy()
            W_test = best_W.copy()
            
            # Try to zero out one entry and adjust another
            U_test[row_U, col_mod] = 0.0
            
            if _is_valid(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
    
    return best_U, best_V, best_W

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start from schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try local search
    if U is not None:
        reduced_U, reduced_V, reduced_W = _local_search(U, V, W, n, m, p, deadline, max_iter=200)
        if reduced_U.shape[1] < best_rank:
            best_U, best_V, best_W = reduced_U, reduced_V, reduced_W
            best_rank = best_U.shape[1]
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def construct():
    return _construct()
# EVOLVE-BLOCK-END
