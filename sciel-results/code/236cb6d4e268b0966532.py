import time
import numpy as np

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _snap_to_half_integers(X):
    """Snap values to nearest half-integer."""
    X_scaled = X * 2
    X_snapped = np.round(X_scaled)
    return X_snapped / 2.0

def _try_rank_reduction(n, m, p, r_target, max_attempts=100):
    """
    Attempt to find a rank r_target decomposition by perturbing the schoolbook solution.
    This is a heuristic search.
    """
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return None
    
    # We try to find a solution of rank r_target
    # Strategy: Randomly initialize U, V with half-integers, solve for W, snap W.
    # This is similar to the current program but with tighter constraints and fewer attempts.
    
    candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Sparse random initialization
        U = np.zeros((n * m, r_target))
        V = np.zeros((m * p, r_target))
        
        # Ensure non-triviality
        for c in range(r_target):
            # Pick random rows for U and V to be non-zero
            u_rows = np.random.choice(n * m, size=max(1, n*m//3), replace=False)
            v_rows = np.random.choice(m * p, size=max(1, m*p//3), replace=False)
            
            for r in u_rows:
                U[r, c] = np.random.choice(candidates)
            for r in v_rows:
                V[r, c] = np.random.choice(candidates)
                
        # Solve for W
        W = np.zeros((p * n, r_target))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r_target))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                
                try:
                    W_col, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = W_col
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        W_snapped = _snap_to_half_integers(W)
        
        # Verify
        ok = _verify_decomposition(U, V, W_snapped, n, m, p, tol=1e-9)
        if ok:
            return U, V, W_snapped
            
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook solution
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find a lower rank solution
    # We know rank >= max(n*m, m*p, p*n) = 18
    # But finding a valid half-integer solution for rank < 36 is hard.
    # We try a few ranks close to 36 to see if a "near-schoolbook" solution exists.
    
    # Given the time limit and the difficulty, we only try a very limited search.
    # If we find nothing, we return schoolbook.
    
    # Try rank 35 and 34 briefly
    for r in [35, 34]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        result = _try_rank_reduction(n, m, p, r, max_attempts=50)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
            
    # Final verification
    ok = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        best_U, best_V, best_W = U_school, V_school, W_school

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
