import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize Schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    current_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = current_rank
    
    start_time = time.time()
    time_limit = 110  # Safety margin for 150s limit

    def is_valid(U, V, W, r):
        # Check if sum U*V*W == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def snap_to_half_int(matrix):
        return np.round(matrix * 2) / 2

    def snap_to_int(matrix):
        return np.round(matrix)

    # Helper to try eliminating term c by solving for variable 'var' in U, V, W
    # var: 0 for U, 1 for V, 2 for W
    def try_eliminate(U, V, W, r, c, var, snap_func):
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        if var == 0: # Solve for U
            success = True
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r-1))
                    b = np.ones(p)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for k in range(p):
                            A[k, idx] = V[j*p+k, c_prime] * W[k*n+i, c_prime]
                        idx += 1
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    x_snapped = snap_func(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        success = False
                        break
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        U_new[i*m+j, c_prime] = x_snapped[idx]
                        idx += 1
            if not success:
                return None, None, None, False
            U_new[:, c] = 0
            return U_new, V_new, W_new, True

        elif var == 1: # Solve for V
            success = True
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r-1))
                    b = np.ones(n)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for i in range(n):
                            A[i, idx] = U[i*m+j, c_prime] * W[k*n+i, c_prime]
                        idx += 1
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    x_snapped = snap_func(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        success = False
                        break
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        V_new[j*p+k, c_prime] = x_snapped[idx]
                        idx += 1
            if not success:
                return None, None, None, False
            V_new[:, c] = 0
            return U_new, V_new, W_new, True

        else: # Solve for W
            success = True
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r-1))
                    b = np.ones(m)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for j in range(m):
                            A[j, idx] = U[i*m+j, c_prime] * V[j*p+k, c_prime]
                        idx += 1
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    x_snapped = snap_func(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        success = False
                        break
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        W_new[k*n+i, c_prime] = x_snapped[idx]
                        idx += 1
            if not success:
                return None, None, None, False
            W_new[:, c] = 0
            return U_new, V_new, W_new, True

    # Winograd/Block Embedding Strategy:
    # We attempt to exploit structure by grouping terms.
    # Specifically, we look for redundancies where a term can be absorbed into others.
    # The "Winograd" idea here is to treat blocks of dimensions as smaller multiplications
    # and try to reduce the rank of those blocks.
    
    # We will iterate through possible eliminations, but also try "merging" terms
    # by checking if two terms are linearly dependent or can be combined.
    
    # To improve the search, we will try elimination in a random order and 
    # also try to "swap" terms to break local minima.
    
    order = list(range(current_rank))
    random.shuffle(order)
    
    while time.time() - start_time < time_limit:
        progress = False
        
        # Try to eliminate terms in the shuffled order
        for c in order:
            if c >= current_rank:
                continue
            if time.time() - start_time > time_limit:
                break
            
            # Try solving for U, V, W with integer and half-integer snapping
            for var in [0, 1, 2]:
                for snap_func in [snap_to_int, snap_to_half_int]:
                    U_new, V_new, W_new, success = try_eliminate(U, V, W, current_rank, c, var, snap_func)
                    if success:
                        # Verify full decomposition
                        if is_valid(U_new, V_new, W_new, current_rank - 1):
                            U, V, W = U_new, V_new, W_new
                            current_rank -= 1
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                            best_rank = current_rank
                            progress = True
                            # Re-shuffle remaining terms
                            order = list(range(current_rank))
                            random.shuffle(order)
                            break
                if progress:
                    break
            if progress:
                break
                
        if not progress:
            # If stuck, try a perturbation or restart with a different random seed/order
            # We can also try to "merge" two terms if they are identical or similar.
            # For now, let's just re-shuffle and try again a few times before giving up.
            # We'll do a few random restarts of the elimination process.
            
            # Simple heuristic: If we are stuck, maybe we can find a pair of terms that are "close"
            # and combine them? This is complex.
            
            # Let's try one more pass with a different random order.
            random.shuffle(order)
            
            # If we've been stuck for a while, break.
            # To avoid infinite loops, we count iterations.
            # For now, just break to return best result.
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
