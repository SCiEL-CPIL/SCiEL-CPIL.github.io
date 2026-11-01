"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the target tensor T with shape (p*n, n*m, m*p)
    # T[k*n+i, i*m+j, j*p+k] = 1 iff indices match
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Start with schoolbook
    rank = best_rank
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
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by zeroing out columns that are linearly dependent
    # Use iterative rank reduction
    deadline = time.time() + 140  # safety margin
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W decompose T"""
        # T_approx[u, v, w] = sum_r U[u, r]*V[v, r]*W[w, r]
        # u: A index (i*m+j), size n*m
        # v: B index (j*p+k), size m*p
        # w: C index (k*n+i), size p*n
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        max_err = 0.0
        # Check all possible (u, v, w)
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    # Determine expected value
                    # u = i*m + j
                    i = u // m
                    j = u % m
                    # v = j*p + k
                    k_v = v % p
                    j_v = v // p
                    # w = k*n + i
                    i_w = w % n
                    k_w = w // n
                    
                    if (j == j_v) and (k_v == k_w) and (i == i_w):
                        expected = 1.0
                    else:
                        expected = 0.0
                        
                    actual = T_approx[u, v, w]
                    err = abs(expected - actual)
                    if err > max_err:
                        max_err = err
                        
        return max_err < tol
    
    # Try rank reduction via column elimination
    for target_rank in range(best_rank - 1, 1, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try to find a rank-target_rank decomposition
        # Initialize with first target_rank columns of schoolbook
        U_try = best_U[:, :target_rank].copy()
        V_try = best_V[:, :target_rank].copy()
        W_try = best_W[:, :target_rank].copy()
        
        if verify(U_try, V_try, W_try):
            best_rank = target_rank
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            continue
        
        # Try random perturbations and optimization
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            U_try = best_U[:, :target_rank].copy()
            V_try = best_V[:, :target_rank].copy()
            W_try = best_W[:, :target_rank].copy()
                
            # Add small random perturbations
            U_try += 0.1 * np.random.randn(*U_try.shape)
            V_try += 0.1 * np.random.randn(*V_try.shape)
            W_try += 0.1 * np.random.randn(*W_try.shape)
                
            # Simple gradient-free optimization: try to minimize error
            best_err = float('inf')
            best_U_opt, best_V_opt, best_W_opt = U_try.copy(), V_try.copy(), W_try.copy()
                
            for step in range(50):
                if time.time() > deadline:
                    break
                    
                # Compute current error using verify (which is now correct)
                # But verify returns bool. We need the error value.
                # Let's compute error directly for speed in inner loop
                T_approx = np.einsum('ur,vr,wr->uvw', U_try, V_try, W_try)
                err_sum = 0.0
                for u in range(n * m):
                    for v in range(m * p):
                        for w in range(p * n):
                            i = u // m
                            j = u % m
                            k_v = v % p
                            j_v = v // p
                            i_w = w % n
                            k_w = w // n
                            if (j == j_v) and (k_v == k_w) and (i == i_w):
                                expected = 1.0
                            else:
                                expected = 0.0
                            err_sum += (T_approx[u, v, w] - expected) ** 2
                err = err_sum
                    
                if err < best_err:
                    best_err = err
                    best_U_opt, best_V_opt, best_W_opt = U_try.copy(), V_try.copy(), W_try.copy()
                    
                if err < 1e-8:
                    break
                    
                # Perturb one matrix at a time
                which = random.randint(0, 2)
                if which == 0:
                    U_try += 0.01 * np.random.randn(*U_try.shape)
                elif which == 1:
                    V_try += 0.01 * np.random.randn(*V_try.shape)
                else:
                    W_try += 0.01 * np.random.randn(*W_try.shape)
                
            if best_err < 1e-8:
                best_rank = target_rank
                best_U, best_V, best_W = best_U_opt, best_V_opt, best_W_opt
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
