import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    nmp = n * m * p
    
    # Schoolbook fallback
    rank_school = nmp
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    time_limit = 140  # seconds, leave 10s margin for 150s kill

    def is_half_integer(matrix, tol=1e-6):
        """Check if matrix entries are close to half-integers. Return (bool, rounded_matrix)."""
        # Convert to numpy for speed
        mat = np.array(matrix)
        # Multiply by 2 to check if integers
        doubled = mat * 2
        rounded = np.round(doubled)
        diff = np.abs(doubled - rounded)
        if np.max(diff) > tol:
            return False, None
        # Convert back to list of lists of int or float
        result = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                val = rounded[i, j] / 2.0
                if val == int(val):
                    row.append(int(val))
                else:
                    row.append(val)
            result.append(row)
        return True, result

    def verify_decomposition(U, V, W, rank):
        """Verify U, V, W exactly decompose the tensor."""
        # Use numpy for speed
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reshape
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        # Compute tensor: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Note: W is indexed by k*n+i, so W_reshaped[k,i,c]
        
        # einsum: 'ijc,jkc,kic->ijk'
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False
            
        # Check if all entries are 1
        if np.allclose(T, 1.0, atol=1e-9):
            return True
        return False

    # Search for lower ranks
    # We iterate R from schoolbook-1 down to 1
    # To be efficient, we allocate time dynamically.
    # If we find a solution, we continue to lower R.
    
    # Pre-compute indices for speed
    # eq_idx for (i,j,k)
    # w_idx for (k,i) -> k*n+i
    
    for R in range(rank_school - 1, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Estimate time per restart
        # Let's try 1000 restarts if fast, else fewer.
        # We'll use a sub-time-limit per R to ensure we check all R.
        # If R is close to 60, it's easier. If R is low, it's harder.
        # Let's give 2 seconds per R? 60 R's * 2s = 120s. Too much.
        # Let's give 1 second per R? 60s. Leaves 80s.
        # Better: allocate remaining time / remaining R's.
        remaining_time = time_limit - (time.time() - start_time)
        remaining_R = rank_school - R
        if remaining_R == 0:
            time_per_R = remaining_time
        else:
            time_per_R = remaining_time / remaining_R
        
        # Ensure at least 0.1s per R
        time_per_R = max(0.1, time_per_R)
        
        restarts_done = 0
        found_in_this_R = False
        
        while time.time() - start_time < start_time + (time_limit - time_per_R): # Stop before next R's turn
             # Actually, simpler: check time inside loop
             if time.time() - start_time > time_limit - 1.0: # 1s safety
                 break
             
             # Random U, V
             # Use {-1, 0, 1}
             U_rand = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)])
             V_rand = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)])
             
             # Build A
             # A shape: (nmp, pn * R)
             A = np.zeros((nmp, pn * R))
             
             # Fill A
             # For each c, we can vectorize the filling?
             # Or just loop i,j,k
             # nmp=60, R~50. 60*50 = 3000 ops. Fast.
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         eq_idx = i * mp + j * p + k
                         w_base = k * n + i
                         # u_row = U_rand[i*m+j]
                         # v_row = V_rand[j*p+k]
                         # coeffs = u_row * v_row
                         # A[eq_idx, w_base*R : (w_base+1)*R] = coeffs
                         u_row = U_rand[i * m + j]
                         v_row = V_rand[j * p + k]
                         coeffs = u_row * v_row
                         A[eq_idx, w_base * R : (w_base + 1) * R] = coeffs
             
             b = np.ones(nmp)
             
             try:
                 W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
             except:
                 continue
                 
             if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-5:
                 continue
                 
             W_sol = W_flat.reshape((pn, R))
             
             ok, W_nice = is_half_integer(W_sol)
             if ok:
                 if verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_nice, R):
                     best_rank = R
                     best_U = U_rand.tolist()
                     best_V = V_rand.tolist()
                     best_W = W_nice
                     found_in_this_R = True
                     # Do not break, keep searching for lower R in this time slot?
                     # Yes, but we might want to move to lower R to find even better.
                     # However, if we found R=59, we should try R=58.
                     # The outer loop handles R decrement.
                     # We can break inner loop to save time for lower R?
                     # Yes, finding one is enough to update best, and lower is better.
                     break
             restarts_done += 1

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
