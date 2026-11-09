import numpy as np
import time
import random
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, rank_schoolbook))
    V0 = np.zeros((m*p, rank_schoolbook))
    W0 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
                
    def compute_error(U, V, W, r):
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        error = np.sum((T - target) ** 2)
        return error

    err0 = compute_error(U0, V0, W0, rank_schoolbook)
    if err0 < 1e-9:
        best_error = err0
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook

    # Allowed values for entries
    allowed_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

    def try_rank(r, num_restarts=10):
        nonlocal best_U, best_V, best_W, best_rank, best_error
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.choice(allowed_vals, size=(n*m, r))
            V = np.random.choice(allowed_vals, size=(m*p, r))
            W = np.random.choice(allowed_vals, size=(p*n, r))
            
            current_err = compute_error(U, V, W, r)
            
            # Stochastic Local Search
            max_steps = 5000
            no_improve_count = 0
            threshold = 100
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                
                # Pick random matrix
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    rows, cols = n*m, r
                elif mat_idx == 1:
                    mat = V
                    rows, cols = m*p, r
                else:
                    mat = W
                    rows, cols = p*n, r
                
                # Pick random entry
                i = random.randint(0, rows - 1)
                j = random.randint(0, cols - 1)
                
                old_val = mat[i, j]
                
                # Pick random new value different from old
                new_val = old_val
                while new_val == old_val:
                    new_val = allowed_vals[random.randint(0, len(allowed_vals)-1)]
                
                mat[i, j] = new_val
                new_err = compute_error(U, V, W, r)
                
                if new_err < current_err:
                    current_err = new_err
                    no_improve_count = 0
                else:
                    mat[i, j] = old_val
                    no_improve_count += 1
                    
                if current_err < 1e-9:
                    break
                
                if no_improve_count > threshold:
                    # Restart local search or break
                    break
            
            # Check if we found a good solution
            if current_err < 1e-6:
                # Snap to nearest half-integer
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify exactness
                final_err = compute_error(U_snapped, V_snapped, W_snapped, r)
                if final_err < 1e-9:
                    if r < best_rank or (r == best_rank and final_err < best_error):
                        best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                        best_rank = r
                        best_error = final_err
                        if best_rank == 16:
                            return # Found minimal known rank, exit early? No, let's just break out of loops.
            
            if best_rank <= 16:
                break
    
    # Try rank 16 first
    if time.time() < deadline - 10:
        try_rank(16, num_restarts=20)
        
    # If rank 16 not found, try rank 17
    if best_rank > 17 and time.time() < deadline - 10:
        try_rank(17, num_restarts=20)
        
    # Ensure we have a valid fallback
    if best_U is None:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook
        best_error = err0

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                nearest = round(val * 2) / 2
                new_row.append(nearest)
            cleaned.append(new_row)
        return cleaned

    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
