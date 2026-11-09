import numpy as np
import time
import random
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

    def try_rank(r, num_trials=5000):
        nonlocal best_U, best_V, best_W, best_rank, best_error
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Random sparse U and V with entries in {-1, 0, 1}
            # Density ~ 0.3 to keep sparsity but ensure connectivity
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            for i in range(n*m):
                for c in range(r):
                    if random.random() < 0.3:
                        U[i, c] = random.choice([-1, 1])
                        
            for j in range(m*p):
                for c in range(r):
                    if random.random() < 0.3:
                        V[j, c] = random.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((p*n, r))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Variables: W[k*n+i, r] for r in 0..R-1
                    # Equations: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for j in 0..m-1
                    # Matrix A is m x r
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    
                    # Solve A * w = 1
                    # Use lstsq for robustness, then check consistency
                    try:
                        w_sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if the solution is close to half-integers
                    # Snap to nearest half-integer
                    w_snapped = np.round(w_sol * 2) / 2
                    
                    # Check if the snapped solution satisfies the equations exactly (within float tolerance)
                    # Residual of snapped solution
                    res_snapped = np.sum((A @ w_snapped - np.ones(m))**2)
                    
                    if res_snapped < 1e-6:
                        W[k*n+i, :] = w_snapped
                    else:
                        valid = False
                        break
                        
                if not valid:
                    break
            
            if valid:
                # Final verification
                err = compute_error(U, V, W, r)
                if err < 1e-6:
                    # Snap all to half-integers just in case
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    
                    final_err = compute_error(U_snapped, V_snapped, W_snapped, r)
                    if final_err < 1e-9:
                        if r < best_rank or (r == best_rank and final_err < best_error):
                            best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                            best_rank = r
                            best_error = final_err
                            if best_rank == 16:
                                return # Found minimal known rank, exit early

    # Try rank 16 first
    if time.time() < deadline - 10:
        try_rank(16, num_trials=10000)
        
    # If rank 16 not found, try rank 17
    if best_rank > 17 and time.time() < deadline - 10:
        try_rank(17, num_trials=10000)
        
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
