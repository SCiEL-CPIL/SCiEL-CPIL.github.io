"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before the 120s limit
    
    # The tensor T_{i,j,k} = 1 for all i,j,k.
    # We want T = sum_r U_i * V_j * W_k (where U_i is the row of U for index i, etc.)
    # Actually, the indices are:
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    
    # We search for half-integer solutions by scaling by 2.
    # Let U = U_int / 2, V = V_int / 2, W = W_int / 2.
    # Then sum (U_int * V_int * W_int) / 8 = 1 => sum (U_int * V_int * W_int) = 8.
    
    # We will search for R from 14 upwards.
    # Theoretical lower bound for 2x3x4 is 14 (since 2*3*4=24, and Strassen-like bounds suggest lower).
    # Actually, for m*n*p matrix multiplication, the rank is at least max(mn, np, pm).
    # Here max(6, 8, 12) = 12. But 14 is a known bound for some cases.
    # Let's try from 14 to 24.
    
    # Precompute index mappings for speed
    # For each (i, j, k), we need:
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    idx_map = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_map.append((i*m+j, j*p+k, k*n+i))
    
    # We'll use a randomized local search with a smooth surrogate objective.
    # Objective: minimize sum_{i,j,k} (S_{i,j,k} - 8)^2 where S_{i,j,k} = sum_r U[u_idx][r]*V[v_idx][r]*W[w_idx][r]
    
    # To speed up, we can use numpy.
    # U_int: shape (n*m, R)
    # V_int: shape (m*p, R)
    # W_int: shape (p*n, R)
    
    # For a given R, we can try multiple random restarts.
    
    for R in range(14, 24):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this R
        # The number of restarts depends on remaining time.
        # Let's do a fixed number of restarts per R, but break early if time is up.
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers
            # Using a range of [-4, 4] to allow some flexibility
            U_int = np.random.randint(-4, 5, size=(n*m, R))
            V_int = np.random.randint(-4, 5, size=(m*p, R))
            W_int = np.random.randint(-4, 5, size=(p*n, R))
            
            # Local search
            # We want sum_r U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r] == 8 for all i,j,k
            
            # Calculate initial error
            # Error is sum of squared differences from 8
            # Or just check if all are 8.
            
            # To speed up, we can compute the current sums S[i,j,k]
            # S[i,j,k] = sum_r U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r]
            
            # Let's use a simple hill climbing on the sum of squared errors
            # But checking exactness is the goal.
            
            # We can try to satisfy one equation at a time?
            # Or just do random walks and check if all are 8.
            
            # A better heuristic:
            # Pick a random (i,j,k).
            # Calculate current sum S.
            # We want S=8.
            # Change a random term in the sum to reduce |S-8|.
            
            # However, changing one term affects multiple (i,j,k) equations.
            # U[i*m+j][r] appears in all k for fixed i,j.
            # V[j*p+k][r] appears in all i for fixed j,k.
            # W[k*n+i][r] appears in all j for fixed k,i.
            
            # Let's try a randomized local search minimizing the max absolute error.
            
            best_err = float('inf')
            best_state = None
            
            for step in range(5000): # Limit steps per restart
                if time.time() > deadline:
                    break
                    
                # Compute errors
                # S[i,j,k] = sum_r U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r]
                # We can vectorize this computation.
                # For each (i,j,k), we have u_idx, v_idx, w_idx.
                # S[i,j,k] = sum_r U_int[u_idx][r] * V_int[v_idx][r] * W_int[w_idx][r]
                
                # Let's compute all S values at once.
                # u_idxs, v_idxs, w_idxs are arrays of shape (24,)
                u_idxs = np.array([x[0] for x in idx_map])
                v_idxs = np.array([x[1] for x in idx_map])
                w_idxs = np.array([x[2] for x in idx_map])
                
                # U_int[u_idxs] has shape (24, R)
                # V_int[v_idxs] has shape (24, R)
                # W_int[w_idxs] has shape (24, R)
                # Element-wise product: (24, R)
                # Sum over r: (24,)
                
                S = np.sum(U_int[u_idxs] * V_int[v_idxs] * W_int[w_idxs], axis=1)
                errors = S - 8
                
                max_err = np.max(np.abs(errors))
                if max_err == 0:
                    # Found a solution
                    U_out = (U_int / 2.0).tolist()
                    V_out = (V_int / 2.0).tolist()
                    W_out = (W_int / 2.0).tolist()
                    # Verify
                    if _verify(U_out, V_out, W_out, n, m, p):
                        return {"U": U_out, "V": V_out, "W": W_out}
                
                if max_err < best_err:
                    best_err = max_err
                    best_state = (U_int.copy(), V_int.copy(), W_int.copy())
                
                if max_err == 0:
                    break
                    
                # Random move
                # Pick a random variable to change
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, n*m-1)
                    r = random.randint(0, R-1)
                    # Try small changes
                    delta = random.choice([-2, -1, 1, 2])
                    old_val = U_int[idx, r]
                    U_int[idx, r] = old_val + delta
                elif mat == 'V':
                    idx = random.randint(0, m*p-1)
                    r = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2])
                    old_val = V_int[idx, r]
                    V_int[idx, r] = old_val + delta
                else:
                    idx = random.randint(0, p*n-1)
                    r = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2])
                    old_val = W_int[idx, r]
                    W_int[idx, r] = old_val + delta
                
                # Check if error decreased (optional, but helps)
                # For simplicity, we just accept and let the loop handle it, 
                # or we can do a proper hill climb.
                # Let's do a simple hill climb: if error increased, revert.
                
                # Recompute max_err quickly? 
                # It's O(R*n*m*p) which is small.
                S_new = np.sum(U_int[u_idxs] * V_int[v_idxs] * W_int[w_idxs], axis=1)
                errors_new = S_new - 8
                new_max_err = np.max(np.abs(errors_new))
                
                if new_max_err > max_err:
                    # Revert
                    if mat == 'U':
                        U_int[idx, r] = old_val
                    elif mat == 'V':
                        V_int[idx, r] = old_val
                    else:
                        W_int[idx, r] = old_val
                else:
                    max_err = new_max_err
                    if new_max_err < best_err:
                        best_err = new_max_err
                        best_state = (U_int.copy(), V_int.copy(), W_int.copy())
                        
                    if new_max_err == 0:
                        U_out = (U_int / 2.0).tolist()
                        V_out = (V_int / 2.0).tolist()
                        W_out = (W_int / 2.0).tolist()
                        if _verify(U_out, V_out, W_out, n, m, p):
                            return {"U": U_out, "V": V_out, "W": W_out}
                        break

    # Fallback to schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
