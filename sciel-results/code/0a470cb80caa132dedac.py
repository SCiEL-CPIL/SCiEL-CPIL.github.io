"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    Uses randomized local search to find lower-rank decompositions.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=float)
        V = np.zeros((m * p, r), dtype=float)
        W = np.zeros((p * n, r), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r
    
    U, V, W, best_rank = schoolbook()
    
    def error(U, V, W):
        """Compute the error in the decomposition."""
        # For each (i,j,k), check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        return err
    
    def snap_to_half(x):
        """Snap values to nearest half-integer."""
        return np.round(x * 2) / 2.0
    
    def try_reduce(U, V, W, r, max_iter=5000):
        """Try to reduce rank by removing one column and re-optimizing."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Try removing each column
        for col_to_remove in range(r):
            if time.time() > deadline:
                break
                
            # Save state
            U_orig = U.copy()
            V_orig = V.copy()
            W_orig = W.copy()
            
            # Remove the column
            U_new = np.delete(U, col_to_remove, axis=1)
            V_new = np.delete(V, col_to_remove, axis=1)
            W_new = np.delete(W, col_to_remove, axis=1)
            
            # Now try to optimize the remaining r-1 columns
            # Use gradient-free local search
            r_new = r - 1
            best_err = error(U_new, V_new, W_new)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Random perturbation
                scale = 0.1 * (1 - it / max_iter) + 0.01
                U_pert = U_new + np.random.randn(*U_new.shape) * scale
                V_pert = V_new + np.random.randn(*V_new.shape) * scale
                W_pert = W_new + np.random.randn(*W_new.shape) * scale
                
                err_pert = error(U_pert, V_pert, W_pert)
                
                if err_pert < best_err:
                    U_new, V_new, W_new = U_pert, V_pert, W_pert
                    best_err = err_pert
                    
                    if best_err < 1e-10:
                        # Found exact solution, snap to half-integers
                        U_snap = snap_to_half(U_new)
                        V_snap = snap_to_half(V_new)
                        W_snap = snap_to_half(W_new)
                        if error(U_snap, V_snap, W_snap) < 1e-8:
                            return U_snap, V_snap, W_snap, r_new
            # End iterations
            
            # If we got close, try snapping
            if best_err < 1e-6:
                U_snap = snap_to_half(U_new)
                V_snap = snap_to_half(V_new)
                W_snap = snap_to_half(W_new)
                if error(U_snap, V_snap, W_snap) < 1e-8:
                    return U_snap, V_snap, W_snap, r_new
        
        return U_orig, V_orig, W_orig, r
    
    # Try to reduce from rank 60
    current_U, current_V, current_W, current_rank = U, V, W, best_rank
    
    while current_rank > 10:  # Don't go too low
        if time.time() > deadline:
            break
            
        new_U, new_V, new_W, new_rank = try_reduce(current_U, current_V, current_W, current_rank, max_iter=2000)
        
        if new_rank < current_rank:
            current_U, current_V, current_W, current_rank = new_U, new_V, new_W, new_rank
        else:
            # Try a different strategy: random restart with lower rank
            if time.time() > deadline:
                break
            
            # Try random initialization with current_rank - 1
            for attempt in range(50):
                if time.time() > deadline:
                    break
                    
                r_try = current_rank - 1
                U_rand = np.random.randn(n*m, r_try) * 0.5
                V_rand = np.random.randn(m*p, r_try) * 0.5
                W_rand = np.random.randn(p*n, r_try) * 0.5
                
                # Local optimization
                best_err = error(U_rand, V_rand, W_rand)
                for it in range(1000):
                    if time.time() > deadline:
                        break
                    scale = 0.1 * (1 - it / 1000) + 0.01
                    U_pert = U_rand + np.random.randn(*U_rand.shape) * scale
                    V_pert = V_rand + np.random.randn(*V_rand.shape) * scale
                    W_pert = W_rand + np.random.randn(*W_rand.shape) * scale
                    err_pert = error(U_pert, V_pert, W_pert)
                    if err_pert < best_err:
                        U_rand, V_rand, W_rand = U_pert, V_pert, W_pert
                        best_err = err_pert
                        if best_err < 1e-10:
                            break
                
                if best_err < 1e-6:
                    U_snap = snap_to_half(U_rand)
                    V_snap = snap_to_half(V_rand)
                    W_snap = snap_to_half(W_rand)
                    if error(U_snap, V_snap, W_snap) < 1e-8:
                        current_U, current_V, current_W, current_rank = U_snap, V_snap, W_snap, r_try
                        break
        
        if time.time() > deadline:
            break
    
    # Final check: verify the solution
    final_err = error(current_U, current_V, current_W)
    if final_err > 1e-6:
        # Fall back to schoolbook
        U, V, W, best_rank = schoolbook()
    else:
        U, V, W, best_rank = current_U, current_V, current_W, current_rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
