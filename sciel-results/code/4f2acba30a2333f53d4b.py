"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Trivial solution as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return mat.tolist()

    # Target rank to search for
    target_rank = 60
    
    start_time = time.time()
    deadline = start_time + 130 # Safety margin
    
    # Precompute target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T_target[a * m + b, b * p + c, c * n + a] = 1.0

    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 10
    steps_per_restart = 1000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Calculate initial error
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        current_error = np.sum((T_approx - T_target) ** 2)
        
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_error < 1e-9:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

        # Local Search
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            r = random.randint(0, target_rank - 1)
            
            u_old = U[:, r].copy()
            v_old = V[:, r].copy()
            w_old = W[:, r].copy()
            
            # Perturb
            U[:, r] += np.random.randn(n * m) * 0.1
            V[:, r] += np.random.randn(m * p) * 0.1
            W[:, r] += np.random.randn(p * n) * 0.1
            
            # Calculate new error
            T_approx_new = np.einsum('ir,jr,kr->ijk', U, V, W)
            new_error = np.sum((T_approx_new - T_target) ** 2)
            
            if new_error < current_error:
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    if best_error < 1e-9:
                        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            else:
                # Revert
                U[:, r] = u_old
                V[:, r] = v_old
                W[:, r] = w_old

    # If we found a good continuous solution, try to snap it
    if best_U is not None:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        T_approx_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
        error_snap = np.sum((T_approx_snap - T_target) ** 2)
        if error_snap < 1e-9:
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
            
    # Fallback to trivial
    return {
        "U": to_list(U_trivial),
        "V": to_list(V_trivial),
        "W": to_list(W_trivial)
    }
# EVOLVE-BLOCK-END
