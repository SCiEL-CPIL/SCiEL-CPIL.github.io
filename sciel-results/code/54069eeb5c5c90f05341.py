import time
import numpy as np
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Uses a discrete local search (Hill Climbing / Simulated Annealing) on the half-integer lattice
    to find a rank-18 decomposition. Falls back to rank 36 if not found in time.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Fallback: Schoolbook decomposition (Rank 36)
    rank_schoolbook = n * m * p
    U_fb = np.zeros((dim_u, rank_schoolbook))
    V_fb = np.zeros((dim_v, rank_schoolbook))
    W_fb = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    def to_lists(mat):
        return mat.tolist()
    
    fallback_result = {"U": to_lists(U_fb), "V": to_lists(V_fb), "W": to_lists(W_fb)}
    
    # Target Rank: Lower bound is max(6, 18, 12) = 18
    target_rank = 18
    
    # Discrete Local Search for Rank 18
    # Entries are restricted to half-integers: {0, +/-0.5, +/-1, ...}
    # We will search in the space of integers for 2*U, 2*V, 2*W
    # Let U_int, V_int, W_int be integer matrices.
    # T = (1/8) * sum U_int_r * V_int_r * W_int_r
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Helper to calculate error efficiently
    # Error = || T - 1/8 * sum(U_r \otimes V_r \otimes W_r) ||^2
    # We minimize sum( (8*T - sum(...))^2 )
    
    def calc_error(U, V, W):
        # U, V, W are integer arrays of shape (dim, rank)
        # Reconstruct tensor
        # Use einsum for speed
        # T_recon = sum_r U[:,r] V[:,r] W[:,r]
        # This is equivalent to np.einsum('ir,jr,kr->ijk', U, V, W)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # Target is 8*T
        diff = T_recon - 8 * T
        return np.sum(diff**2)
    
    # Local Search
    # We try multiple restarts
    max_restarts = 200
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random sparse half-integers
        # We try to be sparse to help convergence
        U = np.zeros((dim_u, target_rank), dtype=int)
        V = np.zeros((dim_v, target_rank), dtype=int)
        W = np.zeros((dim_w, target_rank), dtype=int)
        
        # Initialize with some non-zeros
        # Randomly place non-zeros
        for _ in range(target_rank * 2): # Sparse initialization
            mat = random.choice([U, V, W])
            r = random.randint(0, target_rank - 1)
            idx = random.randint(0, mat.shape[0] - 1)
            mat[idx, r] = random.choice([-2, -1, 1, 2]) # Values are 2*half-int, so even/odd doesn't matter for half-int, but we search integers.
            # Actually, if we search integers for 2*U, then 0.5 -> 1. 
            # So values can be any integer.
        
        current_error = calc_error(U, V, W)
        
        if current_error == 0:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
            
        # Hill Climbing
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Pick a random matrix and entry
            mat_name = random.choice(['U', 'V', 'W'])
            if mat_name == 'U':
                mat = U
            elif mat_name == 'V':
                mat = V
            else:
                mat = W
                
            r = random.randint(0, target_rank - 1)
            idx = random.randint(0, mat.shape[0] - 1)
            
            # Try a small change
            delta = random.choice([-1, 1])
            old_val = mat[idx, r]
            mat[idx, r] = old_val + delta
            
            new_error = calc_error(U, V, W)
            
            # Accept if better, or with small probability if worse (SA-like)
            # Simple hill climbing first: accept if better
            if new_error < current_error:
                current_error = new_error
                if current_error == 0:
                    break
            else:
                # Revert
                mat[idx, r] = old_val
                
            # Early stop if perfect
            if current_error == 0:
                break
        
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_error == 0:
            break
            
    if best_error == 0:
        # Convert back to half-integers
        U_final = best_U / 2.0
        V_final = best_V / 2.0
        W_final = best_W / 2.0
        
        # Verify
        T_verify = np.einsum('ir,jr,kr->ijk', U_final, V_final, W_final)
        if np.allclose(T_verify, T, atol=1e-9):
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        else:
            # Should not happen if error was 0
            pass
            
    return fallback_result
# EVOLVE-BLOCK-END
