"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(S, 1.0, atol=1e-6)

    def calculate_error_fast(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.sum((S - 1.0) ** 2)

    def discrete_local_search(U, V, W, max_iter=10000, val_range=1.0):
        """
        Perform local search on discrete half-integer values.
        """
        current_err = calculate_error_fast(U, V, W)
        if current_err < 1e-6:
            return U, V, W, current_err
        
        # Allowed values: multiples of 0.5 in [-val_range, val_range]
        num_steps = int(val_range * 2)
        allowed_vals = np.arange(-val_range, val_range + 0.5, 0.5)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
                
            i = random.randint(0, mat.shape[0] - 1)
            j = random.randint(0, mat.shape[1] - 1)
            
            old_val = mat[i, j]
            
            # Try a random neighbor or random value from allowed set
            # To encourage movement, pick a random index from allowed_vals
            # but ensure it's different
            if len(allowed_vals) > 1:
                new_val = random.choice(allowed_vals)
                if new_val == old_val:
                    continue
            else:
                continue

            mat[i, j] = new_val
            new_err = calculate_error_fast(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-6:
                    return U, V, W, current_err
            else:
                # Revert
                mat[i, j] = old_val
                
        return U, V, W, current_err

    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb

    # Try ranks from the theoretical lower bound upwards
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, rank_sb):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random half-integers
            # Start with small range
            init_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            U = np.random.choice(init_vals, size=(n * m, target_rank))
            V = np.random.choice(init_vals, size=(m * p, target_rank))
            W = np.random.choice(init_vals, size=(p * n, target_rank))
            
            # Run local search with increasing value range if needed
            # First pass: range 1.0
            U, V, W, err = discrete_local_search(U, V, W, max_iter=5000, val_range=1.0)
            
            if err >= 1e-6:
                # Second pass: range 2.0
                U, V, W, err = discrete_local_search(U, V, W, max_iter=5000, val_range=2.0)
            
            if err >= 1e-6:
                # Third pass: range 3.0
                U, V, W, err = discrete_local_search(U, V, W, max_iter=5000, val_range=3.0)

            if err < 1e-6:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break

    if best_rank == rank_sb:
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    else:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
