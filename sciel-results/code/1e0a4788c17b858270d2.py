"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # 1. Initialize with Schoolbook Decomposition (Rank = n*m*p)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1

    # Build Target Tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Current active columns
    active_cols = list(range(rank))
    
    # Initialize Residual: T_res = T - sum(U_c ^ V_c ^ W_c for c in active)
    # Since we start with schoolbook, T_res should be 0.
    # But to be safe and allow for potential future modifications, we compute it.
    # Actually, for schoolbook, T_res is exactly 0.
    T_res = np.zeros_like(T)
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    # Helper to check if a matrix is close to zero
    def is_zero(mat, tol=1e-9):
        return np.all(np.abs(mat) < tol)

    # Greedy Loop
    # We try to remove columns one by one.
    # To avoid getting stuck in a specific order, we can shuffle the active list 
    # if we make no progress in a full pass.
    
    max_passes = 50 # Limit passes to prevent infinite loops if no progress
    pass_count = 0
    improved_in_pass = False

    while time.time() < deadline and pass_count < max_passes:
        pass_count += 1
        improved_in_pass = False
        
        # Shuffle active columns to break symmetry and escape local minima
        # We only shuffle if we haven't improved in the previous pass
        if pass_count > 1 and not improved_in_pass:
            np.random.shuffle(active_cols)
            # If we've been stuck for a while, maybe restart with a different random seed?
            # For now, shuffling is usually sufficient for this type of problem.
        
        current_len = len(active_cols)
        if current_len == 0:
            break
            
        # Iterate through a copy of active_cols because we might remove items
        # Note: We iterate in the order of active_cols.
        # If we remove an item, we update active_cols and T_res.
        
        # To make it efficient, we iterate over indices in the current active list
        # but we need to be careful with index shifting.
        # It's safer to iterate over a snapshot of the current active columns.
        
        # However, removing a column changes the indices.
        # Let's use a different strategy: 
        # We maintain a list of column indices.
        # We check each column in the list.
        # If removed, we take it out of the list.
        
        i = 0
        while i < len(active_cols):
            if time.time() > deadline:
                return {"U": best_U[:, active_cols].tolist(), 
                        "V": best_V[:, active_cols].tolist(), 
                        "W": best_W[:, active_cols].tolist()}
            
            c = active_cols[i]
            
            # Check if column c can be removed
            # T_res currently = T - sum_{r in active} U_r ^ V_r ^ W_r
            # If we remove c, new_T_res = T_res + U_c ^ V_c ^ W_c
            # If new_T_res is 0, then c is redundant.
            
            term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            new_T_res = T_res + term
            
            if is_zero(new_T_res, tol=1e-9):
                # Remove column c
                active_cols.pop(i)
                T_res = new_T_res # Update residual
                improved_in_pass = True
                # Do NOT increment i, because the next column has shifted into position i
                # Also update best solution
                best_rank = len(active_cols)
                best_U = U[:, active_cols].copy()
                best_V = V[:, active_cols].copy()
                best_W = W[:, active_cols].copy()
            else:
                i += 1
                
        # End of pass
        if not improved_in_pass:
            # If we made no progress in a full pass, we might be stuck.
            # We can try shuffling and continuing, or break if we've done many passes.
            # The loop condition handles the break.
            pass

    # Final check: ensure we return the best found
    # The arrays U, V, W in the return should correspond to the active columns
    # We sliced them in best_U etc.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
