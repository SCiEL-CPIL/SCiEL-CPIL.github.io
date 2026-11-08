"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

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
    T_res = np.zeros_like(T)
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    def is_zero(mat, tol=1e-9):
        return np.all(np.abs(mat) < tol)

    def update_best():
        nonlocal best_rank, best_U, best_V, best_W
        if len(active_cols) < best_rank:
            best_rank = len(active_cols)
            best_U = U[:, active_cols].copy()
            best_V = V[:, active_cols].copy()
            best_W = W[:, active_cols].copy()

    # Phase 1: Greedy single-column removal
    max_passes = 100
    pass_count = 0
    no_progress_count = 0
    
    while time.time() < deadline and pass_count < max_passes and no_progress_count < 10:
        pass_count += 1
        improved_in_pass = False
        
        if pass_count > 1:
            np.random.shuffle(active_cols)
        
        i = 0
        while i < len(active_cols):
            if time.time() > deadline - 5:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            c = active_cols[i]
            term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            new_T_res = T_res + term
            
            if is_zero(new_T_res, tol=1e-9):
                active_cols.pop(i)
                T_res = new_T_res
                improved_in_pass = True
                update_best()
            else:
                i += 1
        
        if improved_in_pass:
            no_progress_count = 0
        else:
            no_progress_count += 1

    # Phase 2: Try to find 2-column or 3-column replacements
    # For each subset of active columns, check if we can replace them with fewer columns
    # This is expensive, so we limit the search
    
    if time.time() < deadline - 20:
        # Try pairwise replacement: can two columns be replaced by one?
        # Or can we find a linear combination that reduces rank?
        
        # Strategy: For small subsets, try to find if the span can be reduced
        # We'll try to find if any column is in the span of others (over the tensor product space)
        
        # More practical: try random linear combinations
        # For each pair of columns, see if a random linear combo works better
        
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            if len(active_cols) < 2:
                break
            
            # Pick two random columns to try to merge
            idx1, idx2 = np.random.choice(len(active_cols), size=2, replace=False)
            c1 = active_cols[idx1]
            c2 = active_cols[idx2]
            
            # Try replacing both with a single column that's a random linear combination
            # The new column should satisfy: U_new = a*U_c1 + b*U_c2, etc.
            # And the residual should still be zero
            
            # Actually, let's try a different approach:
            # Remove one column and see if we can adjust another to compensate
            
            # For now, let's just do more shuffling and single removals with different random seeds
            np.random.shuffle(active_cols)
            
            # Quick pass
            i = 0
            while i < len(active_cols):
                c = active_cols[i]
                term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                new_T_res = T_res + term
                
                if is_zero(new_T_res, tol=1e-9):
                    active_cols.pop(i)
                    T_res = new_T_res
                    update_best()
                else:
                    i += 1

    # Phase 3: Final cleanup - try all columns one more time in multiple orders
    for shuffle_iter in range(20):
        if time.time() > deadline - 5:
            break
        
        np.random.shuffle(active_cols)
        
        i = 0
        while i < len(active_cols):
            c = active_cols[i]
            term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            new_T_res = T_res + term
            
            if is_zero(new_T_res, tol=1e-9):
                active_cols.pop(i)
                T_res = new_T_res
                update_best()
            else:
                i += 1

    # Return the best solution found
    if best_rank < len(active_cols):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U[:, active_cols].tolist(), "V": V[:, active_cols].tolist(), "W": W[:, active_cols].tolist()}
# EVOLVE-BLOCK-END
