import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # 15-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def compute_error(U, V, W):
        """Compute the max absolute error of the decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.max(np.abs(T_rec - T))
    
    def try_structured_search(r, max_attempts=3000):
        """
        Try to find a rank-r decomposition using structured patterns
        that exploit the (i,j,k) index structure.
        """
        if time.time() > deadline:
            return None
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + attempt * 13)
            
            # Strategy 1: Block-structured with shared indices
            # Each rank term covers a subset of (i,j,k) triples
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            if attempt % 4 == 0:
                # Dense random initialization with small integers
                U = np.random.randint(0, 3, (nm, r)) - 1
                V = np.random.randint(0, 3, (mp, r)) - 1
                W = np.random.randint(0, 3, (pn, r)) - 1
            elif attempt % 4 == 1:
                # Sparse: each column has exactly one non-zero per matrix
                for r_idx in range(r):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U[i * m + j, r_idx] = np.random.choice([-1, 1])
                    V[j * p + k, r_idx] = np.random.choice([-1, 1])
                    W[k * n + i, r_idx] = np.random.choice([-1, 1])
            elif attempt % 4 == 2:
                # Grouped: each column covers multiple k values for fixed (i,j)
                for r_idx in range(r):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    U[i * m + j, r_idx] = 1
                    # Choose a random subset of k values
                    k_count = np.random.randint(1, min(3, p))
                    k_values = np.random.choice(p, size=k_count, replace=False)
                    for k in k_values:
                        V[j * p + k, r_idx] = 1
                        W[k * n + i, r_idx] = 1
            else:
                # Hybrid: mix of single and grouped terms
                for r_idx in range(r):
                    if np.random.random() < 0.5:
                        # Single term
                        i = np.random.randint(0, n)
                        j = np.random.randint(0, m)
                        k = np.random.randint(0, p)
                        U[i * m + j, r_idx] = np.random.choice([-1, 1])
                        V[j * p + k, r_idx] = np.random.choice([-1, 1])
                        W[k * n + i, r_idx] = np.random.choice([-1, 1])
                    else:
                        # Grouped term
                        i = np.random.randint(0, n)
                        j = np.random.randint(0, m)
                        U[i * m + j, r_idx] = 1
                        k_count = np.random.randint(2, min(4, p))
                        k_values = np.random.choice(p, size=k_count, replace=False)
                        for k in k_values:
                            V[j * p + k, r_idx] = np.random.choice([-1, 1])
                            W[k * n + i, r_idx] = np.random.choice([-1, 1])
            
            error = compute_error(U, V, W)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                return (U, V, W)
            
            # Local refinement: try flipping individual entries
            if error < 0.5 and attempt % 5 == 0:
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random entry to modify
                    mat_choice = np.random.randint(0, 3)
                    if mat_choice == 0:
                        row, col = np.random.randint(0, nm), np.random.randint(0, r)
                        old_val = U[row, col]
                        # Cycle through possible values
                        U[row, col] = (old_val + 1) % 3 - 1
                    elif mat_choice == 1:
                        row, col = np.random.randint(0, mp), np.random.randint(0, r)
                        old_val = V[row, col]
                        V[row, col] = (old_val + 1) % 3 - 1
                    else:
                        row, col = np.random.randint(0, pn), np.random.randint(0, r)
                        old_val = W[row, col]
                        W[row, col] = (old_val + 1) % 3 - 1
                    
                    new_error = compute_error(U, V, W)
                    
                    if new_error < error:
                        error = new_error
                        if error < 1e-8:
                            return (U, V, W)
                    else:
                        # Revert
                        if mat_choice == 0:
                            U[row, col] = old_val
                        elif mat_choice == 1:
                            V[row, col] = old_val
                        else:
                            W[row, col] = old_val
        
        if best_U is not None and check_decomposition(best_U, best_V, best_W):
            return (best_U, best_V, best_W)
        
        return None
    
    def try_linear_dependency_merging(start_U, start_V, start_W, target_rank, max_iters=1000):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        Start from a valid decomposition and try to merge terms.
        """
        if time.time() > deadline:
            return None
        
        current_U = start_U.copy()
        current_V = start_V.copy()
        current_W = start_W.copy()
        current_rank = current_U.shape[1]
        
        if current_rank <= target_rank:
            return (current_U, current_V, current_W)
        
        best_error = compute_error(current_U, current_V, current_W)
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if current_rank <= target_rank:
                break
            
            # Try to find a linear combination of two terms that can be replaced
            # by a single term with the same effect
            
            # Pick two random columns to try to merge
            idx1 = np.random.randint(0, current_rank)
            idx2 = np.random.randint(0, current_rank)
            if idx1 == idx2:
                continue
            
            # Try to find coefficients a, b such that:
            # a * u1 + b * u2 = u_new (and similarly for v, w)
            # This is a system of linear equations
            
            u1, u2 = current_U[:, idx1], current_U[:, idx2]
            v1, v2 = current_V[:, idx1], current_V[:, idx2]
            w1, w2 = current_W[:, idx1], current_W[:, idx2]
            
            # Try small rational coefficients
            found_merge = False
            for a_num in range(-3, 4):
                for b_num in range(-3, 4):
                    if a_num == 0 and b_num == 0:
                        continue
                    
                    # Check if a*u1 + b*u2 has a simple pattern
                    u_new = a_num * u1 + b_num * u2
                    v_new = a_num * v1 + b_num * v2
                    w_new = a_num * w1 + b_num * w2
                    
                    # Check if this new term, when added to the remaining terms,
                    # still reconstructs T
                    test_U = current_U.copy()
                    test_V = current_V.copy()
                    test_W = current_W.copy()
                    
                    # Remove the two terms and add the new one
                    # This is complex; skip for now and use a different approach
            
            # Alternative: try to zero out one term by adjusting others
            # This is also complex; use a simpler heuristic
            
            # Instead, try a local search on the current decomposition
            if current_rank > target_rank:
                # Try to make one column all zeros by adjusting others
                col_to_zero = np.random.randint(0, current_rank)
                
                # Compute the contribution of this column
                contrib = np.einsum('a,b,c->abc', 
                                    current_U[:, col_to_zero], 
                                    current_V[:, col_to_zero], 
                                    current_W[:, col_to_zero])
                
                # Try to distribute this contribution to other columns
                other_cols = [c for c in range(current_rank) if c != col_to_zero]
                if len(other_cols) == 0:
                    continue
                
                # Pick a random other column to absorb the contribution
                target_col = other_cols[np.random.randint(0, len(other_cols))]
                
                # This is approximate; just try adding the contribution
                old_U = current_U[:, target_col].copy()
                old_V = current_V[:, target_col].copy()
                old_W = current_W[:, target_col].copy()
                
                # Scale the contribution and add
                scale = np.random.choice([0.5, 1.0, 1.5, 2.0])
                current_U[:, target_col] += scale * current_U[:, col_to_zero]
                current_V[:, target_col] += scale * current_V[:, col_to_zero]
                current_W[:, target_col] += scale * current_W[:, col_to_zero]
                
                # Zero out the original column
                current_U[:, col_to_zero] = 0
                current_V[:, col_to_zero] = 0
                current_W[:, col_to_zero] = 0
                
                new_error = compute_error(current_U, current_V, current_W)
                
                if new_error < best_error + 1e-10:
                    best_error = new_error
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    
                    # Remove the zero column
                    keep = [c for c in range(current_rank) if c != col_to_zero]
                    current_U = current_U[:, keep]
                    current_V = current_V[:, keep]
                    current_W = current_W[:, keep]
                    current_rank = len(keep)
                    
                    if current_rank <= target_rank and new_error < 1e-8:
                        return (current_U, current_V, current_W)
                else:
                    # Revert
                    current_U[:, target_col] = old_U
                    current_V[:, target_col] = old_V
                    current_W[:, target_col] = old_W
        
        if check_decomposition(best_U, best_V, best_W):
            return (best_U, best_V, best_W)
        
        return None
    
    # Try to find the minimum rank
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try structured search for various ranks
    ranks_to_try = [50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 21]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # First try structured random search
        result = try_structured_search(r, max_attempts=2000)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
        
        # If that fails, try merging from schoolbook
        if r < rank_sb:
            result = try_linear_dependency_merging(U_sb, V_sb, W_sb, r, max_iters=500)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
