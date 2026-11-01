import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1) > tol:
                    return False
    return True

def is_half_integer(x, tol=1e-6):
    """Check if all entries are half-integers."""
    return np.allclose(x, np.round(x * 2) / 2, atol=tol)

def try_remove_term(U, V, W, term_idx, n, m, p):
    """Try removing a single term and check if decomposition still works."""
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    U_new = np.delete(U, term_idx, axis=1)
    V_new = np.delete(V, term_idx, axis=1)
    W_new = np.delete(W, term_idx, axis=1)
    
    if check_decomposition(U_new, V_new, W_new, n, m, p):
        return (U_new, V_new, W_new)
    return None

def greedy_reduce_rank(U, V, W, n, m, p, deadline):
    """Iteratively remove redundant terms."""
    rank = U.shape[1]
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for r in range(rank):
            if time.time() > deadline:
                break
            result = try_remove_term(U, V, W, r, n, m, p)
            if result is not None:
                U, V, W = result
                rank -= 1
                progress = True
                break
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Start with naive rank-45 decomposition
    rank_naive = n * m * p
    U = np.zeros((n * m, rank_naive))
    V = np.zeros((m * p, rank_naive))
    W = np.zeros((p * n, rank_naive))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Step 1: Greedy removal of redundant terms
    U, V, W = greedy_reduce_rank(U, V, W, n, m, p, deadline - 30)
    
    # Step 2: Try to find a better structured decomposition
    # For 3x3x3, the minimal rank is 23 (Strassen-like)
    # For 3x3x5, we can think of it as two 3x3x3 blocks plus one 3x3x2 block
    # But let's try a different approach: use the fact that the tensor is
    # a sum of 5 slices, each being a 3x3x3 multiplication tensor
    
    # Try random restarts with structured initialization
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to improve by looking for structured low-rank decompositions
    # We know that 3x3x3 has rank 23. For 3x3x5, we can potentially
    # use a decomposition that shares structure across the 5 columns.
    
    # Approach: Try to find a rank-r decomposition where r < current rank
    # by using optimization with structured constraints
    
    # Since we need half-integer entries, let's try a search over
    # structured patterns
    
    # Try ranks from current_rank-1 down to 30
    current_rank = best_rank
    for target_rank in range(current_rank - 1, max(25, current_rank - 15), -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random restarts with structured initialization
        for restart in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with a mix of structured and random values
            # Use the existing decomposition as a starting point and perturb
            if target_rank < current_rank:
                # Take a subset of the existing terms and add some random structure
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                
                # Copy some existing terms
                copy_count = min(target_rank, current_rank)
                for c in range(copy_count):
                    U_init[:, c] = best_U[:, c % current_rank]
                    V_init[:, c] = best_V[:, c % current_rank]
                    W_init[:, c] = best_W[:, c % current_rank]
                
                # Add small random perturbations to non-copied columns
                for c in range(copy_count, target_rank):
                    U_init[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=n * m)
                    V_init[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=m * p)
                    W_init[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=p * n)
            else:
                U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, target_rank))
                V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, target_rank))
                W_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, target_rank))
            
            # Check if this random initialization already works
            if check_decomposition(U_init, V_init, W_init, n, m, p):
                best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
                best_rank = target_rank
                break
            
            # Try local optimization: adjust entries to minimize error
            # Use a simple coordinate descent approach
            U_opt = U_init.copy()
            V_opt = V_init.copy()
            W_opt = W_init.copy()
            
            # Coordinate descent: try changing each entry to a nearby half-integer
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                improved = False
                
                # Try adjusting U entries
                for idx in range(n * m):
                    for c in range(target_rank):
                        old_val = U_opt[idx, c]
                        # Try nearby half-integers
                        for new_val in [old_val - 0.5, old_val + 0.5]:
                            if abs(new_val) > 3:  # Keep values bounded
                                continue
                            U_opt[idx, c] = new_val
                            if check_decomposition(U_opt, V_opt, W_opt, n, m, p):
                                improved = True
                                break
                            U_opt[idx, c] = old_val
                        if improved:
                            break
                    if improved:
                        break
                
                # Try adjusting V entries
                if not improved:
                    for idx in range(m * p):
                        for c in range(target_rank):
                            old_val = V_opt[idx, c]
                            for new_val in [old_val - 0.5, old_val + 0.5]:
                                if abs(new_val) > 3:
                                    continue
                                V_opt[idx, c] = new_val
                                if check_decomposition(U_opt, V_opt, W_opt, n, m, p):
                                    improved = True
                                    break
                                V_opt[idx, c] = old_val
                            if improved:
                                break
                        if improved:
                            break
                
                # Try adjusting W entries
                if not improved:
                    for idx in range(p * n):
                        for c in range(target_rank):
                            old_val = W_opt[idx, c]
                            for new_val in [old_val - 0.5, old_val + 0.5]:
                                if abs(new_val) > 3:
                                    continue
                                W_opt[idx, c] = new_val
                                if check_decomposition(U_opt, V_opt, W_opt, n, m, p):
                                    improved = True
                                    break
                                W_opt[idx, c] = old_val
                            if improved:
                                break
                        if improved:
                            break
                
                if not improved:
                    break
            
            # Final check
            if check_decomposition(U_opt, V_opt, W_opt, n, m, p):
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                best_rank = target_rank
                break
    
    # Final greedy reduction
    if time.time() < deadline - 5:
        best_U, best_V, best_W = greedy_reduce_rank(best_U, best_V, best_W, n, m, p, deadline)
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
