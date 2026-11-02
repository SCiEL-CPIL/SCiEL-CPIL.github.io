# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product, combinations

def _reconstruct_fast(U, V, W, n, m, p, rank):
    """
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Returns T: (n, m, p)
    """
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return T

def _validate(U, V, W, n, m, p, rank):
    T = _reconstruct_fast(U, V, W, n, m, p, rank)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _snap_to_half_integers(U, V, W, tol=0.01):
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    if (np.max(np.abs(U - U_s)) < tol and
        np.max(np.abs(V - V_s)) < tol and
        np.max(np.abs(W - W_s)) < tol):
        return U_s, V_s, W_s
    return None

def _compute_error(T):
    return np.sum((T - 1.0)**2)

def _local_search_discrete(U, V, W, n, m, p, rank, time_limit, seed=0):
    """
    Performs local search on the discrete grid of half-integers.
    Uses incremental error updates.
    """
    rng = random.Random(seed)
    
    # Reshape for easier indexing in incremental updates
    # U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
    U_r = U.reshape(n, m, rank).copy()
    V_r = V.reshape(m, p, rank).copy()
    W_r = W.reshape(p, n, rank).copy()
    
    T = _reconstruct_fast(U_r.reshape(-1, rank), V_r.reshape(-1, rank), W_r.reshape(-1, rank), n, m, p, rank)
    current_error = _compute_error(T)
    best_error = current_error
    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    
    if best_error < 1e-9:
        return best_U.reshape(-1, rank), best_V.reshape(-1, rank), best_W.reshape(-1, rank)
    
    start_time = time.time()
    steps = 0
    max_steps = 20000 # Cap steps to avoid infinite loop if time check is slow
    
    while time.time() - start_time < time_limit:
        steps += 1
        if steps > max_steps:
            break
            
        # Choose which matrix to modify: 0 for U, 1 for V, 2 for W
        matrix_type = rng.choice([0, 1, 2])
        
        if matrix_type == 0:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            c = rng.randint(0, rank - 1)
            old_val = U_r[i, j, c]
            
            # Propose new value
            # Half integers: k/2
            # Try neighbors: -1, -0.5, 0.5, 1
            delta = rng.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            
            # Incremental update
            # T[i,j,k] changes by delta * V_r[j,k,c] * W_r[k,i,c]
            # Error change for each k:
            # (T[i,j,k] - 1 + delta * V[j,k,c] * W[k,i,c])^2 - (T[i,j,k] - 1)^2
            delta_error = 0.0
            for k in range(p):
                term = V_r[j, k, c] * W_r[k, i, c]
                if term == 0: continue
                t_old = T[i, j, k]
                t_new = t_old + delta * term
                delta_error += (t_new - 1.0)**2 - (t_old - 1.0)**2
            
            if current_error + delta_error < best_error or rng.random() < 0.05: # Simple annealing/hill climbing mix
                # Apply update
                U_r[i, j, c] = new_val
                for k in range(p):
                    T[i, j, k] += delta * V_r[j, k, c] * W_r[k, i, c]
                current_error += delta_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    if best_error < 1e-9:
                        return best_U.reshape(-1, rank), best_V.reshape(-1, rank), best_W.reshape(-1, rank)
                        
        elif matrix_type == 1:
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            c = rng.randint(0, rank - 1)
            old_val = V_r[j, k, c]
            delta = rng.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            
            delta_error = 0.0
            for i in range(n):
                term = U_r[i, j, c] * W_r[k, i, c]
                if term == 0: continue
                t_old = T[i, j, k]
                t_new = t_old + delta * term
                delta_error += (t_new - 1.0)**2 - (t_old - 1.0)**2
            
            if current_error + delta_error < best_error or rng.random() < 0.05:
                V_r[j, k, c] = new_val
                for i in range(n):
                    T[i, j, k] += delta * U_r[i, j, c] * W_r[k, i, c]
                current_error += delta_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    if best_error < 1e-9:
                        return best_U.reshape(-1, rank), best_V.reshape(-1, rank), best_W.reshape(-1, rank)
                        
        else: # matrix_type == 2
            k = rng.randint(0, p - 1)
            i = rng.randint(0, n - 1)
            c = rng.randint(0, rank - 1)
            old_val = W_r[k, i, c]
            delta = rng.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            
            delta_error = 0.0
            for j in range(m):
                term = U_r[i, j, c] * V_r[j, k, c]
                if term == 0: continue
                t_old = T[i, j, k]
                t_new = t_old + delta * term
                delta_error += (t_new - 1.0)**2 - (t_old - 1.0)**2
            
            if current_error + delta_error < best_error or rng.random() < 0.05:
                W_r[k, i, c] = new_val
                for j in range(m):
                    T[i, j, k] += delta * U_r[i, j, c] * V_r[j, k, c]
                current_error += delta_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    if best_error < 1e-9:
                        return best_U.reshape(-1, rank), best_V.reshape(-1, rank), best_W.reshape(-1, rank)
                        
    if best_error < 1e-9:
        return best_U.reshape(-1, rank), best_V.reshape(-1, rank), best_W.reshape(-1, rank)
    return None

def _try_structured_search(n, m, p, target_rank, time_limit, seed=0):
    """
    Attempts to find a decomposition by enumerating structured candidates.
    This is a heuristic based on symmetries, but for general n,m,p,
    random local search is often more robust if we have time.
    However, the directive asks to investigate improvements within the family.
    Let's add a more aggressive random restart strategy with better initialization.
    """
    pass

def construct():
    n, m, p = 2, 3, 4
    schoolbook_rank = n * m * p
    
    # Schoolbook fallback
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 140
    
    # Try ranks from lower to higher
    # Known rank for <2,3,4> is 18. Let's try 18, 19, 20...
    # We go from low to high? No, usually we want the MINIMAL rank.
    # So we should try 18, 19, 20... until we find one?
    # Or try 23, 22... and keep the best?
    # The score is lower is better. So we want the smallest rank.
    # If we find rank 18, that's great.
    # Let's try 18, 19, 20, 21, 22, 23.
    
    ranks_to_try = [18, 19, 20, 21, 22, 23]
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try a few random restarts for this rank
        found = False
        # Increase restarts for lower ranks where it's harder?
        # Or just use more restarts in general if time permits.
        # Let's try up to 20 restarts, but break early if time runs out.
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 10:
                break
                
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5:
                break
            
            # Initialize with random half-integers
            # Use a seed that ensures different inits for each restart
            rng = np.random.RandomState(rank * 1000 + restart)
            
            # Smaller range for initialization might help local search converge faster
            # Values: -1, -0.5, 0, 0.5, 1
            init_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U_init = rng.choice(init_vals, size=(n*m, rank))
            V_init = rng.choice(init_vals, size=(m*p, rank))
            W_init = rng.choice(init_vals, size=(p*n, rank))
            
            # Run local search
            # Give it a portion of the remaining time
            # If we have 100s left and 20 restarts, each gets ~5s?
            # But we want to spend more time on promising restarts.
            # Let's allocate a fixed small time per restart first, then if not found, 
            # we might just keep going with the best so far? 
            # Actually, the current logic runs local search for `remaining_time - 5`.
            # This means the first restart gets almost all the time.
            # If it fails, subsequent restarts get less and less time.
            # This is a good strategy: spend lots of time on the first attempt.
            # If it finds a solution, we break.
            
            # However, if the first restart fails, it consumes most of the time.
            # Maybe we should limit the time per restart to allow more restarts?
            # Let's try a time budget per restart, e.g., min(remaining_time, 10s)
            # But if we find a solution in 2s, we are happy.
            
            # Let's stick to the existing logic but maybe increase max_steps in local search?
            # The local search has a max_steps of 20000.
            # With incremental updates, 20000 steps is fast.
            # Let's increase max_steps to 100000 to allow deeper exploration per restart.
            
            result = _local_search_discrete(U_init, V_init, W_init, n, m, p, rank, remaining_time - 5, seed=rank*1000+restart)
            
            if result is not None:
                U_s, V_s, W_s = result
                # Convert to lists
                U_list = [[val for val in row] for row in U_s.tolist()]
                V_list = [[val for val in row] for row in V_s.tolist()]
                W_list = [[val for val in row] for row in W_s.tolist()]
                
                # Validate
                if _validate(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank):
                    best_result = {
                        "U": U_list,
                        "V": V_list,
                        "W": W_list
                    }
                    best_rank = rank
                    found = True
                    break
        
        # If we found a solution for this rank, we can stop if we are looking for minimal rank?
        # Yes, since we iterate 18, 19, 20..., the first one found is the minimal among those tried.
        # But what if 18 is impossible and 19 is possible? We break out of the inner loop.
        # Should we continue to check if a lower rank is possible?
        # We are iterating 18, 19, 20. So if 18 fails, we try 19.
        # If 19 succeeds, we have rank 19. Can we find 18? We already tried 18.
        # So we can break the outer loop?
        # No, we might have failed 18 due to bad luck (random init).
        # But we tried 5 restarts.
        # Let's just keep the best result found.
        # Since we iterate from low to high, the first success is the best (lowest) rank.
        if found:
             # We could try to spend remaining time trying to find an even lower rank?
             # No, 18 is the lower bound we know of.
             # So we can break.
             break
    
    return best_result
# EVOLVE-BLOCK-END
