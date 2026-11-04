import time
import numpy as np
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    NM = n * m
    MP = m * p
    PN = p * n
    
    # Total number of tensor entries
    total_entries = n * m * p
    
    # Indices mapping
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Precompute the list of (i, j, k) for all tensor entries
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((i, j, k, u_idx, v_idx, w_idx))
                
    # For incremental updates, we need to know which tensor entries are affected by a change in a specific matrix entry.
    # If we change U[u_idx, r], it affects all (i, j, k) such that i*m+j == u_idx.
    # For a fixed u_idx (i, j), the affected k's are all k in [0, p).
    # The corresponding v_idx is j*p+k and w_idx is k*n+i.
    
    # Let's build maps:
    # U_AFFECTS[u_idx][r] -> list of (tensor_idx, v_idx, w_idx)
    # V_AFFECTS[v_idx][r] -> list of (tensor_idx, u_idx, w_idx)
    # W_AFFECTS[w_idx][r] -> list of (tensor_idx, u_idx, v_idx)
    
    # tensor_idx is just the index in the 'indices' list, 0 to total_entries-1
    # We can map (i,j,k) to tensor_idx.
    idx_map = {}
    for t_idx, (i, j, k, u_idx, v_idx, w_idx) in enumerate(indices):
        idx_map[(i, j, k)] = t_idx
        
    U_AFFECTS = [[] for _ in range(NM)]
    V_AFFECTS = [[] for _ in range(MP)]
    W_AFFECTS = [[] for _ in range(PN)]
    
    for t_idx, (i, j, k, u_idx, v_idx, w_idx) in enumerate(indices):
        U_AFFECTS[u_idx].append((t_idx, v_idx, w_idx))
        V_AFFECTS[v_idx].append((t_idx, u_idx, w_idx))
        W_AFFECTS[w_idx].append((t_idx, u_idx, v_idx))
        
    def get_trivial_solution():
        rank = total_entries
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def search_rank(R, time_budget):
        # We work with integers.
        # Target: sum_r U[u,r] V[v,r] W[w,r] = 1 for all (u,v,w) in support.
        # Note: The trivial solution uses 1s.
        # If we allow half-integers, we can multiply everything by 2 and target 2.
        # But integer entries are simpler. The problem says "integer/half-integer".
        # Let's stick to integers first. If that fails, try half-integers (scale by 2, target 2).
        
        # Actually, let's just use integers. The trivial solution is integer.
        # Is it possible that a lower rank solution requires half-integers?
        # Yes, potentially.
        # Let's try to find an integer solution first.
        
        # State:
        # U, V, W: numpy arrays of shape (dim, R), dtype int32
        # T: numpy array of shape (total_entries,), dtype float64 (current tensor values)
        
        # Initialize T
        def compute_T(U, V, W):
            T = np.zeros(total_entries, dtype=np.float64)
            for t_idx, (i, j, k, u_idx, v_idx, w_idx) in enumerate(indices):
                # T[t_idx] = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                T[t_idx] = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            return T
            
        def error(T):
            return np.sum((1.0 - T) ** 2)
            
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Time limit check
        start = time.time()
        
        # Number of restarts depends on time budget
        # Each restart: init, compute T, local search
        # Local search: try random moves
        
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start > time_budget:
                break
                
            # Random initialization
            # Values in {-1, 0, 1} or similar
            U = np.random.randint(-1, 2, size=(NM, R), dtype=np.int32)
            V = np.random.randint(-1, 2, size=(MP, R), dtype=np.int32)
            W = np.random.randint(-1, 2, size=(PN, R), dtype=np.int32)
            
            T = compute_T(U, V, W)
            err = error(T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-9:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            # Local Search
            # We maintain T and update it incrementally.
            # We try to reduce err.
            
            # To make it faster, we can precompute the "weight" of each r for each affected tensor entry?
            # No, just update T directly.
            
            max_iters = 2000
            for it in range(max_iters):
                if time.time() - start > time_budget:
                    break
                    
                # Pick a random matrix, row, col
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    mat = U
                    dim = NM
                    affects = U_AFFECTS
                    name = 'U'
                elif mat_choice == 1:
                    mat = V
                    dim = MP
                    affects = V_AFFECTS
                    name = 'V'
                else:
                    mat = W
                    dim = PN
                    affects = W_AFFECTS
                    name = 'W'
                    
                row = random.randint(0, dim - 1)
                col = random.randint(0, R - 1)
                
                old_val = mat[row, col]
                
                # Try +1 and -1
                # We calculate delta_err for each
                best_delta = None
                best_new_val = None
                min_delta_err = 0.0 # We only accept if err decreases (delta_err < 0)
                
                for delta in [1, -1]:
                    new_val = old_val + delta
                    if new_val == old_val:
                        continue
                    
                    # Calculate change in T and Error
                    # Affected tensor entries are in affects[row]
                    # For each affected (t_idx, other1, other2):
                    #   old_T = T[t_idx]
                    #   new_T = old_T + delta * mat[other1, col] * mat[other2, col]
                    #   old_err_contrib = (1 - old_T)^2
                    #   new_err_contrib = (1 - new_T)^2
                    #   delta_err += new_err_contrib - old_err_contrib
                    
                    delta_err = 0.0
                    # We need to store the old T values to restore if we don't accept?
                    # Or we can just simulate.
                    # Since we only have 2 deltas to check, we can simulate both and pick the best.
                    
                    # Let's simulate the changes for this delta
                    # We need to know the "other" values.
                    
                    if name == 'U':
                        # affects[row] is list of (t_idx, v_idx, w_idx)
                        for t_idx, v_idx, w_idx in affects[row]:
                            old_T = T[t_idx]
                            # V[v_idx, col] and W[w_idx, col] are constant for this move
                            factor = V[v_idx, col] * W[w_idx, col]
                            new_T = old_T + delta * factor
                            delta_err += (1.0 - new_T)**2 - (1.0 - old_T)**2
                    elif name == 'V':
                        # affects[row] is list of (t_idx, u_idx, w_idx)
                        for t_idx, u_idx, w_idx in affects[row]:
                            old_T = T[t_idx]
                            factor = U[u_idx, col] * W[w_idx, col]
                            new_T = old_T + delta * factor
                            delta_err += (1.0 - new_T)**2 - (1.0 - old_T)**2
                    else: # W
                        # affects[row] is list of (t_idx, u_idx, v_idx)
                        for t_idx, u_idx, v_idx in affects[row]:
                            old_T = T[t_idx]
                            factor = U[u_idx, col] * V[v_idx, col]
                            new_T = old_T + delta * factor
                            delta_err += (1.0 - new_T)**2 - (1.0 - old_T)**2
                            
                    if delta_err < min_delta_err:
                        min_delta_err = delta_err
                        best_delta = delta
                        best_new_val = new_val
                        
                # If we found an improving move
                if best_delta is not None:
                    # Apply the move
                    mat[row, col] = best_new_val
                    
                    # Update T
                    if name == 'U':
                        for t_idx, v_idx, w_idx in affects[row]:
                            T[t_idx] += best_delta * V[v_idx, col] * W[w_idx, col]
                    elif name == 'V':
                        for t_idx, u_idx, w_idx in affects[row]:
                            T[t_idx] += best_delta * U[u_idx, col] * W[w_idx, col]
                    else:
                        for t_idx, u_idx, v_idx in affects[row]:
                            T[t_idx] += best_delta * U[u_idx, col] * V[v_idx, col]
                            
                    err += min_delta_err
                    
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if err < 1e-9:
                            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                            
                    # Optional: if err is not decreasing for a long time, break?
                    # Simple: just keep going until time limit or max iters.
                    
        return None

    # Try ranks
    # Lower bound is not strictly known, but 18 is a guess.
    # Let's try 36 down to 18? Or 18 up to 36?
    # If we find a solution at 20, we should check 19, 18.
    # But checking 18 might be very hard if it's impossible.
    # Let's try 36 first to ensure we have a valid baseline (trivial).
    # Then try 35, 34...
    # Actually, the trivial solution is always valid at 36.
    # So we can just return 36 if we don't find better.
    
    # Let's try to find a solution with rank < 36.
    # Time budget: 120s.
    # Allocate time to different ranks.
    
    # Let's try rank 35 first, then 34, etc.
    # Or randomize?
    # Let's just iterate.
    
    start_time = time.time()
    time_limit = 110 # Safety margin
    
    for R in range(35, 17, -1):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
            
        # Give more time to higher ranks? Or equal?
        # Let's give 5 seconds per rank attempt.
        t_budget = min(5.0, remaining)
        
        result = search_rank(R, t_budget)
        if result is not None:
            return result
            
    # Fallback
    return get_trivial_solution()

