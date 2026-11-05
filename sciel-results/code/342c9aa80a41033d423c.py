<<<<<<< SEARCH
def _local_search_integer(r, max_steps=5000, time_limit=10.0):
    """
    Perform a local search on the integer/half-integer grid to minimize the tensor error.
    Returns (U, V, W, success).
    """
    start_time = time.time()
    
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    U = np.random.choice(vals, size=(n * m, r))
    V = np.random.choice(vals, size=(m * p, r))
    W = np.random.choice(vals, size=(p * n, r))
    
    current_err = _compute_error(U, V, W)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        if current_err < 1e-10:
            return U, V, W, True
            
        mat_choice = np.random.randint(3)
        
        if mat_choice == 0:
            i = np.random.randint(n * m)
            j = np.random.randint(r)
            old_val = U[i, j]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            U[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                U[i, j] = old_val
            else:
                current_err = new_err
                
        elif mat_choice == 1:
            i = np.random.randint(m * p)
            j = np.random.randint(r)
            old_val = V[i, j]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            V[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                V[i, j] = old_val
            else:
                current_err = new_err
                
        else:
            i = np.random.randint(p * n)
            j = np.random.randint(r)
            old_val = W[i, j]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            W[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                W[i, j] = old_val
            else:
                current_err = new_err
                
    return U, V, W, False
=======
def _local_search_integer(r, max_steps=5000, time_limit=10.0):
    """
    Perform a local search on the integer/half-integer grid to minimize the tensor error.
    Returns (U, V, W, success).
    """
    start_time = time.time()
    
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Initialize with a structured starting point
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Initialize with a sparse random pattern
    for c in range(r):
        # Pick a few non-zero entries in each column
        u_indices = np.random.choice(n * m, size=max(1, r // 4), replace=False)
        v_indices = np.random.choice(m * p, size=max(1, r // 4), replace=False)
        w_indices = np.random.choice(p * n, size=max(1, r // 4), replace=False)
        
        U[u_indices, c] = np.random.choice([0.5, -0.5, 1.0, -1.0], size=len(u_indices))
        V[v_indices, c] = np.random.choice([0.5, -0.5, 1.0, -1.0], size=len(v_indices))
        W[w_indices, c] = np.random.choice([0.5, -0.5, 1.0, -1.0], size=len(w_indices))
    
    current_err = _compute_error(U, V, W)
    
    # Simulated annealing style search
    temperature = 1.0
    cooling = 0.999
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        if current_err < 1e-10:
            return U, V, W, True
            
        # Try a random perturbation
        mat_choice = np.random.randint(3)
        num_changes = np.random.randint(1, 4)  # Change 1-3 entries
        
        old_vals = []
        
        for _ in range(num_changes):
            if mat_choice == 0:
                i = np.random.randint(n * m)
                j = np.random.randint(r)
                old_vals.append(('U', i, j, U[i, j]))
                U[i, j] = np.random.choice(vals)
            elif mat_choice == 1:
                i = np.random.randint(m * p)
                j = np.random.randint(r)
                old_vals.append(('V', i, j, V[i, j]))
                V[i, j] = np.random.choice(vals)
            else:
                i = np.random.randint(p * n)
                j = np.random.randint(r)
                old_vals.append(('W', i, j, W[i, j]))
                W[i, j] = np.random.choice(vals)
        
        new_err = _compute_error(U, V, W)
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if new_err < current_err:
            current_err = new_err
        else:
            delta = new_err - current_err
            if np.random.random() < np.exp(-delta / temperature):
                current_err = new_err
            else:
                # Revert
                for mat_name, i, j, old_val in old_vals:
                    if mat_name == 'U':
                        U[i, j] = old_val
                    elif mat_name == 'V':
                        V[i, j] = old_val
                    else:
                        W[i, j] = old_val
        
        temperature *= cooling
    
    # Final polish: try to snap to nearest 0.5 values
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    final_err = _compute_error(U, V, W)
    if final_err < 1e-10:
        return U, V, W, True
    
    return U, V, W, False
>>>>>>> REPLACE

