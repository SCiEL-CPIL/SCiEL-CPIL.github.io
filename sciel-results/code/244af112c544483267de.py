def _local_search(n, m, p, target_rank, max_time=5, restarts=3):
    start_time = time.time()
    rng = random.Random(42)
    
    for r in range(restarts):
        # Initialize
        U = [[0]*target_rank for _ in range(n*m)]
        V = [[0]*target_rank for _ in range(m*p)]
        W = [[0]*target_rank for _ in range(p*n)]
        
        # Random init
        for c in range(target_rank):
            for idx in range(n*m):
                if rng.random() < 0.3:
                    U[idx][c] = rng.choice([-1, 1])
            for idx in range(m*p):
                if rng.random() < 0.3:
                    V[idx][c] = rng.choice([-1, 1])
            for idx in range(p*n):
                if rng.random() < 0.3:
                    W[idx][c] = rng.choice([-1, 1])
                    
        current_error = _error_fast(U, V, W, n, m, p)
        
        # Local search loop
        while time.time() - start_time < max_time:
            mat_type = rng.choice(['U', 'V', 'W'])
            if mat_type == 'U':
                idx = rng.randint(0, n*m - 1)
                c = rng.randint(0, target_rank - 1)
                old_val = U[idx][c]
            elif mat_type == 'V':
                idx = rng.randint(0, m*p - 1)
                c = rng.randint(0, target_rank - 1)
                old_val = V[idx][c]
            else:
                idx = rng.randint(0, p*n - 1)
                c = rng.randint(0, target_rank - 1)
                old_val = W[idx][c]
                
            valid_vals = [-1, -0.5, 0, 0.5, 1]
            new_val = rng.choice(valid_vals)
            
            if mat_type == 'U':
                U[idx][c] = new_val
            elif mat_type == 'V':
                V[idx][c] = new_val
            else:
                W[idx][c] = new_val
                
            new_error = _error_fast(U, V, W, n, m, p)
            
            if new_error < current_error:
                current_error = new_error
                if current_error < 1e-9:
                    if _check_all_entries(U, V, W):
                        return U, V, W
            else:
                if mat_type == 'U':
                    U[idx][c] = old_val
                elif mat_type == 'V':
                    V[idx][c] = old_val
                else:
                    W[idx][c] = old_val
                    
    return None
