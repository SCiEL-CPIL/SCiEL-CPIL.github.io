# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use Simulated Annealing to escape local minima.
    3. Use structured initializations (block-circulant, random sparse).
    4. Use fast vectorized error computation.
    """
    start_time = time.time()
    deadline = start_time + 145  # 145s budget, leaving 5s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 9, 12, 12
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    # Precompute all triples that need to be checked
    all_triples = list(itertools.product(range(nm), range(mp), range(pn)))
    active_set = set(active_triples)
    
    # Create index arrays for vectorized computation
    A_idx = np.array([a for a, b, c in all_triples])
    B_idx = np.array([b for a, b, c in all_triples])
    C_idx = np.array([c for a, b, c in all_triples])
    
    # Expected values: 1 for active triples, 0 for inactive
    expected_vals = np.array([1 if (a, b, c) in active_set else 0 for a, b, c in all_triples])
    
    # Precompute for fast error computation
    triple_a = np.array([a for a, b, c in all_triples])
    triple_b = np.array([b for a, b, c in all_triples])
    triple_c = np.array([c for a, b, c in all_triples])
    
    def compute_prod(U_arr, V_arr, W_arr, r):
        """Compute the tensor product sum for all triples."""
        prod = np.zeros(len(all_triples), dtype=np.int32)
        for ci in range(r):
            prod += U_arr[A_idx, ci] * V_arr[B_idx, ci] * W_arr[C_idx, ci]
        return prod
    
    def verify_numpy(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints using numpy."""
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        prod = compute_prod(U_arr, V_arr, W_arr, r)
        return np.array_equal(prod, expected_vals)
    
    def error_count_numpy(U, V, W, r):
        """Count how many triples are violated using numpy."""
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        prod = compute_prod(U_arr, V_arr, W_arr, r)
        return int(np.sum(prod != expected_vals))
    
    def simulated_annealing(U, V, W, r, max_iters=20000, T_start=10.0, T_end=0.1):
        """
        Simulated Annealing to minimize the number of violated constraints.
        """
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        
        prod = compute_prod(U_arr, V_arr, W_arr, r)
        current_errors = int(np.sum(prod != expected_vals))
        best_errors = current_errors
        best_U = U_arr.copy()
        best_V = V_arr.copy()
        best_W = W_arr.copy()
        
        T = T_start
        cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_errors == 0:
                return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
            
            # Randomly pick a matrix, row, column
            mat_choice = random.random()
            if mat_choice < 0.33:
                mat = 'U'
                row = random.randint(0, nm - 1)
            elif mat_choice < 0.66:
                mat = 'V'
                row = random.randint(0, mp - 1)
            else:
                mat = 'W'
                row = random.randint(0, pn - 1)
            
            col = random.randint(0, r - 1)
            
            if mat == 'U':
                old_val = U_arr[row, col]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                delta_prod = delta * V_arr[triple_b, col] * W_arr[triple_c, col]
                new_prod = prod + delta_prod
                new_errors = int(np.sum(new_prod != expected_vals))
                delta_E = new_errors - current_errors
                
                if delta_E < 0 or (T > 0 and random.random() < np.exp(-delta_E / T)):
                    U_arr[row, col] = new_val
                    prod = new_prod
                    current_errors = new_errors
                    if current_errors < best_errors:
                        best_errors = current_errors
                        best_U = U_arr.copy()
                        best_V = V_arr.copy()
                        best_W = W_arr.copy()
            elif mat == 'V':
                old_val = V_arr[row, col]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                delta_prod = delta * U_arr[triple_a, col] * W_arr[triple_c, col]
                new_prod = prod + delta_prod
                new_errors = int(np.sum(new_prod != expected_vals))
                delta_E = new_errors - current_errors
                
                if delta_E < 0 or (T > 0 and random.random() < np.exp(-delta_E / T)):
                    V_arr[row, col] = new_val
                    prod = new_prod
                    current_errors = new_errors
                    if current_errors < best_errors:
                        best_errors = current_errors
                        best_U = U_arr.copy()
                        best_V = V_arr.copy()
                        best_W = W_arr.copy()
            else:  # W
                old_val = W_arr[row, col]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                delta_prod = delta * U_arr[triple_a, col] * V_arr[triple_b, col]
                new_prod = prod + delta_prod
                new_errors = int(np.sum(new_prod != expected_vals))
                delta_E = new_errors - current_errors
                
                if delta_E < 0 or (T > 0 and random.random() < np.exp(-delta_E / T)):
                    W_arr[row, col] = new_val
                    prod = new_prod
                    current_errors = new_errors
                    if current_errors < best_errors:
                        best_errors = current_errors
                        best_U = U_arr.copy()
                        best_V = V_arr.copy()
                        best_W = W_arr.copy()
            
            T *= cooling_factor
        
        # Return the best found during the annealing process
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank_structured(r, max_attempts=50):
        """Try to find a rank-r decomposition using structured random search + SA."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with structured patterns
            strategy = attempt % 3
            
            if strategy == 0:
                # Sparse random
                U = [[0] * r for _ in range(nm)]
                V = [[0] * r for _ in range(mp)]
                W = [[0] * r for _ in range(pn)]
                for a in range(nm):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(mp):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(pn):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
            
            else:
                # Block-circulant inspired
                U = [[0] * r for _ in range(nm)]
                V = [[0] * r for _ in range(mp)]
                W = [[0] * r for _ in range(pn)]
                # Simple structured initialization
                for a in range(nm):
                    for ci in range(r):
                        if (a + ci) % r < r // 2:
                            U[a][ci] = random.choice([-1, 1])
                for b in range(mp):
                    for ci in range(r):
                        if (b + ci) % r < r // 2:
                            V[b][ci] = random.choice([-1, 1])
                for c in range(pn):
                    for ci in range(r):
                        if (c + ci) % r < r // 2:
                            W[c][ci] = random.choice([-1, 1])
            
            # Quick check
            if verify_numpy(U, V, W, r):
                return {"U": U, "V": V, "W": W}
            
            # Run Simulated Annealing
            time_remaining = deadline - time.time()
            if time_remaining > 5:
                # Allocate time for SA
                max_iters_sa = min(20000, int(time_remaining * 1000))
                U_sa, V_sa, W_sa = simulated_annealing(U, V, W, r, max_iters=max_iters_sa)
                if error_count_numpy(U_sa, V_sa, W_sa, r) == 0:
                    return {"U": U_sa, "V": V_sa, "W": W_sa}
        
        return None
    
    # Try to find lower rank decompositions
    for r in range(21, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Spend more time on promising ranks
        if r <= 23:
            max_attempts = min(100, int(time_remaining * 5))
        else:
            max_attempts = min(50, int(time_remaining * 2))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
