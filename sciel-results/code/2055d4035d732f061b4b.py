<<<<<<< SEARCH
    # Try to find lower rank using structured search
    # Key insight: many entries can be 0, ±1, or ±1/2
    # Use a smarter approach: try ranks from 38 down, using random search
    # with constraint that values are in {-1, -0.5, 0, 0.5, 1}

    # For efficiency, represent the problem differently.
    # For a given rank R, we need U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all (a,b,c) in triples: sum_r U[a,r]*V[b,r]*W[c,r] = 1

    # Try ranks 38 down to 36 with random search
    for R in range(38, 35, -1):
        if time.time() > deadline:
            break

        # Each row of U, V, W is a vector of length R with entries in {-1,-0.5,0,0.5,1}
        # That's 5^R possibilities per row - too many for brute force.
        # Instead, use random sampling with local search.

        num_rows_U = n * m  # 6
        num_rows_V = m * p  # 21
        num_rows_W = p * n  # 14
        total_rows = num_rows_U + num_rows_V + num_rows_W

        # Random search with multiple restarts
        max_restarts = 8
        if time.time() > deadline - 10:
            max_restarts = 2

        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break

            np.random.seed(hash((R, attempt)) % (2**32))

            # Initialize with random values from the allowed set
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

            U = np.zeros((num_rows_U, R))
            V = np.zeros((num_rows_V, R))
            W = np.zeros((num_rows_W, R))

            # For each entry, randomly choose from vals with some probability of 0
            for r in range(R):
                # Random pattern for each column
                for i in range(num_rows_U):
                    if np.random.random() < 0.4:
                        U[i, r] = np.random.choice(vals)
                for i in range(num_rows_V):
                    if np.random.random() < 0.3:
                        V[i, r] = np.random.choice(vals)
                for i in range(num_rows_W):
                    if np.random.random() < 0.3:
                        W[i, r] = np.random.choice(vals)

            # Local search: try flipping individual entries
            for step in range(500):
                if time.time() > deadline - 3:
                    break

                # Check current quality
                err = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a], np.multiply(V[b], W[c]))
                    err += (val - 1.0) ** 2

                if err < 1e-9:
                    # Verify and return
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if check_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break

                # Pick a random position to modify
                choice = np.random.random()
                if choice < num_rows_U / total_rows:
                    row = np.random.randint(num_rows_U)
                    mat = U
                    mat_name = 'U'
                elif choice < (num_rows_U + num_rows_V) / total_rows:
                    row = np.random.randint(num_rows_V)
                    mat = V
                    mat_name = 'V'
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W
                    mat_name = 'W'

                col = np.random.randint(R)
                old_val = mat[row, col]

                # Try a new value
                new_val = np.random.choice(vals)
                if new_val == old_val:
                    continue

                mat[row, col] = new_val

                # Compute error change
                err_new = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a], np.multiply(V[b], W[c]))
                    err_new += (val - 1.0) ** 2

                if err_new >= err:
                    mat[row, col] = old_val
                else:
                    err = err_new
=======
    # Try to find lower rank using structured search
    # Key insight: many entries can be 0, ±1, or ±1/2
    # Use a smarter approach: try ranks from 38 down, using random search
    # with constraint that values are in {-1, -0.5, 0, 0.5, 1}

    # For efficiency, represent the problem differently.
    # For a given rank R, we need U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all (a,b,c) in triples: sum_r U[a,r]*V[b,r]*W[c,r] = 1

    # Precompute which triples are affected by each (matrix, row, col) position
    # For faster error updates
    num_rows_U = n * m  # 6
    num_rows_V = m * p  # 21
    num_rows_W = p * n  # 14
    total_rows = num_rows_U + num_rows_V + num_rows_W

    # Build index: for each U row a, which triples have that a
    U_triples = {a: [] for a in range(num_rows_U)}
    V_triples = {b: [] for b in range(num_rows_V)}
    W_triples = {c: [] for c in range(num_rows_W)}
    for idx, (a, b, c) in enumerate(triples):
        U_triples[a].append(idx)
        V_triples[b].append(idx)
        W_triples[c].append(idx)

    def compute_error(U, V, W):
        """Compute sum of squared errors efficiently."""
        err = 0.0
        for (a, b, c) in triples:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            err += (val - 1.0) ** 2
        return err

    def compute_error_incremental(U, V, W, mat_name, row, col, old_val, new_val):
        """Compute error change when modifying one entry."""
        delta = new_val - old_val
        if delta == 0:
            return 0.0
        
        err_change = 0.0
        if mat_name == 'U':
            for idx in U_triples[row]:
                a, b, c = triples[idx]
                old_contrib = U[a, col] * V[b, col] * W[c, col]
                new_contrib = (U[a, col] + delta) * V[b, col] * W[c, col]
                old_val_trip = np.dot(U[a], np.multiply(V[b], W[c]))
                new_val_trip = old_val_trip + delta * V[b, col] * W[c, col]
                err_change += (new_val_trip - 1.0) ** 2 - (old_val_trip - 1.0) ** 2
        elif mat_name == 'V':
            for idx in V_triples[row]:
                a, b, c = triples[idx]
                old_val_trip = np.dot(U[a], np.multiply(V[b], W[c]))
                new_val_trip = old_val_trip + delta * U[a, col] * W[c, col]
                err_change += (new_val_trip - 1.0) ** 2 - (old_val_trip - 1.0) ** 2
        else:  # W
            for idx in W_triples[row]:
                a, b, c = triples[idx]
                old_val_trip = np.dot(U[a], np.multiply(V[b], W[c]))
                new_val_trip = old_val_trip + delta * U[a, col] * V[b, col]
                err_change += (new_val_trip - 1.0) ** 2 - (old_val_trip - 1.0) ** 2
        
        return err_change

    # Try ranks 38 down to 36 with random search
    for R in range(38, 35, -1):
        if time.time() > deadline:
            break

        # Each row of U, V, W is a vector of length R with entries in {-1,-0.5,0,0.5,1}
        # That's 5^R possibilities per row - too many for brute force.
        # Instead, use random sampling with local search.

        # Random search with multiple restarts
        max_restarts = 12
        if time.time() > deadline - 10:
            max_restarts = 3

        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break

            np.random.seed(hash((R, attempt)) % (2**32))

            # Initialize with random values from the allowed set
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

            U = np.zeros((num_rows_U, R))
            V = np.zeros((num_rows_V, R))
            W = np.zeros((num_rows_W, R))

            # For each entry, randomly choose from vals with some probability of 0
            for r in range(R):
                # Random pattern for each column
                for i in range(num_rows_U):
                    if np.random.random() < 0.4:
                        U[i, r] = np.random.choice(vals)
                for i in range(num_rows_V):
                    if np.random.random() < 0.3:
                        V[i, r] = np.random.choice(vals)
                for i in range(num_rows_W):
                    if np.random.random() < 0.3:
                        W[i, r] = np.random.choice(vals)

            # Local search: try flipping individual entries
            for step in range(800):
                if time.time() > deadline - 3:
                    break

                # Check current quality
                err = compute_error(U, V, W)

                if err < 1e-9:
                    # Verify and return
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if check_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break

                # Pick a random position to modify
                choice = np.random.random()
                if choice < num_rows_U / total_rows:
                    row = np.random.randint(num_rows_U)
                    mat = U
                    mat_name = 'U'
                elif choice < (num_rows_U + num_rows_V) / total_rows:
                    row = np.random.randint(num_rows_V)
                    mat = V
                    mat_name = 'V'
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W
                    mat_name = 'W'

                col = np.random.randint(R)
                old_val = mat[row, col]

                # Try a new value
                new_val = np.random.choice(vals)
                if new_val == old_val:
                    continue

                # Compute error change incrementally
                err_change = compute_error_incremental(U, V, W, mat_name, row, col, old_val, new_val)

                if err_change >= 0:
                    continue

                # Accept the change
                mat[row, col] = new_val

    # Try a more systematic approach for rank 38
    if time.time() < deadline - 20:
        R = 38
        # Use the rank-39 solution and try to merge two columns
        U39 = np.array(fallback_result['U'], dtype=float)
        V39 = np.array(fallback_result['V'], dtype=float)
        W39 = np.array(fallback_result['W'], dtype=float)

        # Try removing one column and adjusting others
        for r_remove in range(39):
            if time.time() > deadline - 5:
                break
            
            # Create a rank-38 version by removing column r_remove
            keep_cols = [r for r in range(39) if r != r_remove]
            U38 = U39[:, keep_cols]
            V38 = V39[:, keep_cols]
            W38 = W39[:, keep_cols]
            
            # Check if this already works (unlikely but possible)
            err = compute_error(U38, V38, W38)
            if err < 1e-9:
                U_list = U38.tolist()
                V_list = V38.tolist()
                W_list = W38.tolist()
                if check_decomposition(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try local search on this rank-38 starting point
            np.random.seed(hash((38, r_remove)) % (2**32))
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            for step in range(400):
                if time.time() > deadline - 3:
                    break
                
                err = compute_error(U38, V38, W38)
                
                if err < 1e-9:
                    U_list = U38.tolist()
                    V_list = V38.tolist()
                    W_list = W38.tolist()
                    if check_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break
                
                # Pick a random position to modify
                choice = np.random.random()
                if choice < num_rows_U / total_rows:
                    row = np.random.randint(num_rows_U)
                    mat = U38
                    mat_name = 'U'
                elif choice < (num_rows_U + num_rows_V) / total_rows:
                    row = np.random.randint(num_rows_V)
                    mat = V38
                    mat_name = 'V'
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W38
                    mat_name = 'W'
                
                col = np.random.randint(R)
                old_val = mat[row, col]
                
                # Try a new value
                new_val = np.random.choice(vals)
                if new_val == old_val:
                    continue
                
                # Compute error change incrementally
                err_change = compute_error_incremental(U38, V38, W38, mat_name, row, col, old_val, new_val)
                
                if err_change >= 0:
                    continue
                
                # Accept the change
                mat[row, col] = new_val
>>>>>>> REPLACE

