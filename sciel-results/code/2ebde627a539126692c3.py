def construct():
    import time
    start = time.time()
    DEADLINE = 110

    def time_check():
        return (time.time() - start) < DEADLINE

    n, m, p = 2, 4, 5
    # U: (n*m, rank) = (8, rank), row = i*m + j = i*4 + j
    # V: (m*p, rank) = (20, rank), row = j*p + k = j*5 + k
    # W: (p*n, rank) = (10, rank), row = k*n + i = k*2 + i

    U = {}
    V = {}
    W = {}
    next_col = 0

    # Strassen steps for 2x2x2
    STRASSE = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    def add_24x42(a_cols, b_cols):
        """Add a 2x4 * 4x2 block using two 2x2 Strassen steps."""
        nonlocal next_col
        for part in range(2):
            for (X, Y, P) in STRASSE:
                col = next_col
                next_col += 1
                for (i, j), c in X.items():
                    r = i*4 + a_cols[part*2 + j]
                    U.setdefault(r, {})[col] = U.get(r, {}).get(col, 0) + c
                for (a, b), c in Y.items():
                    r = a_cols[part*2 + a]*5 + b_cols[b]
                    V.setdefault(r, {})[col] = V.get(r, {}).get(col, 0) + c
                for (i, b), c in P.items():
                    r = b_cols[b]*2 + i
                    W.setdefault(r, {})[col] = W.get(r, {}).get(col, 0) + c

    def add_24x41(a_cols, b_col):
        """Add a 2x4 * 4x1 block using schoolbook (8 rank-1 terms)."""
        nonlocal next_col
        for part in range(2):
            for l_local in range(2):
                for i in range(2):
                    col = next_col
                    next_col += 1
                    r_u = i*4 + a_cols[part*2 + l_local]
                    U.setdefault(r_u, {})[col] = U.get(r_u, {}).get(col, 0) + 1
                    r_v = a_cols[part*2 + l_local]*5 + b_col
                    V.setdefault(r_v, {})[col] = V.get(r_v, {}).get(col, 0) + 1
                    r_w = b_col*2 + i
                    W.setdefault(r_w, {})[col] = W.get(r_w, {}).get(col, 0) + 1

    # Sub-problems:
    # C2: A * B[:, 2:4]  (2x4 * 4x2) -> 14 rank terms
    if time_check():
        add_24x42([0,1,2,3], [2,3])
    # C3: A * B[:, 4:5]  (2x4 * 4x1) -> 8 rank terms
    if time_check():
        add_24x41([0,1,2,3], 4)
    # C4: A * B[:, 0:2]  (2x4 * 4x2) -> 14 rank terms
    if time_check():
        add_24x42([0,1,2,3], [0,1])

    rank = next_col

    # Try to find a lower-rank decomposition via column elimination
    # Check if any column can be expressed as a linear combination of others
    # using rational arithmetic (Fraction) for exactness
    
    from fractions import Fraction
    import itertools
    
    def try_eliminate_columns(U_dict, V_dict, W_dict, rank):
        """Try to eliminate columns that are linear combinations of others."""
        # Build the tensor as a 3D array for verification
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # We want to check if column r can be removed by adjusting other columns
        
        # For each column r, check if it's in the span of the others
        # This is a linear algebra problem in the space of tensors
        # T_r = U_r ⊗ V_r ⊗ W_r (outer product)
        # We want to see if T_r = sum_{s != r} c_s * T_s
        
        # This is a system of (n*m*p) equations in (rank-1) unknowns
        # Let's build the matrix
        
        n, m, p = 2, 4, 5
        total_entries = n * m * p  # 40
        
        # Build matrix A where A[:, s] = flatten(U_s ⊗ V_s ⊗ W_s) for s != r
        # And b = flatten(U_r ⊗ V_r ⊗ W_r)
        
        def flatten_outer(u_col, v_col, w_col):
            """Flatten the outer product of three vectors."""
            # u_col: length n*m=8
            # v_col: length m*p=20
            # w_col: length p*n=10
            # Result: length 40, indexed by (i, j, k) where i in [0,2), j in [0,4), k in [0,5)
            # T[i,j,k] = u[i*4+j] * v[j*5+k] * w[k*2+i]
            result = [Fraction(0)] * total_entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        result[idx] = Fraction(u_col[i*4+j]) * Fraction(v_col[j*5+k]) * Fraction(w_col[k*2+i])
            return result
        
        # Extract column vectors as lists of Fractions
        def get_col(matrix_dict, num_rows, r):
            return [Fraction(matrix_dict.get(i, {}).get(r, 0)) for i in range(num_rows)]
        
        U_rows = 8
        V_rows = 20
        W_rows = 10
        
        # For each column r, try to eliminate it
        best_rank = rank
        eliminated = set()
        
        for r in range(rank):
            if r in eliminated:
                continue
            
            # Build the system for eliminating column r
            # Unknowns: c_s for s != r
            # Equations: sum_{s != r} c_s * T_s = T_r
            
            other_cols = [s for s in range(rank) if s != r and s not in eliminated]
            num_unknowns = len(other_cols)
            
            if num_unknowns == 0:
                continue
            
            # Build augmented matrix
            # A: total_entries x num_unknowns
            # b: total_entries
            
            A = []
            b = []
            
            u_r = get_col(U_dict, U_rows, r)
            v_r = get_col(V_dict, V_rows, r)
            w_r = get_col(W_dict, W_rows, r)
            b = flatten_outer(u_r, v_r, w_r)
            
            for s in other_cols:
                u_s = get_col(U_dict, U_rows, s)
                v_s = get_col(V_dict, V_rows, s)
                w_s = get_col(W_dict, W_rows, s)
                col = flatten_outer(u_s, v_s, w_s)
                A.append(col)
            
            # Now solve A^T * c = b (least squares, but we want exact solution)
            # Actually, we want to check if b is in the span of A's columns
            # Use Gaussian elimination
            
            # Transpose: we have num_unknowns columns, total_entries rows
            # We want to solve: sum_s c_s * A_s = b
            # This is a system of total_entries equations in num_unknowns unknowns
            
            # Use Gaussian elimination on the augmented matrix [A | b]
            # A is total_entries x num_unknowns
            
            # Build augmented matrix as list of lists
            aug = []
            for i in range(total_entries):
                row = [A[s][i] for s in range(num_unknowns)]
                row.append(b[i])
                aug.append(row)
            
            # Gaussian elimination
            num_rows = total_entries
            num_cols = num_unknowns
            
            pivot = 0
            for col in range(num_cols):
                # Find pivot
                max_row = -1
                max_val = Fraction(0)
                for row in range(pivot, num_rows):
                    val = abs(aug[row][col])
                    if val > max_val:
                        max_val = val
                        max_row = row
                
                if max_row == -1 or max_val == 0:
                    continue
                
                # Swap
                if max_row != pivot:
                    aug[pivot], aug[max_row] = aug[max_row], aug[pivot]
                
                # Eliminate
                pivot_val = aug[pivot][col]
                for row in range(pivot + 1, num_rows):
                    factor = aug[row][col] / pivot_val
                    if factor == 0:
                        continue
                    for j in range(col, num_cols + 1):
                        aug[row][j] -= factor * aug[pivot][j]
                
                pivot += 1
            
            # Check if the system is consistent
            # After elimination, check if any row has 0s in the coefficient part but nonzero in b
            for row in range(pivot, num_rows):
                all_zero = True
                for j in range(num_cols):
                    if aug[row][j] != 0:
                        all_zero = False
                        break
                if all_zero and aug[row][num_cols] != 0:
                    # Inconsistent system, column r cannot be eliminated
                    break
            else:
                # System is consistent, column r can be eliminated
                eliminated.add(r)
                best_rank = rank - len(eliminated)
        
        return best_rank
    
    # Try column elimination
    if time_check():
        improved_rank = try_eliminate_columns(U, V, W, rank)
        # Note: We don't actually perform the elimination in this version
        # We just check if it's possible. For now, we keep the original rank.
        # A full implementation would need to actually modify the matrices.
    
    rank = next_col  # Keep original rank for now

    # Convert to lists
    U_list = [[0]*rank for _ in range(8)]
    V_list = [[0]*rank for _ in range(20)]
    W_list = [[0]*rank for _ in range(10)]
    for r, cols in U.items():
        for c, val in cols.items():
            U_list[r][c] = val
    for r, cols in V.items():
        for c, val in cols.items():
            V_list[r][c] = val
    for r, cols in W.items():
        for c, val in cols.items():
            W_list[r][c] = val

    return {"U": U_list, "V": V_list, "W": W_list}
