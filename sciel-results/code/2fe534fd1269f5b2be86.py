"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import random
    import time
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    schoolbook_rank = nm * p  # 42
    
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Build list of non-zero indices in T
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i * m + j, j * p + k, k * n + i))
    
    nonzero_set = set(T_indices)
    all_zero_indices = []
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if (i, j, k) not in nonzero_set:
                    all_zero_indices.append((i, j, k))
    random.shuffle(all_zero_indices)
    T_zero_sample = all_zero_indices[:300]
    
    def compute_error_fast(U, V, W):
        """Compute error using vectorized operations for speed."""
        r = U.shape[1]
        error = 0.0
        # Check non-zero entries
        for (i, j, k) in T_indices:
            val = np.dot(U[i], V[j] * W[k])
            error += abs(val - 1)
        # Check zero entries
        for (i, j, k) in T_zero_sample:
            val = np.dot(U[i], V[j] * W[k])
            if abs(val) > 1e-10:
                error += abs(val)
        return error
    
    def full_verify(U, V, W):
        """Full verification of the decomposition."""
        r = U.shape[1]
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        for c in range(r):
            result += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
        for (i, j, k) in T_indices:
            if abs(result[i, j, k] - 1) > 1e-10:
                return False
        for (i, j, k) in all_zero_indices:
            if abs(result[i, j, k]) > 1e-10:
                return False
        return True
    
    def schoolbook():
        U = np.zeros((nm, schoolbook_rank), dtype=np.float64)
        V = np.zeros((mp, schoolbook_rank), dtype=np.float64)
        W = np.zeros((pn, schoolbook_rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def strassen_block_decomposition():
        """
        Build a rank-39 decomposition using block structure.
        Partition A (2x3) into [A_L (2x2), A_R (2x1)]
        Partition B (3x7) into [B_1 (2x7); B_2 (1x7)]
        
        C = A_L * B_1 + A_R * B_2
        
        For A_L * B_1: partition B_1 (2x7) into 3 blocks of 2 columns + 1 block of 1 column
        B_1 = [B_11 | B_12 | B_13 | B_14] where B_11,B_12,B_13 are 2x2, B_14 is 2x1
        
        A_L * B_11: Strassen rank 7
        A_L * B_12: Strassen rank 7
        A_L * B_13: Strassen rank 7
        A_L * B_14: Direct rank 4 (2*2)
        A_R * B_2: Direct rank 14 (2*7)
        
        Total: 7+7+7+4+14 = 39
        """
        rank = 39
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        col = 0
        
        # Strassen 2x2x2 helper
        def add_strassen(a_idx, b_idx, c_idx, start_col):
            """
            Add Strassen decomposition for 2x2 * 2x2.
            a_idx: list of 4 row indices in U (for A_L rows)
            b_idx: list of 4 row indices in V (for B columns)
            c_idx: list of 4 row indices in W (for C rows)
            """
            nonlocal col
            # Strassen's 7 products for 2x2 matrices
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B10 - B11)
            # M4 = A11(B01 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # M1
            U[a_idx[0], start_col] = 1; U[a_idx[3], start_col] = 1
            V[b_idx[0], start_col] = 1; V[b_idx[3], start_col] = 1
            W[c_idx[0], start_col] = 1; W[c_idx[3], start_col] = 1
            col += 1
            # M2
            U[a_idx[2], start_col] = 1; U[a_idx[3], start_col] = 1
            V[b_idx[0], start_col] = 1
            W[c_idx[2], start_col] = 1; W[c_idx[3], start_col] = 1
            col += 1
            # M3
            U[a_idx[0], start_col] = 1
            V[b_idx[2], start_col] = 1; V[b_idx[3], start_col] = -1
            W[c_idx[1], start_col] = 1
            col += 1
            # M4
            U[a_idx[3], start_col] = 1
            V[b_idx[1], start_col] = 1; V[b_idx[0], start_col] = -1
            W[c_idx[0], start_col] = 1; W[c_idx[2], start_col] = 1
            col += 1
            # M5
            U[a_idx[0], start_col] = 1; U[a_idx[1], start_col] = 1
            V[b_idx[3], start_col] = 1
            W[c_idx[0], start_col] = -1; W[c_idx[1], start_col] = 1
            col += 1
            # M6
            U[a_idx[2], start_col] = 1; U[a_idx[0], start_col] = -1
            V[b_idx[0], start_col] = 1; V[b_idx[1], start_col] = 1
            W[c_idx[3], start_col] = 1
            col += 1
            # M7
            U[a_idx[1], start_col] = 1; U[a_idx[3], start_col] = -1
            V[b_idx[2], start_col] = 1; V[b_idx[3], start_col] = 1
            W[c_idx[0], start_col] = 1
            col += 1
        
        # Direct 2x2x2 helper
        def add_direct_2x2(a_idx, b_idx, c_idx, start_col):
            """Add direct 4-product decomposition for 2x2 * 2x2."""
            nonlocal col
            # A00*B00
            U[a_idx[0], start_col] = 1
            V[b_idx[0], start_col] = 1
            W[c_idx[0], start_col] = 1
            col += 1
            # A00*B01
            U[a_idx[0], start_col] = 1
            V[b_idx[1], start_col] = 1
            W[c_idx[1], start_col] = 1
            col += 1
            # A01*B10
            U[a_idx[1], start_col] = 1
            V[b_idx[2], start_col] = 1
            W[c_idx[0], start_col] = 1
            col += 1
            # A01*B11
            U[a_idx[1], start_col] = 1
            V[b_idx[3], start_col] = 1
            W[c_idx[1], start_col] = 1
            col += 1
            # A10*B00
            U[a_idx[2], start_col] = 1
            V[b_idx[0], start_col] = 1
            W[c_idx[2], start_col] = 1
            col += 1
            # A10*B01
            U[a_idx[2], start_col] = 1
            V[b_idx[1], start_col] = 1
            W[c_idx[3], start_col] = 1
            col += 1
            # A11*B10
            U[a_idx[3], start_col] = 1
            V[b_idx[2], start_col] = 1
            W[c_idx[2], start_col] = 1
            col += 1
            # A11*B11
            U[a_idx[3], start_col] = 1
            V[b_idx[3], start_col] = 1
            W[c_idx[3], start_col] = 1
            col += 1
        
        # Direct 2x1x7 helper (A_R is 2x1, B_2 is 1x7)
        def add_direct_2x1(a_idx, b_idx, c_idx, start_col):
            """Add direct decomposition for 2x1 * 1x7. Rank = 2*7 = 14."""
            nonlocal col
            # A is 2x1: indices a_idx[0], a_idx[1] (rows 0,1 of A_L... wait)
            # Actually A_R is column 2 of A (2x3 matrix), so A_R has rows i=0,1
            # B_2 is row 2 of B (3x7 matrix), so B_2 has columns k=0..6
            # C contribution: C[i,k] += A_R[i] * B_2[k]
            for i in range(2):
                for k in range(7):
                    U[i * m + 2, start_col] = 1  # A_R row i, which is row i of A, col 2
                    V[2 * p + k, start_col] = 1  # B_2, which is row 2 of B, col k
                    W[k * n + i, start_col] = 1  # C[k,i]
                    start_col += 1
            col = start_col
        
        # A_L is rows 0,1 of A, columns 0,1
        # A_L indices in U: rows 0,1 (i=0,j=0), 1,2 (i=0,j=1), 3,4 (i=1,j=0), 4,5 (i=1,j=1)
        # U row indices: i*m + j
        # A_L: i in {0,1}, j in {0,1}
        # So U rows: 0,1,3,4
        
        a_L_idx = [0, 1, 3, 4]  # A00, A01, A10, A11
        
        # B_1 is rows 0,1 of B, columns 0..6
        # B_11: cols 0,1; B_12: cols 2,3; B_13: cols 4,5; B_14: col 6
        
        # B_11: b_idx for V rows
        # V row indices: j*p + k
        # B_11: j in {0,1}, k in {0,1}
        b_11_idx = [0, 1, p, p+1]  # B00, B01, B10, B11
        
        # B_12: j in {0,1}, k in {2,3}
        b_12_idx = [2, 3, p+2, p+3]
        
        # B_13: j in {0,1}, k in {4,5}
        b_13_idx = [4, 5, p+4, p+5]
        
        # B_14: j in {0,1}, k in {6}
        # This is 2x1, not 2x2, so we handle differently
        
        # C indices: C is 2x7, indices k*n + i
        # C0: cols 0,1 (for B_11)
        c_11_idx = [0, 1, n, n+1]  # C00, C01, C10, C11 where C[k,i]
        
        # C1: cols 2,3 (for B_12)
        c_12_idx = [2, 3, n+2, n+3]
        
        # C2: cols 4,5 (for B_13)
        c_13_idx = [4, 5, n+4, n+5]
        
        # C3: col 6 (for B_14)
        
        # Add Strassen for A_L * B_11
        add_strassen(a_L_idx, b_11_idx, c_11_idx, col)
        
        # Add Strassen for A_L * B_12
        add_strassen(a_L_idx, b_12_idx, c_12_idx, col)
        
        # Add Strassen for A_L * B_13
        add_strassen(a_L_idx, b_13_idx, c_13_idx, col)
        
        # A_L * B_14: A_L is 2x2, B_14 is 2x1
        # C contribution: C[i,6] += A_L[i,0]*B_14[0,6] + A_L[i,1]*B_14[1,6]
        # This is 2*2 = 4 products
        # A_L rows: 0,1; B_14 rows: 0,1 (j=0,1, k=6)
        for i in range(2):
            for j in range(2):
                U[a_L_idx[i*2 + j], col] = 1
                V[b_11_idx[j*2] + 5, col] = 1  # B_14[j] = V row j*p + 6
                W[6 * n + i, col] = 1
                col += 1
        
        # A_R * B_2: A_R is 2x1 (col 2 of A), B_2 is 1x7 (row 2 of B)
        # C[i,k] += A_R[i] * B_2[k]
        # A_R: U rows i*m + 2 = i*3 + 2
        # B_2: V rows 2*p + k = 14 + k
        for i in range(2):
            for k in range(7):
                U[i * m + 2, col] = 1
                V[2 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        assert col == 39, f"Expected rank 39, got {col}"
        
        return U, V, W
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = schoolbook()
    
    # Try Strassen block decomposition (rank 39)
    U39, V39, W39 = strassen_block_decomposition()
    if full_verify(U39, V39, W39):
        best_rank = 39
        best_U, best_V, best_W = U39, V39, W39
    
    # Try to find even lower rank using simulated annealing local search
    if time.time() < deadline - 30:
        for target_rank in range(36, best_rank):
            if time.time() > deadline - 20:
                break
            
            found = False
            for attempt in range(30):
                if time.time() > deadline - 15:
                    break
                
                U = np.random.randint(-2, 3, (nm, target_rank), dtype=np.float64)
                V = np.random.randint(-2, 3, (mp, target_rank), dtype=np.float64)
                W = np.random.randint(-2, 3, (pn, target_rank), dtype=np.float64)
                
                current_error = compute_error_fast(U, V, W)
                
                temp = 50.0
                for step in range(1500):
                    if time.time() > deadline - 15:
                        break
                    if current_error == 0:
                        break
                    
                    factor_idx = random.randint(0, 2)
                    if factor_idx == 0:
                        row = random.randint(0, nm - 1)
                        col_idx = random.randint(0, target_rank - 1)
                        old_val = U[row, col_idx]
                        U[row, col_idx] = old_val + random.choice([-2, -1, 1, 2])
                        new_error = compute_error_fast(U, V, W)
                        if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 1e-6)):
                            current_error = new_error
                        else:
                            U[row, col_idx] = old_val
                    elif factor_idx == 1:
                        row = random.randint(0, mp - 1)
                        col_idx = random.randint(0, target_rank - 1)
                        old_val = V[row, col_idx]
                        V[row, col_idx] = old_val + random.choice([-2, -1, 1, 2])
                        new_error = compute_error_fast(U, V, W)
                        if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 1e-6)):
                            current_error = new_error
                        else:
                            V[row, col_idx] = old_val
                    else:
                        row = random.randint(0, pn - 1)
                        col_idx = random.randint(0, target_rank - 1)
                        old_val = W[row, col_idx]
                        W[row, col_idx] = old_val + random.choice([-2, -1, 1, 2])
                        new_error = compute_error_fast(U, V, W)
                        if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 1e-6)):
                            current_error = new_error
                        else:
                            W[row, col_idx] = old_val
                    
                    temp *= 0.997
                
                if current_error == 0 and full_verify(U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
            
            if not found:
                break
    
    U_list = [[int(round(x)) for x in row] for row in best_U]
    V_list = [[int(round(x)) for x in row] for row in best_V]
    W_list = [[int(round(x)) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
