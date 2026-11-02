# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave margin before 120s limit
    
    n, m, p = 2, 2, 7
    
    # Build the rank-25 solution (Strassen for 3 blocks + schoolbook for 1 block)
    def build_rank_25():
        rank = 25
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        strassen_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        strassen_W_local = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        strassen_V_local = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        
        for b in range(3):
            for r in range(7):
                for i in range(4):
                    U[i][col] += strassen_U[r][i]
                for loc_idx, coeff in strassen_V_local[r].items():
                    v_idx = (loc_idx // 2) * p + (2 * b + loc_idx % 2)
                    V[v_idx][col] += coeff
                for loc_idx, coeff in strassen_W_local[r].items():
                    w_idx = (2 * b + loc_idx % 2) * n + (loc_idx // 2)
                    W[w_idx][col] += coeff
                col += 1
        
        schoolbook_U = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        for r in range(4):
            for i in range(4):
                U[i][col] += schoolbook_U[r][i]
            if r == 0: V[0*p+6][col] = 1
            elif r == 1: V[1*p+6][col] = 1
            elif r == 2: V[0*p+6][col] = 1
            elif r == 3: V[1*p+6][col] = 1
            if r < 2: W[6*n+0][col] = 1
            else: W[6*n+1][col] = 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find a rank-24 solution by searching for a linear dependency
    # among the 25 rank-1 components
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose the tensor"""
        n, m, p = 2, 2, 7
        rank = len(U[0])
        
        # Build tensor T[a,b,c] = sum_r U[a][r]*V[b][r]*W[c][r]
        T = {}
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = 0
                    for r in range(rank):
                        if U[a][r] != 0 and V[b][r] != 0 and W[c][r] != 0:
                            val += U[a][r] * V[b][r] * W[c][r]
                    if val != 0:
                        T[(a,b,c)] = val
        
        # Expected: T[(i*m+j, j*p+k, k*n+i)] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    expected = 1
                    actual = T.get((a_idx, b_idx, c_idx), 0)
                    if abs(actual - expected) > 1e-9:
                        return False
        
        # Check no spurious terms
        for (a,b,c), val in T.items():
            # Check if (a,b,c) is a valid term
            valid = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if a == i*m+j and b == j*p+k and c == k*n+i:
                            valid = True
                            break
                    if valid:
                        break
                if valid:
                    break
            if not valid and abs(val) > 1e-9:
                return False
        
        return True
    
    # First, get the rank-25 solution
    sol25 = build_rank_25()
    
    # Try to reduce to rank 24 by finding a linear combination
    # that cancels one column
    U25 = sol25["U"]
    V25 = sol25["V"]
    W25 = sol25["W"]
    rank25 = len(U25[0])
    
    # Check if any two columns are linearly dependent
    # For rank reduction, we need to find coefficients c_r such that
    # sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0
    # This means for all a,b,c: sum_r c_r * U[a][r]*V[b][r]*W[c][r] = 0
    
    # This is a system of equations. Let's set up the matrix.
    # Variables: c_0, ..., c_24 (25 variables)
    # Equations: for each (a,b,c), sum_r c_r * U[a][r]*V[b][r]*W[c][r] = 0
    
    # Number of equations: (n*m)*(m*p)*(p*n) = 4*14*14 = 784
    # But many are trivially satisfied. Let's just check if the 25 columns
    # of the "flattened" tensor are linearly dependent.
    
    # Flatten: for each r, create a vector of length (n*m)*(m*p)*(p*n) = 784
    # where entry (a,b,c) is U[a][r]*V[b][r]*W[c][r]
    # Then check if these 25 vectors are linearly dependent.
    
    # This is a 784x25 matrix. Check its rank.
    
    def compute_rank_of_flattened(U, V, W, max_entries=None):
        """Compute rank of the matrix whose columns are the flattened rank-1 tensors"""
        n, m, p = 2, 2, 7
        rank = len(U[0])
        num_a = n * m
        num_b = m * p
        num_c = p * n
        
        # Build matrix: rows = (a,b,c) combinations, cols = r
        # Use Gaussian elimination on a subset if too large
        rows = []
        for a in range(num_a):
            for b in range(num_b):
                for c in range(num_c):
                    row = []
                    for r in range(rank):
                        row.append(U[a][r] * V[b][r] * W[c][r])
                    rows.append(row)
        
        # Gaussian elimination
        mat = [row[:] for row in rows]
        num_rows = len(mat)
        num_cols = rank
        
        rank_found = 0
        for col in range(num_cols):
            # Find pivot
            pivot_row = -1
            for row in range(rank_found, num_rows):
                if abs(mat[row][col]) > 1e-12:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            
            # Swap
            mat[rank_found], mat[pivot_row] = mat[pivot_row], mat[rank_found]
            
            # Eliminate
            pivot_val = mat[rank_found][col]
            for row in range(num_rows):
                if row != rank_found and abs(mat[row][col]) > 1e-12:
                    factor = mat[row][col] / pivot_val
                    for c2 in range(col, num_cols):
                        mat[row][c2] -= factor * mat[rank_found][c2]
            
            rank_found += 1
        
        return rank_found
    
    # Check rank of the 25 flattened vectors
    # This might be slow, so add a deadline check
    if time.time() < deadline - 20:
        actual_rank = compute_rank_of_flattened(U25, V25, W25)
        if actual_rank < 25:
            # We can potentially reduce rank
            # But finding the actual reduced decomposition is complex
            # For now, just note it
            pass
    
    # Return the rank-25 solution
    return sol25
# EVOLVE-BLOCK-END
