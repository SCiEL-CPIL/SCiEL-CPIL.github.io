import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # safety margin
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, k*p+l, r*n+s] = 1 iff j==k and i==r
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i * m + j, j * p + l, l * n + i] = 1
    
    def verify(U, V, W, tol=1e-9):
        """Check that sum_r U[:,r] outer V[:,r] outer W[:,r] == T exactly."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        r = U.shape[1]
        recon = np.zeros(T.shape, dtype=np.float64)
        for c in range(r):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(recon, T, atol=tol, rtol=0):
            return False
        # Check integer/half-integer constraint
        for M in (U, V, W):
            for v in M.flat:
                if abs(v) > 1e-12:
                    if abs(v - round(v)) > 1e-9 and abs(2*v - round(2*v)) > 1e-9:
                        return False
        return True
    
    def make_base_solution():
        """Return the known rank-22 solution."""
        rank = 22
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx, col] += val
            for idx, val in v_indices:
                V[idx, col] += val
            for idx, val in w_indices:
                W[idx, col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W
    
    # Start with the base rank-22 solution
    U_base, V_base, W_base = make_base_solution()
    best_rank = 22
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    
    # Strategy 1: Try to find a rank-20 or rank-21 decomposition using structured search
    # The key idea: try to merge outer product terms with Strassen terms
    
    # Try rank 21: see if we can eliminate one of the 8 outer product terms
    # by absorbing it into a Strassen term
    for target_rank in range(21, 17, -1):
        if time.time() - start_time > deadline:
            break
        
        # Try multiple random perturbations of the base solution
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline:
                break
            
            # Start from base and try to zero out (22 - target_rank) columns
            # while adjusting remaining columns
            U_try = U_base.copy()
            V_try = V_base.copy()
            W_try = W_base.copy()
            
            # Randomly select columns to eliminate
            cols_to_keep = list(range(22))
            random.shuffle(cols_to_keep)
            cols_to_keep = cols_to_keep[:target_rank]
            cols_to_eliminate = [c for c in range(22) if c not in cols_to_keep]
            
            if len(cols_to_eliminate) == 0:
                continue
            
            # Zero out the eliminated columns
            for c in cols_to_eliminate:
                U_try[:, c] = 0
                V_try[:, c] = 0
                W_try[:, c] = 0
            
            # Now try to adjust the remaining columns to compensate
            # Use a simple iterative approach: for each eliminated column,
            # try to add its contribution to other columns with small adjustments
            
            # Check if the current truncated solution works
            if verify(U_try, V_try, W_try):
                # Trim to actual rank
                U_trim = U_try[:, cols_to_keep]
                V_trim = V_try[:, cols_to_keep]
                W_trim = W_try[:, cols_to_keep]
                if verify(U_trim, V_trim, W_trim):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_trim.copy(), V_trim.copy(), W_trim.copy()
                    break
            
            # Try small perturbations of the remaining columns
            # Add small integer/half-integer adjustments
            U_adj = U_try.copy()
            V_adj = V_try.copy()
            W_adj = W_try.copy()
            
            for _ in range(random.randint(1, 3)):
                # Pick a random column to adjust
                col = random.choice(cols_to_keep)
                # Add a small adjustment
                adj = random.choice([-1, 1, -0.5, 0.5])
                row_u = random.randint(0, n*m-1)
                row_v = random.randint(0, m*p-1)
                row_w = random.randint(0, p*n-1)
                U_adj[row_u, col] += adj
                V_adj[row_v, col] += adj
                W_adj[row_w, col] += adj
            
            if verify(U_adj, V_adj, W_adj):
                # Trim
                U_trim = U_adj[:, cols_to_keep]
                V_trim = V_adj[:, cols_to_keep]
                W_trim = W_adj[:, cols_to_keep]
                if verify(U_trim, V_trim, W_trim):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_trim.copy(), V_trim.copy(), W_trim.copy()
                    break
        
        if best_rank == target_rank:
            break
    
    # Strategy 2: Direct random search for low-rank solutions
    if best_rank > 18:
        for target_rank in range(18, 14, -1):
            if time.time() - start_time > deadline:
                break
            
            max_attempts = 50
            for attempt in range(max_attempts):
                if time.time() - start_time > deadline:
                    break
                
                U_rand = np.zeros((n * m, target_rank), dtype=np.float64)
                V_rand = np.zeros((m * p, target_rank), dtype=np.float64)
                W_rand = np.zeros((p * n, target_rank), dtype=np.float64)
                
                for c in range(target_rank):
                    for _ in range(random.randint(1, 3)):
                        U_rand[random.randint(0, n*m-1), c] = random.choice([-1, 1, -0.5, 0.5])
                    for _ in range(random.randint(1, 3)):
                        V_rand[random.randint(0, m*p-1), c] = random.choice([-1, 1, -0.5, 0.5])
                    for _ in range(random.randint(1, 3)):
                        W_rand[random.randint(0, p*n-1), c] = random.choice([-1, 1, -0.5, 0.5])
                
                if verify(U_rand, V_rand, W_rand):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    break
            
            if best_rank == target_rank:
                break
    
    # Use the best solution found
    if best_rank < 22:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    else:
        U, V, W = make_base_solution()
        rank = 22
    
    # Convert to lists with proper integer/half-integer representation
    def to_list(mat, rank):
        result = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(rank):
                val = mat[i, j]
                if abs(val - round(val)) < 1e-9:
                    row.append(int(round(val)))
                elif abs(2*val - round(2*val)) < 1e-9:
                    row.append(round(2*val) / 2.0)
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    U_list = to_list(U, rank)
    V_list = to_list(V, rank)
    W_list = to_list(W, rank)
    
    # Final verification
    if not verify(U_list, V_list, W_list):
        # Fall back to base solution
        U, V, W = make_base_solution()
        rank = 22
        U_list = to_list(U, rank)
        V_list = to_list(V, rank)
        W_list = to_list(W, rank)
    
    return {"U": U_list, "V": V_list, "W": W_list}
