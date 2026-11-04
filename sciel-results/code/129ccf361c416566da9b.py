"""Program for matrix multiplication 2x4x8 tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    
    Strategy:
    1. Start with a known valid decomposition (Strassen Rank 28).
    2. Use Simulated Annealing to minimize the rank by:
       a. Attempting to set columns to zero (pruning).
       b. Attempting to merge columns (linear combination) if it reduces error.
       c. Perturbing integer entries to find better fits for a fixed rank.
    3. Verify exactness at the end.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s safety margin
    
    # --- Step 1: Construct Initial Strassen Rank-28 Decomposition ---
    def build_strassen():
        U = np.zeros((n * m, 28), dtype=int)
        V = np.zeros((m * p, 28), dtype=int)
        W = np.zeros((p * n, 28), dtype=int)
        
        def add_strassen(offset, a_block_idx, b_block_idx, c_block_idx):
            def a_idx(local_r, local_c, block):
                global_r = local_r
                global_c = block * 2 + local_c
                return global_r * m + global_c
                
            def b_idx(local_r, local_c, block):
                global_r = (block // 2) * 2 + local_r
                global_c = (block % 2) * 2 + local_c
                return global_r * p + global_c
                
            def c_idx(local_r, local_c, block):
                global_k = local_r
                global_i = local_c
                return global_k * n + global_i

            terms = [
                (0, [(3, 1), (1, 1)], [(3, 1), (1, 1)], [(0, 1), (3, 1)]),
                (1, [(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
                (2, [(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
                (3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
                (4, [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
                (5, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
                (6, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
            ]
            
            for m_idx, a_coeff, b_coeff, c_coeff in terms:
                col = offset + m_idx
                for (li, lc) in a_coeff:
                    idx = a_idx(li // 2, li % 2, a_block_idx)
                    U[idx, col] += lc
                for (li, lc) in b_coeff:
                    idx = b_idx(li // 2, li % 2, b_block_idx)
                    V[idx, col] += lc
                for (li, lc) in c_coeff:
                    idx = c_idx(li // 2, li % 2, c_block_idx)
                    W[idx, col] += lc

        add_strassen(0, 0, 0, 0)
        add_strassen(7, 1, 2, 0)
        add_strassen(14, 0, 1, 1)
        add_strassen(21, 1, 3, 1)
        
        return U, V, W

    U, V, W = build_strassen()
    
    # --- Step 2: Define Target Tensor ---
    # T[u, v, w] = 1 if indices are consistent, else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # Consistent if:
    # u % m == v // p  (j from U == j from V)
    # v % p == w // n  (k from V == k from W)
    # w % n == u // m  (i from W == i from U)
    
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    # --- Step 3: Verification Function ---
    def verify(U, V, W):
        # Check if sum_c U[u,c] V[v,c] W[w,c] == T[u,v,w] for all u,v,w
        # Use broadcasting for speed
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Need to compute tensor product and sum over R
        # This is expensive if done naively for every step.
        # For final verification, we can do:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    # Check 1s
                    val_1 = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                    if val_1 != 1:
                        return False
                    
        # Check 0s (sampled or full?)
        # Full check is (nm * mp * pn) * R ops.
        # 8 * 32 * 16 * 28 = 114,688 ops. Very fast.
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                    if val != T[u, v, w]:
                        return False
        return True

    # --- Step 4: Error Function (for optimization) ---
    def calculate_error(U, V, W):
        # Compute sum of squared errors for all u,v,w
        # Vectorized:
        # We want S[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
        # This is a 3-way tensor contraction.
        # We can compute it as:
        # Temp[u, v, c] = U[u, c] * V[v, c]
        # S[u, v, w] = sum_c Temp[u, v, c] * W[w, c]
        
        # U: (nm, R), V: (mp, R)
        # Temp = U[:, :, None] * V[None, :, :] -> (nm, mp, R)
        # This creates a large array (8 * 32 * 28 = 7168). Small.
        # Then S = Temp @ W.T -> (nm, mp, pn)
        # 7168 * 16 = 114,688. Small.
        
        Temp = U[:, :, np.newaxis] * V[np.newaxis, :, :]
        S = Temp @ W.T
        # S shape: (nm, mp, pn)
        
        Error = np.sum((S - T) ** 2)
        return Error

    # --- Step 5: Optimization Loop (Simulated Annealing / Local Search) ---
    current_error = calculate_error(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = current_error
    
    temperature = 100.0
    cooling_rate = 0.999
    
    # Try to reduce rank by zeroing columns
    # We iterate through columns and try to set them to 0.
    # If error stays low or decreases, we keep it.
    
    R = 28
    # Randomize column order to avoid bias
    col_order = list(range(R))
    
    # We will try to prune columns.
    # Strategy: Iterate through columns. For each column, try setting it to 0.
    # If the new error is <= old error + noise, accept.
    
    # Actually, simpler: Try to find a subset of columns that works.
    # But we can also tweak values.
    
    # Let's do a simple local search on the values first, then prune.
    
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # 1. Try to prune a random column
        if random.random() < 0.5:
            c = random.randint(0, R - 1)
            # Save old
            old_U = U[:, c].copy()
            old_V = V[:, c].copy()
            old_W = W[:, c].copy()
            
            # Set to 0
            U[:, c] = 0
            V[:, c] = 0
            W[:, c] = 0
            
            new_error = calculate_error(U, V, W)
            
            if new_error < current_error or (random.random() < np.exp((current_error - new_error) / temperature)):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                # Revert
                U[:, c] = old_U
                V[:, c] = old_V
                W[:, c] = old_W
                
        # 2. Try to tweak a random entry
        else:
            c = random.randint(0, R - 1)
            # Pick random matrix
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                idx = random.randint(0, n * m - 1)
                old_val = U[idx, c]
                U[idx, c] += random.randint(-1, 1)
                new_error = calculate_error(U, V, W)
                if new_error < current_error or (random.random() < np.exp((current_error - new_error) / temperature)):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    U[idx, c] = old_val
                    
            elif mat_choice == 1:
                idx = random.randint(0, m * p - 1)
                old_val = V[idx, c]
                V[idx, c] += random.randint(-1, 1)
                new_error = calculate_error(U, V, W)
                if new_error < current_error or (random.random() < np.exp((current_error - new_error) / temperature)):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    V[idx, c] = old_val
                    
            else:
                idx = random.randint(0, p * n - 1)
                old_val = W[idx, c]
                W[idx, c] += random.randint(-1, 1)
                new_error = calculate_error(U, V, W)
                if new_error < current_error or (random.random() < np.exp((current_error - new_error) / temperature)):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    W[idx, c] = old_val
                    
        temperature *= cooling_rate

    # --- Step 6: Final Verification and Rank Reduction ---
    # Use the best found
    U, V, W = best_U, best_V, best_W
    
    # Check if it's exact
    if verify(U, V, W):
        # Try to reduce rank by finding non-zero columns
        active_cols = []
        for c in range(R):
            if not (np.all(U[:, c] == 0) and np.all(V[:, c] == 0) and np.all(W[:, c] == 0)):
                active_cols.append(c)
        
        if len(active_cols) < R:
            U = U[:, active_cols]
            V = V[:, active_cols]
            W = W[:, active_cols]
            
            # Verify again just in case
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to original Strassen if optimization failed or didn't improve
    U, V, W = build_strassen()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
