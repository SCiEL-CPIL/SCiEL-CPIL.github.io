import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    # Build the rank-46 decomposition using 6 Strassen blocks + 1 single block
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    def add_strassen_block(k0, col_offset):
        prods = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        c_combos = [
            (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
            (0, k0+1,   [(2,1),(4,1)]),
            (1, k0,     [(1,1),(3,1)]),
            (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff
    
    def add_single_block(k, col_offset):
        prods = [
            ((0,0),(0,k)),
            ((0,1),(1,k)),
            ((1,0),(0,k)),
            ((1,1),(1,k)),
        ]
        for t, ((i,j),(jj,kk)) in enumerate(prods):
            col = col_offset + t
            U[A_idx(i, j), col] += 1
            V[B_idx(jj, kk), col] += 1
        W[C_idx(0, k), col_offset+0] += 1
        W[C_idx(0, k), col_offset+1] += 1
        W[C_idx(1, k), col_offset+2] += 1
        W[C_idx(1, k), col_offset+3] += 1
    
    # 6 Strassen blocks for k0 = 0,2,4,6,8,10
    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    # 1 single block for k=12
    add_single_block(12, col)
    col += 4
    assert col == R, f"col={col}, R={col}"
    
    # Verify the decomposition
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    target_arr = np.array(target_indices)
    a_idx = target_arr[:, 0]
    b_idx = target_arr[:, 1]
    c_idx = target_arr[:, 2]
    
    def verify(U_mat, V_mat, W_mat):
        U_t = U_mat[a_idx, :]
        V_t = V_mat[b_idx, :]
        W_t = W_mat[c_idx, :]
        prod = U_t * V_t * W_t
        sums = np.sum(prod, axis=1)
        return np.all(np.abs(sums - 1.0) < 1e-9)
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
    
    # Try to reduce rank by finding linear dependencies
    # Build the contribution matrix: for each target triple and each column,
    # the product U[a]*V[b]*W[c]. We need sum over columns = 1 for each triple.
    # If a column's contribution is a linear combination of others, we can remove it.
    
    # More efficient approach: try removing columns one at a time
    # For each column c, check if the system can still be satisfied without it
    
    # Even more efficient: check if column c's contribution vector is in the span
    # of the other columns' contribution vectors
    
    # Build contribution vectors: for column r, the vector is U[a_idx,r] * V[b_idx,r] * W[c_idx,r]
    # This is a vector of length 52 (n*m*p)
    
    # For rank reduction, we need to find if any column is redundant
    # Try greedy elimination
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_R = R
    removed = []
    
    # Try removing each column
    for attempt in range(20):  # Try up to 20 removals
        if time.time() > deadline - 10:
            break
        
        found_removal = False
        for col_to_remove in range(current_R):
            if time.time() > deadline - 10:
                break
            
            U_test = np.delete(current_U, col_to_remove, axis=1)
            V_test = np.delete(current_V, col_to_remove, axis=1)
            W_test = np.delete(current_W, col_to_remove, axis=1)
            
            if verify(U_test, V_test, W_test):
                current_U, current_V, current_W = U_test, V_test, W_test
                current_R -= 1
                removed.append(col_to_remove)
                found_removal = True
                best_U, best_V, best_W, best_R = current_U.copy(), current_V.copy(), current_W.copy(), current_R
                break
        
        if not found_removal:
            break
    
    # Try more aggressive: remove multiple columns at once by checking subsets
    # This is expensive, so only try if we have time
    if time.time() < deadline - 30:
        # Try removing pairs of columns
        for i in range(current_R):
            if time.time() > deadline - 30:
                break
            for j in range(i+1, current_R):
                if time.time() > deadline - 30:
                    break
                
                # Remove columns i and j
                cols_to_keep = [k for k in range(current_R) if k != i and k != j]
                U_test = current_U[:, cols_to_keep]
                V_test = current_V[:, cols_to_keep]
                W_test = current_W[:, cols_to_keep]
                
                if verify(U_test, V_test, W_test):
                    current_U, current_V, current_W = U_test, V_test, W_test
                    current_R -= 2
                    best_U, best_V, best_W, best_R = current_U.copy(), current_V.copy(), current_W.copy(), current_R
                    break
    
    # Snap to half-integers for clean output
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # If snapping broke it, use the un-snapped version
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
