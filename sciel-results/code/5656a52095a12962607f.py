import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Helper to create target tensor non-zero indices
    def get_nonzero_indices(n, m, p):
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    indices.append((u, v, w))
        return indices

    # Helper to compute error
    def compute_error(U, V, W, nonzero_indices, n, m, p):
        T_comp = np.einsum('uc,vc,wc->uvw', U, V, W)
        # Target is 1 at nonzero_indices, 0 elsewhere
        # Error = sum(T_comp^2) - 2 * sum(T_comp at nonzero) + count(nonzero)
        # This avoids creating T_target
        err = np.sum(T_comp * T_comp)
        for u, v, w in nonzero_indices:
            err -= 2.0 * T_comp[u, v, w]
        err += len(nonzero_indices)
        return err

    # Local search for a given problem size
    def local_search(n, m, p, r, time_limit=10.0):
        start_time = time.time()
        nonzero_indices = get_nonzero_indices(n, m, p)
        
        U = np.random.randint(-1, 2, (n*m, r))
        V = np.random.randint(-1, 2, (m*p, r))
        W = np.random.randint(-1, 2, (p*n, r))
        
        current_error = compute_error(U, V, W, nonzero_indices, n, m, p)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = current_error
        
        num_steps = 100000
        T_init = 1.0
        T_final = 0.01
        
        for step in range(num_steps):
            if time.time() - start_time > time_limit:
                break
                
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row = np.random.randint(n*m)
                col = np.random.randint(r)
                old_val = U[row, col]
                new_val = old_val + np.random.randint(-1, 2)
                if new_val == old_val: continue
                U[row, col] = new_val
                mat, idx = U, (row, col)
            elif mat_idx == 1:
                row = np.random.randint(m*p)
                col = np.random.randint(r)
                old_val = V[row, col]
                new_val = old_val + np.random.randint(-1, 2)
                if new_val == old_val: continue
                V[row, col] = new_val
                mat, idx = V, (row, col)
            else:
                row = np.random.randint(p*n)
                col = np.random.randint(r)
                old_val = W[row, col]
                new_val = old_val + np.random.randint(-1, 2)
                if new_val == old_val: continue
                W[row, col] = new_val
                mat, idx = W, (row, col)
                
            new_error = compute_error(U, V, W, nonzero_indices, n, m, p)
            delta = new_error - current_error
            
            T = T_init * (T_final / T_init) ** (step / num_steps)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error == 0:
                        return best_U, best_V, best_W
            else:
                if mat_idx == 0: U[idx[0], idx[1]] = old_val
                elif mat_idx == 1: V[idx[0], idx[1]] = old_val
                else: W[idx[0], idx[1]] = old_val
                
        return best_U, best_V, best_W

    # Strategy 1: Find rank 14 for 3x3x2, then double it to get rank 28 for 3x3x4
    # Time budget: 40s for this part
    time_start = time.time()
    limit_332 = 40.0
    
    found_332 = False
    U_332 = V_332 = W_332 = None
    
    # Try rank 14
    for restart in range(10):
        if time.time() - time_start > limit_332:
            break
        U, V, W = local_search(3, 3, 2, 14, time_limit=3.0)
        if compute_error(U, V, W, get_nonzero_indices(3,3,2), 3,3,2) == 0:
            found_332 = True
            U_332, V_332, W_332 = U, V, W
            break
            
    if found_332:
        # Construct 3x3x4 solution
        # U_total = [U_332, U_332]
        # V_total = [V_332, V_332_shifted]
        # W_total = [W_332, W_332_shifted]
        
        # Wait, the second part is for k in [2,4).
        # The indices for V are j*p + k.
        # For the first part (k=0,1), V indices are j*2+0, j*2+1.
        # For the second part (k=2,3), we need a new V factor.
        # But the problem 3x3x2 is identical for k=2,3 if we shift indices.
        # So we can reuse the same U, V, W factors?
        # No, V and W have different indices.
        # V_1 has non-zeros at (j, 0) and (j, 1).
        # V_2 needs non-zeros at (j, 2) and (j, 3).
        # So V_total will have V_1 in the first r columns and a shifted V_1 in the next r columns?
        # No, V is a matrix of size (m*p, r).
        # For the first part, V_1 is (3*2, 14) = (6, 14).
        # For the second part, we need V_2 of size (6, 14) but placed at the correct indices.
        # The full V is (3*4, 28) = (12, 28).
        # The first 6 rows of V_1 correspond to k=0,1.
        # The next 6 rows of V_2 correspond to k=2,3.
        # So V_total[row, col] = V_1[row, col] for row < 6, col < 14.
        # V_total[row, col] = V_1[row-6, col-14] for row >= 6, col >= 14.
        
        # Similarly for W.
        # W_1 is (p*n, 14) = (2*3, 14) = (6, 14).
        # W_total is (4*3, 28) = (12, 28).
        # W_1 indices are k*n + i.
        # For k=0,1, indices are 0..5.
        # For k=2,3, indices are 6..11.
        # So W_total[row, col] = W_1[row, col] for row < 6, col < 14.
        # W_total[row, col] = W_1[row-6, col-14] for row >= 6, col >= 14.
        
        # U is shared?
        # U_1 is (n*m, 14) = (9, 14).
        # U_total is (9, 28).
        # U_total[:, :14] = U_1.
        # U_total[:, 14:] = U_1.
        
        U_total = np.hstack([U_332, U_332])
        V_total = np.zeros((12, 28))
        V_total[:6, :14] = V_332
        V_total[6:, 14:] = V_332
        
        W_total = np.zeros((12, 28))
        W_total[:6, :14] = W_332
        W_total[6:, 14:] = W_332
        
        # Verify
        nz = get_nonzero_indices(3, 3, 4)
        if compute_error(U_total, V_total, W_total, nz, 3, 3, 4) == 0:
            return {
                "U": U_total.tolist(),
                "V": V_total.tolist(),
                "W": W_total.tolist()
            }

    # Strategy 2: Direct search for 3x3x4
    time_remaining = 120.0 - (time.time() - time_start) - 10.0 # 10s safety
    if time_remaining > 0:
        # Try rank 29
        for r in [29, 30, 31]:
            if time.time() - time_start > 110.0:
                break
            for restart in range(5):
                if time.time() - time_start > 110.0:
                    break
                U, V, W = local_search(3, 3, 4, r, time_limit=5.0)
                if compute_error(U, V, W, get_nonzero_indices(3,3,4), 3,3,4) == 0:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }

    # Fallback
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
