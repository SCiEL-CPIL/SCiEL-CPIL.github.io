import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Target Tensor T
    # T[i, j, k] = 1 if indices match
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T is 1 at (i*m+j, j*p+k, k*n+i)
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int32)
        V = np.zeros((m*p, rank), dtype=np.int32)
        W = np.zeros((p*n, rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = schoolbook()
    best_rank = n * m * p

    # Discrete Local Search
    # We search for U, V, W with entries in {-1, 0, 1}
    
    def compute_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # S = sum_r U[:, r] o V[:, r] o W[:, r]
        # Using einsum
        S = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.sum(np.abs(S - T))

    def local_search(target_rank, max_iters=50000):
        # Initialize randomly
        U = np.random.randint(-1, 2, (n*m, target_rank))
        V = np.random.randint(-1, 2, (m*p, target_rank))
        W = np.random.randint(-1, 2, (p*n, target_rank))
        
        # Precompute K_r = V[:, r] o W[:, r]
        # K shape: (R, m*p, p*n)
        K = np.einsum('br,cr->bcr', V, W, optimize=True)
        
        S = np.einsum('ar,bcr->abc', U, K, optimize=True)
        current_error = np.sum(np.abs(S - T))
        
        values = np.array([-1, 0, 1])
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick random matrix and index
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                i = random.randint(0, n*m - 1)
                r = random.randint(0, target_rank - 1)
                old_val = U[i, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in S is delta * K[r]
                # Check error change
                # We can approximate or compute exact
                # Exact: new_S = S + delta * K[r]
                # new_error = sum(abs(new_S - T))
                # This is O(m*p*p*n) which is 12*12=144 ops. Very fast.
                new_S = S + delta * K[r]
                new_error = np.sum(np.abs(new_S - T))
                if new_error < current_error:
                    U[i, r] = new_val
                    S = new_S
                    current_error = new_error
            elif mat_choice == 1:
                j = random.randint(0, m*p - 1)
                r = random.randint(0, target_rank - 1)
                old_val = V[j, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in S: U[:, r] o (delta * e_j) o W[:, r]
                # S_new[a, b, c] = S[a, b, c] + delta * U[a, r] * (1 if b==j else 0) * W[c, r]
                # This is a rank-1 update in the (a,c) plane for fixed b=j
                # We need to update S[:, j, :]
                # S[:, j, :] += delta * U[:, r] * W[:, r]
                # We need to compute error change.
                # old_col = S[:, j, :]
                # new_col = old_col + delta * U[:, r] * W[:, r]
                # old_err_part = sum(abs(old_col - T[:, j, :]))
                # new_err_part = sum(abs(new_col - T[:, j, :]))
                # total_new_error = current_error - old_err_part + new_err_part
                
                old_col = S[:, j, :]
                update_vec = delta * U[:, r] * W[:, r]
                new_col = old_col + update_vec
                old_err_part = np.sum(np.abs(old_col - T[:, j, :]))
                new_err_part = np.sum(np.abs(new_col - T[:, j, :]))
                new_error = current_error - old_err_part + new_err_part
                
                if new_error < current_error:
                    V[j, r] = new_val
                    S[:, j, :] = new_col
                    # Update K[r] because V changed
                    # K[r] = V[:, r] o W[:, r]
                    # K[r, :, :] = V[:, r] o W[:, r]
                    # We only changed V[j, r], so K[r, j, :] changed?
                    # K[r, b, c] = V[b, r] * W[c, r]
                    # So K[r, j, :] = new_val * W[:, r]
                    K[r, j, :] = new_val * W[:, r]
                    current_error = new_error
            else:
                k = random.randint(0, p*n - 1)
                r = random.randint(0, target_rank - 1)
                old_val = W[k, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in S: U[:, r] o V[:, r] o (delta * e_k)
                # S[:, :, k] += delta * U[:, r] * V[:, r]
                
                old_col = S[:, :, k]
                update_vec = delta * U[:, r] * V[:, r]
                new_col = old_col + update_vec
                old_err_part = np.sum(np.abs(old_col - T[:, :, k]))
                new_err_part = np.sum(np.abs(new_col - T[:, :, k]))
                new_error = current_error - old_err_part + new_err_part
                
                if new_error < current_error:
                    W[k, r] = new_val
                    S[:, :, k] = new_col
                    # Update K[r]
                    # K[r, b, c] = V[b, r] * W[c, r]
                    # K[r, :, k] = V[:, r] * new_val
                    K[r, :, k] = V[:, r] * new_val
                    current_error = new_error

            if current_error == 0:
                return U, V, W
        
        return None

    # Try ranks from 25 to 35
    for target_rank in range(25, 36):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try a few restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            res = local_search(target_rank, max_iters=10000)
            if res is not None:
                U, V, W = res
                # Verify
                S = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                if np.array_equal(S, T):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return best_solution
