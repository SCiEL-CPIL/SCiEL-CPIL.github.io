import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size

    rank = n_blocks ** 3 * 7

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                # M1
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M2
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1

                # M3
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M4
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1

                # M5
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1

                # M6
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M7
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1

    # Precompute the target tensor: T[i,k,j] = 1 for all i,k,j (matrix mult tensor)
    # The decomposition must satisfy: sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t] = 1 for all i,k,j
    # Vectorized check: for a given set of columns, compute the tensor and compare to 1s.
    
    def compute_tensor(U_arr, V_arr, W_arr):
        """Compute the decomposed tensor T[i,k,j] = sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t]"""
        # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
        # T[i,k,j] = sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
        r = U_arr.shape[1]
        if r == 0:
            return np.zeros((n, m, p), dtype=int)
        
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_3d = U_arr.reshape(n, m, r)
        V_3d = V_arr.reshape(m, p, r)
        W_3d = W_arr.reshape(p, n, r)
        
        # T[i,k,j] = sum_t U_3d[i,k,t] * V_3d[k,j,t] * W_3d[j,i,t]
        # Use einsum: sum over t
        T = np.einsum('ikt,kjt,jit->ijk', U_3d, V_3d, W_3d, optimize=True)
        return T
    
    def is_valid_decomposition(U_arr, V_arr, W_arr):
        T = compute_tensor(U_arr, V_arr, W_arr)
        return np.all(T == 1)
    
    def is_valid_decomposition_fast(U_arr, V_arr, W_arr, keep_mask):
        """Check validity using a boolean mask for which columns to keep."""
        r_full = U_arr.shape[1]
        if not np.any(keep_mask):
            return False
        T = compute_tensor(U_arr[:, keep_mask], V_arr[:, keep_mask], W_arr[:, keep_mask])
        return np.all(T == 1)

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Verify initial decomposition
    assert is_valid_decomposition(U, V, W), "Initial decomposition is invalid"

    # Strategy 1: Multiple random-order greedy removals (vectorized check)
    n_trials = 30
    for trial in range(n_trials):
        if time.time() > deadline - 15:
            break
        
        indices = list(range(rank))
        random.shuffle(indices)
        
        U_shuf = U[:, indices]
        V_shuf = V[:, indices]
        W_shuf = W[:, indices]
        
        keep_mask = np.ones(rank, dtype=bool)
        
        # Backward greedy: try removing each column in shuffled order
        for t in range(rank - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            if not keep_mask[t]:
                continue
            
            keep_mask[t] = False
            if not is_valid_decomposition_fast(U_shuf, V_shuf, W_shuf, keep_mask):
                keep_mask[t] = True
        
        if keep_mask.sum() < best_rank:
            best_rank = int(keep_mask.sum())
            best_U = U_shuf[:, keep_mask]
            best_V = V_shuf[:, keep_mask]
            best_W = W_shuf[:, keep_mask]

    # Strategy 2: Forward greedy on the best result found so far
    if time.time() < deadline - 15:
        cur_U, cur_V, cur_W = best_U.copy(), best_V.copy(), best_W.copy()
        cur_rank = cur_U.shape[1]
        
        for t in range(cur_rank):
            if time.time() > deadline - 10:
                break
            
            # Try removing column t
            keep_mask = np.ones(cur_rank, dtype=bool)
            keep_mask[t] = False
            if keep_mask.sum() == 0:
                continue
            
            if is_valid_decomposition_fast(cur_U, cur_V, cur_W, keep_mask):
                cur_U = cur_U[:, keep_mask]
                cur_V = cur_V[:, keep_mask]
                cur_W = cur_W[:, keep_mask]
                cur_rank -= 1
        
        if cur_rank < best_rank:
            best_rank = cur_rank
            best_U, best_V, best_W = cur_U, cur_V, cur_W

    # Strategy 3: Try to replace pairs of columns with a single column (merge)
    # For each pair (a,b), check if column a can be expressed as a linear combo of others after removing b,
    # or more simply: try removing two columns at once.
    if time.time() < deadline - 15:
        cur_rank = best_U.shape[1]
        if cur_rank > 2:
            for a in range(cur_rank):
                if time.time() > deadline - 10:
                    break
                for b in range(a + 1, cur_rank):
                    if time.time() > deadline - 10:
                        break
                    keep_mask = np.ones(cur_rank, dtype=bool)
                    keep_mask[a] = False
                    keep_mask[b] = False
                    if keep_mask.sum() == 0:
                        continue
                    if is_valid_decomposition_fast(best_U, best_V, best_W, keep_mask):
                        best_U = best_U[:, keep_mask]
                        best_V = best_V[:, keep_mask]
                        best_W = best_W[:, keep_mask]
                        best_rank = int(keep_mask.sum())
                        cur_rank = best_rank
                        break  # restart inner loop with new cur_rank

    # Strategy 4: Another round of random greedy on the improved result
    if time.time() < deadline - 15 and best_U.shape[1] > 0:
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            cur_rank = best_U.shape[1]
            indices = list(range(cur_rank))
            random.shuffle(indices)
            
            U_shuf = best_U[:, indices]
            V_shuf = best_V[:, indices]
            W_shuf = best_W[:, indices]
            
            keep_mask = np.ones(cur_rank, dtype=bool)
            
            for t in range(cur_rank - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                if not keep_mask[t]:
                    continue
                keep_mask[t] = False
                if not is_valid_decomposition_fast(U_shuf, V_shuf, W_shuf, keep_mask):
                    keep_mask[t] = True
            
            if keep_mask.sum() < best_rank:
                best_rank = int(keep_mask.sum())
                best_U = U_shuf[:, keep_mask]
                best_V = V_shuf[:, keep_mask]
                best_W = W_shuf[:, keep_mask]

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
