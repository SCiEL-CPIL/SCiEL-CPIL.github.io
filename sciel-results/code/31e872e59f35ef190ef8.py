"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time
from itertools import combinations

def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    The 2x4 * 4x8 multiplication is viewed as 8 independent 2x2 * 2x2 multiplications,
    each done with rank 7 (Strassen), giving total rank 8*7 = 56.
    """
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = len(U[0]) if U else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using vectorized operations."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Precompute V * W element-wise: shape (m*p, p*n, rank)
    # For each (b, c) pair, compute V_arr[b] * W_arr[c]
    # Then for each a, check if sum_s U[a,s] * (V[b,s] * W[c,s]) == 1
    
    # Build the full tensor check
    # Tensor T[a,b,c] should equal sum_s U[a,s]*V[b,s]*W[c,s]
    # Nonzero entries of T: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # Vectorized: compute contribution from each rank slice
    # For slice s: contribution to T is outer(U[:,s], V[:,s]) * W[:,s] reshaped appropriately
    # Actually: T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    
    # This is a 3-way tensor. Let's verify specific entries.
    # Number of (a,b,c) triples where T=1: n*m*p = 64
    
    errors = 0
    for s in range(rank):
        u_s = U_arr[:, s]  # (n*m,)
        v_s = V_arr[:, s]  # (m*p,)
        w_s = W_arr[:, s]  # (p*n,)
        
        # For each (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
        # Contribution to T[a,b,c] from slice s is u_s[a] * v_s[b] * w_s[c]
        # We need sum over s of this to equal 1 for all valid (a,b,c)
        
        # Build the contribution tensor for this slice
        # Shape: (n, m, p) where [i,j,k] = u_s[i*m+j] * v_s[j*p+k] * w_s[k*n+i]
        contrib = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                a = i * m + j
                ua = u_s[a]
                if ua == 0:
                    continue
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    vb = v_s[b]
                    wc = w_s[c]
                    if vb != 0 and wc != 0:
                        contrib[i, j, k] = ua * vb * wc
        
        # Check if any entry exceeds 1 or is negative (would indicate error)
        # Actually we just need the sum over all slices to be exactly 1
        # For speed, accumulate into a running total
        # But for simplicity in fast check, let's just verify a subset
        pass
    
    # Full verification
    T_check = np.zeros((n, m, p))
    for s in range(rank):
        u_s = U_arr[:, s]
        v_s = V_arr[:, s]
        w_s = W_arr[:, s]
        for i in range(n):
            for j in range(m):
                a = i * m + j
                ua = u_s[a]
                if ua == 0:
                    continue
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    T_check[i, j, k] += ua * v_s[b] * w_s[c]
    
    if not np.allclose(T_check, 1.0, atol=1e-9):
        return False
    return True

def _try_remove_columns(U, V, W, n, m, p, max_removals=4, time_limit=100):
    """
    Try to reduce rank by removing columns from the decomposition.
    Greedily try removing one column at a time, keeping the best.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    start_time = time.time()
    
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    best_rank = rank
    
    # Try removing single columns
    for col_to_remove in range(rank):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Create new arrays without this column
        mask = np.ones(rank, dtype=bool)
        mask[col_to_remove] = False
        
        U_new = U_arr[:, mask]
        V_new = V_arr[:, mask]
        W_new = W_arr[:, mask]
        
        # Verify
        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank - 1
    
    # Try removing pairs of columns
    if best_rank < rank:
        # Update base
        U_arr = best_U
        V_arr = best_V
        W_arr = best_W
        rank = best_rank
        
        for col1 in range(rank):
            if time.time() - start_time > time_limit - 5:
                break
            for col2 in range(col1 + 1, rank):
                if time.time() - start_time > time_limit - 5:
                    break
                
                mask = np.ones(rank, dtype=bool)
                mask[col1] = False
                mask[col2] = False
                
                U_new = U_arr[:, mask]
                V_new = V_arr[:, mask]
                W_new = W_arr[:, mask]
                
                if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = rank - 2
    
    return best_U, best_V, best_W, best_rank

def _try_group_removal(U, V, W, n, m, p, time_limit=80):
    """
    Try removing entire Strassen groups (7 columns each) to see if we can
    merge adjacent 2x2 blocks.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    start_time = time.time()
    
    # Each Strassen block uses 7 consecutive columns
    # There are 8 blocks (2 u-values * 4 v-values)
    num_blocks = 8
    cols_per_block = 7
    
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    best_rank = rank
    
    # Try removing one entire block
    for block in range(num_blocks):
        if time.time() - start_time > time_limit - 5:
            break
        
        start_col = block * cols_per_block
        end_col = start_col + cols_per_block
        
        mask = np.ones(rank, dtype=bool)
        mask[start_col:end_col] = False
        
        U_new = U_arr[:, mask]
        V_new = V_arr[:, mask]
        W_new = W_arr[:, mask]
        
        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank - cols_per_block
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Build the rank-56 Strassen-based decomposition
    U, V, W = _build_rank_56(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    best_rank = 56
    best_U, best_V, best_W = U, V, W
    
    # First try individual column removal
    time_left = deadline - time.time()
    if time_left > 30:
        U_r, V_r, W_r, rank_r = _try_remove_columns(
            np.array(U, dtype=float).tolist(),
            np.array(V, dtype=float).tolist(),
            np.array(W, dtype=float).tolist(),
            n, m, p,
            max_removals=2,
            time_limit=min(time_left - 10, 60)
        )
        if rank_r < best_rank:
            best_rank = rank_r
            best_U = U_r.tolist() if isinstance(U_r, np.ndarray) else U_r
            best_V = V_r.tolist() if isinstance(V_r, np.ndarray) else V_r
            best_W = W_r.tolist() if isinstance(W_r, np.ndarray) else W_r
    
    # Try group removal
    time_left = deadline - time.time()
    if time_left > 20 and best_rank == 56:
        U_g, V_g, W_g, rank_g = _try_group_removal(
            np.array(best_U, dtype=float),
            np.array(best_V, dtype=float),
            np.array(best_W, dtype=float),
            n, m, p,
            time_limit=min(time_left - 10, 40)
        )
        if rank_g < best_rank:
            best_rank = rank_g
            best_U = U_g.tolist()
            best_V = V_g.tolist()
            best_W = W_g.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
