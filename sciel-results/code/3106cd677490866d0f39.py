"""Matrix multiplication 4x4x8 with search for lower rank decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 8]

def _build_2x2x2_strassen():
    """Build the rank-7 Strassen decomposition for 2x2x2."""
    terms = []
    terms.append(([1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    terms.append(([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]))
    terms.append(([1, 0, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]))
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W

def _build_4x4x4_strassen2():
    n = 4
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    U2, V2, W2 = _build_2x2x2_strassen()
    
    block_ops = [
        ([1, 0, 0, -1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [0, 0, 1, -1]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 0, 1, -1], [0, 1, 0, 1]),
        ([1, 0, -1, 0], [1, 0, 1, 0]),
    ]
    c_combos = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    for op_idx in range(7):
        lc, rc = block_ops[op_idx]
        for t in range(7):
            col = op_idx * 7 + t
            for a in range(4):
                if lc[a] != 0:
                    a_row = a // 2
                    a_col = a % 2
                    for li in range(2):
                        for lj in range(2):
                            u_idx = (a_row * 2 + li) * 4 + (a_col * 2 + lj)
                            U[u_idx, col] += lc[a] * U2[2 * li + lj, t]
            for b in range(4):
                if rc[b] != 0:
                    b_row = b // 2
                    b_col = b % 2
                    for lj in range(2):
                        for lk in range(2):
                            v_idx = (b_row * 2 + lj) * 4 + (b_col * 2 + lk)
                            V[v_idx, col] += rc[b] * V2[2 * lj + lk, t]
            for c_blk in range(4):
                c_coeff = c_combos[c_blk][op_idx]
                if c_coeff == 0:
                    continue
                c_row = c_blk // 2
                c_col = c_blk % 2
                for lk in range(2):
                    for li in range(2):
                        w_idx = (c_row * 2 + lk) * 4 + (c_col * 2 + li)
                        W[w_idx, col] += c_coeff * W2[2 * lk + li, t]
    return U, V, W

def _build_4x4x8_rank98():
    n, m, p = 4, 4, 8
    U44, V44, W44 = _build_4x4x4_strassen2()
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    U[:, 0:49] = U44
    V[0:16, 0:49] = V44
    W[0:16, 0:49] = W44
    U[:, 49:98] = U44
    V[16:32, 49:98] = V44
    W[16:32, 49:98] = W44
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _try_find_lower_rank(n, m, p, target_rank, time_limit=5.0):
    """Try to find a decomposition of rank target_rank."""
    start_time = time.time()
    num_u = n * m
    num_v = m * p
    num_w = p * n
    total_vars = target_rank * (num_u + num_v + num_w)
    
    # Target tensor
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0

    def residuals(x):
        U = x[:num_u * target_rank].reshape(num_u, target_rank)
        V = x[num_u * target_rank : num_u * target_rank + num_v * target_rank].reshape(num_v, target_rank)
        W = x[num_u * target_rank + num_v * target_rank :].reshape(num_w, target_rank)
        
        # Compute tensor
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Use einsum for speed
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Need to reshape to 3D for einsum?
        # U[i,j,r] -> U[i*m+j, r]
        # V[j,k,r] -> V[j*p+k, r]
        # W[k,i,r] -> W[k*n+i, r]
        
        # Reshape U to (n, m, R)
        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        W3 = W.reshape(p, n, target_rank)
        
        # Sum over r
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is not a standard einsum because indices are mixed.
        # T[i,j,k] = sum_r U3[i,j,r] V3[j,k,r] W3[k,i,r]
        
        # We can do this in a loop or with careful einsum.
        # Let's use a loop for clarity and correctness, it's only 128 terms.
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # u_idx = i*m+j
                    # v_idx = j*p+k
                    # w_idx = k*n+i
                    T[i,j,k] = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
        
        return (T - target).flatten()

    best_x = None
    best_cost = np.inf
    
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        x0 = np.random.randn(total_vars)
        # Scale down?
        x0 /= np.sqrt(target_rank)
        
        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_x = res.x
                if res.cost < 1e-8:
                    break
        except Exception:
            pass

    if best_cost < 1e-8 and best_x is not None:
        U = best_x[:num_u * target_rank].reshape(num_u, target_rank)
        V = best_x[num_u * target_rank : num_u * target_rank + num_v * target_rank].reshape(num_v, target_rank)
        W = best_x[num_u * target_rank + num_v * target_rank :].reshape(num_w, target_rank)
        
        # Check if it can be rounded to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-5):
            return U_rounded, V_rounded, W_rounded
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find lower rank
    # Start from 97 down to 90
    for r in range(97, 89, -1):
        res = _try_find_lower_rank(n, m, p, r, time_limit=5.0)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to 98
    U, V, W = _build_4x4x8_rank98()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
