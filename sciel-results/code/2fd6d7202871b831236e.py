import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check of the decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 4 * 7 = 28.
    
    U has shape (16, 28), V has shape (8, 28), W has shape (8, 28).
    U[i*4+j, r], V[j*2+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c

    # 4 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Build 4x4x4 decomposition using Kronecker product of 2x2x2 Strassen.
    Rank = 7 * 7 = 49.
    U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            # U part: U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1 * 2 + j1, r1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2 * 2 + j2, r2]
                            if c2 == 0: continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, col] += c1 * c2
            
            # V part: V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1 * 2 + k1, r1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2 * 2 + k2, r2]
                            if c2 == 0: continue
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, col] += c1 * c2
            
            # W part: W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1 * 2 + i1, r1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2 * 2 + i2, r2]
                            if c2 == 0: continue
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, col] += c1 * c2
    
    return U, V, W

def _build_4x5x6_improved():
    """
    Build 4x5x6 decomposition using 4x4x4 Strassen and 4x4x2 Strassen.
    
    Strategy:
    - Split A (4x5) into A_L (4x4) and A_R (4x1)
    - Split B (5x6) into B_T (4x6) and B_B (1x6)
    - C = A_L @ B_T + A_R @ B_B
    
    For A_L @ B_T (4x4 @ 4x6):
      - Split B_T into B_T1 (4x4) and B_T2 (4x2)
      - Use 4x4x4 Strassen for A_L @ B_T1 (rank 49)
      - Use 4x4x2 Strassen for A_L @ B_T2 (rank 28)
    
    For A_R @ B_B (4x1 @ 1x6):
      - Schoolbook: rank 24
    
    Total rank: 49 + 28 + 24 = 101
    """
    n, m, p = 4, 5, 6
    
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    
    total_rank = 49 + 28 + 24
    U = np.zeros((n * m, total_rank))  # 4x5 = 20 rows
    V = np.zeros((m * p, total_rank))  # 5x6 = 30 rows
    W = np.zeros((p * n, total_rank))  # 6x4 = 24 rows
    
    # Part 1: A_L @ B_T1 (4x4 @ 4x4)
    base = 0
    # U: A_L (4x4) -> rows i*5+j
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+49] = U444[i * 4 + j, :]
    
    # V: B_T1 (4x4) -> rows j*6+k
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, base:base+49] = V444[j * 4 + k, :]
    
    # W: C (4x4 part) -> rows k*4+i
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, base:base+49] = W444[k * 4 + i, :]
    
    # Part 2: A_L @ B_T2 (4x4 @ 4x2)
    base = 49
    # U: A_L (4x4) -> rows i*5+j
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
    
    # V: B_T2 (4x2) -> rows j*6+k (k=4,5)
    for j in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            V[j * 6 + k_global, base:base+28] = V442[j * 2 + k_local, :]
    
    # W: C (4x2 part) -> rows k*4+i (k=4,5)
    for k_local in range(2):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global * 4 + i, base:base+28] = W442[k_local * 4 + i, :]
    
    # Part 3: A_R @ B_B (4x1 @ 1x6)
    base = 49 + 28
    for i in range(4):
        for k in range(6):
            col = base + i * 6 + k
            U[i * 5 + 4, col] = 1.0
            V[4 * 6 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_tensor_matrix_fast(U, V, W, n, m, p):
    """
    Build the tensor matrix using vectorized operations.
    Returns M of shape (n*m*p, R) where M[idx, r] is the contribution of rank-r term.
    """
    R = U.shape[1]
    U3 = U.T.reshape(n, m, R)
    V3 = V.T.reshape(m, p, R)
    W3 = W.T.reshape(p, n, R)
    # For each r, the tensor is U3[:,:,r] * V3[:,:,r] * W3[:,:,r]
    # Flatten to (n*m*p, R)
    T = np.einsum('ijr,jkr,kir->ijk r', U3, V3, W3)
    return T.reshape(-1, R)

def _local_reduce_fast(U, V, W, n, m, p, max_iters=200, budget_s=45.0):
    """
    Fast local column-removal using vectorized tensor matrix.
    Key insight: sum of all columns must equal ones vector.
    If we remove column c, the new sum = old_sum - M[:,c] must still equal ones.
    So column c is removable iff M[:,c] == 0 (in exact arithmetic) or
    more generally, we need to check if the remaining columns still sum to ones.
    
    Better approach: Precompute the tensor matrix. Then for any subset of columns,
    the sum is just the sum of corresponding columns of M.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    dim = n * m * p
    target = np.ones(dim)
    
    # Build tensor matrix once
    M = _build_tensor_matrix_fast(U, V, W, n, m, p)
    
    # Current sum of all columns
    current_sum = M.sum(axis=1)
    
    # Verify initial
    if not np.allclose(current_sum, target, atol=1e-6):
        return U, V, W
    
    for _ in range(max_iters):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        cols = list(range(R))
        rng.shuffle(cols)
        
        for c in cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            # If we remove column c, new sum = current_sum - M[:,c]
            new_sum = current_sum - M[:, c]
            if np.allclose(new_sum, target, atol=1e-6):
                # Remove column c
                keep_mask = np.ones(R, dtype=bool)
                keep_mask[c] = False
                U = U[:, keep_mask]
                V = V[:, keep_mask]
                W = W[:, keep_mask]
                R -= 1
                # Update M
                M = M[:, keep_mask]
                current_sum = M.sum(axis=1)
                found = True
                break
        if not found:
            break
    
    # Final exact verification
    if _verify_decomp(U, V, W, n, m, p, 1e-9):
        return U, V, W
    
    # If exact verification fails, try to recover
    return U, V, W

def _local_reduce_pairs(U, V, W, n, m, p, max_iters=100, budget_s=30.0):
    """
    Try removing pairs of columns simultaneously.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    dim = n * m * p
    target = np.ones(dim)
    
    M = _build_tensor_matrix_fast(U, V, W, n, m, p)
    current_sum = M.sum(axis=1)
    
    if not np.allclose(current_sum, target, atol=1e-6):
        return U, V, W
    
    for _ in range(max_iters):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        if R < 3:
            break
        found = False
        cols = list(range(R))
        rng.shuffle(cols)
        
        for c1 in cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            for c2 in cols:
                if c2 == c1 or c2 <= c1:
                    continue
                if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                    break
                new_sum = current_sum - M[:, c1] - M[:, c2]
                if np.allclose(new_sum, target, atol=1e-6):
                    keep_mask = np.ones(R, dtype=bool)
                    keep_mask[[c1, c2]] = False
                    U = U[:, keep_mask]
                    V = V[:, keep_mask]
                    W = W[:, keep_mask]
                    R -= 2
                    M = M[:, keep_mask]
                    current_sum = M.sum(axis=1)
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    if _verify_decomp(U, V, W, n, m, p, 1e-9):
        return U, V, W
    return U, V, W

def _try_optimize_rank(U, V, W, n, m, p, target_rank, max_nfev=200, n_restarts=3):
    """
    Try to optimize the decomposition to a lower target rank using least squares.
    Multiple restarts with different initializations.
    """
    from scipy.optimize import least_squares
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.ones((n, m, p), dtype=np.float64)
    
    num_params = target_rank * (dim_U + dim_V + dim_W)
    
    best_result = None
    best_cost = np.inf
    
    for restart in range(n_restarts):
        if _time_left() < 10.0:
            break
        
        rng = np.random.RandomState(42 + restart * 7)
        
        # Initialize: take first target_rank columns + random perturbation
        U_seed = U[:, :min(target_rank, U.shape[1])]
        V_seed = V[:, :min(target_rank, V.shape[1])]
        W_seed = W[:, :min(target_rank, W.shape[1])]
        
        if target_rank > U_seed.shape[1]:
            pad = target_rank - U_seed.shape[1]
            U_seed = np.hstack([U_seed, rng.normal(0, 0.1, (dim_U, pad))])
            V_seed = np.hstack([V_seed, rng.normal(0, 0.1, (dim_V, pad))])
            W_seed = np.hstack([W_seed, rng.normal(0, 0.1, (dim_W, pad))])
        
        x0_parts = []
        for r in range(target_rank):
            x0_parts.append(U_seed[:, r].flatten())
        for r in range(target_rank):
            x0_parts.append(V_seed[:, r].flatten())
        for r in range(target_rank):
            x0_parts.append(W_seed[:, r].flatten())
        x0 = np.concatenate(x0_parts)
        
        # Add noise
        x0 += rng.normal(0, 0.05, num_params)
        
        def residual(x, R):
            U_flat = x[0 : R * dim_U]
            V_flat = x[R * dim_U : R * dim_U + R * dim_V]
            W_flat = x[R * dim_U + R * dim_V : ]
            
            U = U_flat.reshape(dim_U, R)
            V = V_flat.reshape(dim_V, R)
            W = W_flat.reshape(dim_W, R)
            
            U_3d = U.T.reshape(n, m, R)
            V_3d = V.T.reshape(m, p, R)
            W_3d = W.T.reshape(p, n, R)
            
            tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            res = tensor_sum - T
            return res.flatten()
        
        try:
            res = least_squares(residual, x0, args=(target_rank,), method='lm', 
                               max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if res.cost < best_cost:
                best_cost = res.cost
                if res.cost < 1e-10:
                    U_flat = res.x[0 : target_rank * dim_U]
                    V_flat = res.x[target_rank * dim_U : target_rank * dim_U + target_rank * dim_V]
                    W_flat = res.x[target_rank * dim_U + target_rank * dim_V : ]
                    
                    U_out = U_flat.reshape(dim_U, target_rank)
                    V_out = V_flat.reshape(dim_V, target_rank)
                    W_out = W_flat.reshape(dim_W, target_rank)
                    
                    # Verify
                    U3 = U_out.T.reshape(n, m, target_rank)
                    V3 = V_out.T.reshape(m, p, target_rank)
                    W3 = W_out.T.reshape(p, n, target_rank)
                    T_out = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                    error = np.max(np.abs(T_out - 1.0))
                    
                    if error < 1e-8:
                        # Try to round to half-integers
                        U_half = np.round(U_out * 2) / 2
                        V_half = np.round(V_out * 2) / 2
                        W_half = np.round(W_out * 2) / 2
                        
                        U3h = U_half.T.reshape(n, m, target_rank)
                        V3h = V_half.T.reshape(m, p, target_rank)
                        W3h = W_half.T.reshape(p, n, target_rank)
                        T_out_h = np.einsum('ijr,jkr,kir->ijk', U3h, V3h, W3h)
                        error_h = np.max(np.abs(T_out_h - 1.0))
                        
                        if error_h < 1e-8:
                            best_result = (U_half, V_half, W_half)
                            break
                        else:
                            best_result = (U_out, V_out, W_out)
        except Exception:
            pass
    
    return best_result

def construct():
    n, m, p = 4, 5, 6
    
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Build the improved 4x5x6 decomposition
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_improved()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Fast local column removal
    try:
        if _time_left() > 20.0:
            cand = _local_reduce_fast(best_U, best_V, best_W, n, m, p, max_iters=200, budget_s=30.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Pair removal
    try:
        if _time_left() > 15.0:
            cand = _local_reduce_pairs(best_U, best_V, best_W, n, m, p, max_iters=100, budget_s=20.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Second pass of single removal after pair removal
    try:
        if _time_left() > 10.0:
            cand = _local_reduce_fast(best_U, best_V, best_W, n, m, p, max_iters=200, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Try optimization to lower ranks
    try:
        if _time_left() > 25.0:
            for delta in range(1, min(8, best_rank)):
                if _time_left() < 15.0:
                    break
                target = best_rank - delta
                if target < 1:
                    break
                result = _try_optimize_rank(best_U, best_V, best_W, n, m, p, target, max_nfev=150, n_restarts=3)
                if result is not None:
                    U_opt, V_opt, W_opt = result
                    if _verify_decomp(U_opt, V_opt, W_opt, n, m, p, 1e-9):
                        best_rank = target
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        break
    except Exception:
        pass
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
