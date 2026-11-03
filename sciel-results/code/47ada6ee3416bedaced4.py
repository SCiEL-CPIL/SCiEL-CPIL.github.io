"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses Strassen block decomposition as baseline, then attempts rank reduction
via ALS with half-integer rounding, SVD-based subspace elimination, and
gradient-based polishing."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=atol)

def _compute_error(U, V, W, n, m, p, rank):
    """Compute the L2 error of the decomposition."""
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.sqrt(np.sum((T - 1.0)**2))

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _kron_strassen_construct(n, m, p):
    """Rank-65: Kronecker Strassen for 4x4x4 (rank 49) + naive for 4x4x1 (rank 16)."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * m + j] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * p + k] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] = val1 * val2
            
            add_term(u, v, w)
    
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i * m + j] = 1
            v = np.zeros(m*p); v[j * p + 4] = 1
            w = np.zeros(p*n); w[4 * n + i] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _strassen_block_construct(n, m, p):
    """Rank-72: 2x2 block Strassen for 4x4x5."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(gi, gj): return gi * m + gj
    def get_v_idx(gj, gk): return gj * p + gk
    def get_w_idx(gk, gi): return gk * n + gi
    
    strassen_terms = _strassen_2x2x2_terms()
    
    def add_strassen_2x2_2(ai, aj, bj, bk):
        for u_sparse, v_sparse, w_sparse in strassen_terms:
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for idx, val in u_sparse:
                ui, uj = idx // 2, idx % 2
                u[get_u_idx(ai+ui, aj+uj)] = val
            for idx, val in v_sparse:
                vj, vk = idx // 2, idx % 2
                v[get_v_idx(bj+vj, bk+vk)] = val
            for idx, val in w_sparse:
                wk, wi = idx // 2, idx % 2
                w[get_w_idx(bk+wk, ai+wi)] = val
            add_term(u, v, w)
    
    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)
    
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _als_decompose(n, m, p, rank, seed=42, max_iters=300, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=80):
    """ALS with half-integer rounding."""
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _als_iteration():
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-12:
            break
    
    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-14:
            break
    
    candidates = []
    U_r1 = _round_matrix_to_half_int(best_U)
    V_r1 = _round_matrix_to_half_int(best_V)
    W_r1 = _round_matrix_to_half_int(best_W)
    candidates.append((U_r1, V_r1, W_r1))
    
    U_r2 = np.round(best_U)
    V_r2 = np.round(best_V)
    W_r2 = np.round(best_W)
    candidates.append((U_r2, V_r2, W_r2))
    
    for offset in [0.0, 0.25, -0.25, 0.125, -0.125]:
        U_r3 = _round_matrix_to_half_int(best_U + offset)
        V_r3 = _round_matrix_to_half_int(best_V + offset)
        W_r3 = _round_matrix_to_half_int(best_W + offset)
        candidates.append((U_r3, V_r3, W_r3))
    
    for Uc, Vc, Wc in candidates:
        if _verify_fast(Uc.tolist(), Vc.tolist(), Wc.tolist(), n, m, p, atol=1e-9):
            return Uc.tolist(), Vc.tolist(), Wc.tolist()
    
    return None

def _svd_rank_reduction(U, V, W, n, m, p, target_rank, deadline=None):
    """Try to reduce rank by finding linear dependencies via SVD of the term matrices.
    
    The key insight: if the rank-r decomposition has linearly dependent terms,
    we can project onto a lower-dimensional subspace.
    """
    rank = U.shape[1]
    
    # Build the "term matrix" for each factor
    # For U: shape (n*m, rank)
    # Check if we can find a rank-target_rank subspace that still works
    
    # Approach: try random projections and re-optimize
    for attempt in range(10):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(attempt * 7 + target_rank)
        
        if attempt == 0:
            # Keep first target_rank columns
            U_init = U[:, :target_rank]
            V_init = V[:, :target_rank]
            W_init = W[:, :target_rank]
        elif attempt == 1:
            # Keep last target_rank columns
            U_init = U[:, rank-target_rank:]
            V_init = V[:, rank-target_rank:]
            W_init = W[:, rank-target_rank:]
        elif attempt == 2:
            # SVD-based: project U to target_rank via SVD
            _, _, Vt_U = np.linalg.svd(U, full_matrices=False)
            U_init = U @ Vt_U[:target_rank].T
            _, _, Vt_V = np.linalg.svd(V, full_matrices=False)
            V_init = V @ Vt_V[:target_rank].T
            _, _, Vt_W = np.linalg.svd(W, full_matrices=False)
            W_init = W @ Vt_W[:target_rank].T
        elif attempt == 3:
            # Random rotation
            Q = np.random.randn(rank, target_rank)
            Q, _ = np.linalg.qr(Q)
            U_init = U @ Q
            V_init = V @ Q
            W_init = W @ Q
        elif attempt == 4:
            # Random subset with perturbation
            idx = np.random.choice(rank, target_rank, replace=False)
            U_init = U[:, idx] + np.random.randn(n*m, target_rank) * 0.05
            V_init = V[:, idx] + np.random.randn(m*p, target_rank) * 0.05
            W_init = W[:, idx] + np.random.randn(p*n, target_rank) * 0.05
        elif attempt == 5:
            # SVD with mixing
            _, S_U, Vt_U = np.linalg.svd(U, full_matrices=False)
            _, S_V, Vt_V = np.linalg.svd(V, full_matrices=False)
            _, S_W, Vt_W = np.linalg.svd(W, full_matrices=False)
            # Use top components but with random mixing
            mix = np.random.randn(target_rank, target_rank)
            U_init = U @ Vt_U[:target_rank].T @ mix
            V_init = V @ Vt_V[:target_rank].T @ mix
            W_init = W @ Vt_W[:target_rank].T @ mix
        else:
            # Gradient-based: start from full and zero out some
            U_init = U[:, :target_rank].copy()
            V_init = V[:, :target_rank].copy()
            W_init = W[:, :target_rank].copy()
            # Add small random perturbation
            U_init += np.random.randn(n*m, target_rank) * 0.1
            V_init += np.random.randn(m*p, target_rank) * 0.1
            W_init += np.random.randn(p*n, target_rank) * 0.1
        
        result = _als_decompose(n, m, p, target_rank,
                                U_init=U_init, V_init=V_init, W_init=W_init,
                                max_iters=200, deadline=deadline, polish_iters=60)
        if result is not None:
            return result
    
    return None

def _gradient_polish(U, V, W, n, m, p, rank, deadline=None, max_iters=200):
    """Polish a near-solution using gradient descent on the error."""
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    
    lr = 0.01
    best_err = _compute_error(U, V, W, n, m, p, rank)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradients via finite differences (expensive but accurate)
        # Use a simpler approach: ALS with very small step
        # Actually, let's do a proper gradient step
        
        # Gradient of error w.r.t. U[i*m+j]
        # err^2 = sum_{i,j,k} (T[i,j,k] - 1)^2
        # dT[i,j,k]/dU[i*m+j] = V[j*p+k] * W[k*n+i]
        # d(err^2)/dU[i*m+j] = 2 * sum_{i',j',k'} (T[i',j',k']-1) * dT[i',j',k']/dU[i*m+j]
        # = 2 * sum_{k'} (T[i,j,k']-1) * V[j*p+k'] * W[k*n+i]
        
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        # Compute T
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            uc = Ur[:, :, c]
            vc = Vr[:, :, c]
            wc = Wr[:, :, c]
            wcT = wc.T
            for j in range(m):
                T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
        
        # Gradient for U
        dU = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    dU[i*m+j] += 2 * (T[i,j,k] - 1) * V[j*p+k] * W[k*n+i]
        
        # Gradient for V
        dV = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    dV[j*p+k] += 2 * (T[i,j,k] - 1) * U[i*m+j] * W[k*n+i]
        
        # Gradient for W
        dW = np.zeros_like(W)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    dW[k*n+i] += 2 * (T[i,j,k] - 1) * U[i*m+j] * V[j*p+k]
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm > 0:
            step = lr * err / (grad_norm + 1e-10)
            U -= step * dU
            V -= step * dV
            W -= step * dW
        
        # Decay learning rate
        lr *= 0.995
    
    # Try rounding the polished result
    candidates = []
    U_r1 = _round_matrix_to_half_int(best_U)
    V_r1 = _round_matrix_to_half_int(best_V)
    W_r1 = _round_matrix_to_half_int(best_W)
    candidates.append((U_r1, V_r1, W_r1))
    
    U_r2 = np.round(best_U)
    V_r2 = np.round(best_V)
    W_r2 = np.round(best_W)
    candidates.append((U_r2, V_r2, W_r2))
    
    for Uc, Vc, Wc in candidates:
        if _verify_fast(Uc.tolist(), Vc.tolist(), Wc.tolist(), n, m, p, atol=1e-9):
            return Uc.tolist(), Vc.tolist(), Wc.tolist()
    
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build baselines
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    kron_ok = _verify_fast(U_kron.tolist(), V_kron.tolist(), W_kron.tolist(), n, m, p, atol=1e-9)
    
    U_2x2, V_2x2, W_2x2 = _strassen_block_construct(n, m, p)
    twox2_ok = _verify_fast(U_2x2.tolist(), V_2x2.tolist(), W_2x2.tolist(), n, m, p, atol=1e-9)
    
    if not kron_ok and not twox2_ok:
        return _schoolbook()
    
    # Use the lower-rank valid baseline
    if kron_ok and (not twox2_ok or U_kron.shape[1] <= U_2x2.shape[1]):
        U_base, V_base, W_base = U_kron, V_kron, W_kron
    else:
        U_base, V_base, W_base = U_2x2, V_2x2, W_2x2
    
    best_result = {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    best_rank = U_base.shape[1]
    
    # Try rank reduction via SVD-based methods
    if time.time() < deadline - 60:
        for target_rank in range(best_rank - 1, max(40, best_rank - 20), -1):
            if time.time() > deadline - 60:
                break
            result = _svd_rank_reduction(U_base, V_base, W_base, n, m, p, target_rank, deadline=deadline - 60)
            if result is not None:
                U_r, V_r, W_r = result
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = target_rank
                break
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(45, best_rank - 25), -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 30:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=300,
                                    deadline=deadline - 30, polish_iters=60)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
        else:
            break
    
    # Final gradient polish attempt on best result
    if time.time() < deadline - 10 and best_rank < 80:
        U_b = np.array(best_result["U"], dtype=np.float64)
        V_b = np.array(best_result["V"], dtype=np.float64)
        W_b = np.array(best_result["W"], dtype=np.float64)
        polished = _gradient_polish(U_b, V_b, W_b, n, m, p, best_rank, deadline=deadline - 10, max_iters=100)
        if polished is not None:
            best_result = {"U": polished[0], "V": polished[1], "W": polished[2]}
    
    return best_result
# EVOLVE-BLOCK-END
