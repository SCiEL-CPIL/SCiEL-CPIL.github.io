"""
Kronecker-embedded Strassen (rank 65) + optimized rank reduction via
fast einsum verification, structured projections, and iterative peeling.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """7 Strassen terms for 2x2x2. Each: (u_sparse, v_sparse, w_sparse)."""
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """Kronecker Strassen (49) + naive 4x4x1 (16) = 65 terms."""
    U_cols, V_cols, W_cols = [], [], []
    strassen = _strassen_2x2x2_terms()
    
    def gU(i,j): return i*m+j
    def gV(j,k): return j*p+k
    def gW(k,i): return k*n+i
    
    for (u1s, v1s, w1s) in strassen:
        for (u2s, v2s, w2s) in strassen:
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for (i1, val1) in u1s:
                ii1, jj1 = i1//2, i1%2
                for (i2, val2) in u2s:
                    ii2, jj2 = i2//2, i2%2
                    u[gU(ii1*2+ii2, jj1*2+jj2)] = val1*val2
            for (i1, val1) in v1s:
                jj1, kk1 = i1//2, i1%2
                for (i2, val2) in v2s:
                    jj2, kk2 = i2//2, i2%2
                    v[gV(jj1*2+jj2, kk1*2+kk2)] = val1*val2
            for (i1, val1) in w1s:
                kk1, ii1 = i1//2, i1%2
                for (i2, val2) in w2s:
                    kk2, ii2 = i2//2, i2%2
                    w[gW(kk1*2+kk2, ii1*2+ii2)] = val1*val2
            U_cols.append(u); V_cols.append(v); W_cols.append(w)
    
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[gU(i,j)] = 1
            v = np.zeros(m*p); v[gV(j,4)] = 1
            w = np.zeros(p*n); w[gW(4,i)] = 1
            U_cols.append(u); V_cols.append(v); W_cols.append(w)
    
    return np.array(U_cols).T, np.array(V_cols).T, np.array(W_cols).T

def _build_coefficients(n, m, p):
    """
    Precompute coefficient matrices for fast verification.
    For each (i,j,k) in the tensor, we need:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    We can rewrite: for fixed (i,j,k), this is a trilinear form.
    Using einsum: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    For fast verification, we precompute the index mappings.
    """
    # Index arrays
    u_idx = np.array([i*m+j for i in range(n) for j in range(m) for k in range(p)])
    v_idx = np.array([j*p+k for i in range(n) for j in range(m) for k in range(p)])
    w_idx = np.array([k*n+i for i in range(n) for j in range(m) for k in range(p)])
    
    return u_idx, v_idx, w_idx

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using reshaped einsum."""
    rank = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
    return np.allclose(T, 1.0, atol=atol)

def _round_to_half(x):
    return round(x * 2) / 2.0

def _round_matrix(M):
    if isinstance(M, np.ndarray):
        return np.round(M * 2) / 2.0
    return [[_round_to_half(x) for x in row] for row in M]

def _als_reduce(n, m, p, U_init, V_init, W_init, target_rank, max_iters=300, deadline=None, seed=42):
    """
    ALS-based rank reduction with structured projection.
    """
    rng = np.random.RandomState(seed)
    init_rank = U_init.shape[1]
    
    if target_rank >= init_rank:
        U = U_init.copy().astype(float)
        V = V_init.copy().astype(float)
        W = W_init.copy().astype(float)
        if target_rank > init_rank:
            U = np.hstack([U, np.zeros((n*m, target_rank-init_rank))])
            V = np.hstack([V, np.zeros((m*p, target_rank-init_rank))])
            W = np.hstack([W, np.zeros((p*n, target_rank-init_rank))])
    else:
        # Structured random projection with QR for orthonormality
        P_U = rng.randn(init_rank, target_rank)
        P_V = rng.randn(init_rank, target_rank)
        P_W = rng.randn(init_rank, target_rank)
        for P in [P_U, P_V, P_W]:
            Q, _ = np.linalg.qr(P)
            P[:] = Q[:, :target_rank]
        U = U_init @ P_U
        V = V_init @ P_V
        W = W_init @ P_W
    
    # Precompute index structures for vectorized ALS
    # U[i*m+j] depends on: sum_k V[j*p+k] * W[k*n+i]
    # V[j*p+k] depends on: sum_i U[i*m+j] * W[k*n+i]
    # W[k*n+i] depends on: sum_j U[i*m+j] * V[j*p+k]
    
    prev_res = np.inf
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for k in range(p):
                    num += V[j*p+k] * W[k*n+i]
                    den += (V[j*p+k] * W[k*n+i])**2
                mask = den > 1e-14
                if np.any(mask):
                    U[i*m+j, mask] = num[mask] / den[mask]
        
        # Update V
        for j in range(m):
            for k in range(p):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for i in range(n):
                    num += U[i*m+j] * W[k*n+i]
                    den += (U[i*m+j] * W[k*n+i])**2
                mask = den > 1e-14
                if np.any(mask):
                    V[j*p+k, mask] = num[mask] / den[mask]
        
        # Update W
        for k in range(p):
            for i in range(n):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for j in range(m):
                    num += U[i*m+j] * V[j*p+k]
                    den += (U[i*m+j] * V[j*p+k])**2
                mask = den > 1e-14
                if np.any(mask):
                    W[k*n+i, mask] = num[mask] / den[mask]
        
        # Compute residual every 10 iterations
        if (it + 1) % 10 == 0 or it == max_iters - 1:
            Ur = U.reshape(n, m, target_rank)
            Vr = V.reshape(m, p, target_rank)
            Wr = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
            res = np.max(np.abs(T - 1.0))
            if res < prev_res * 0.999 and res < 1e-12:
                break
            prev_res = res
    
    return U, V, W

def _try_peel_columns(n, m, p, U, V, W, deadline):
    """
    Try to peel off columns one at a time by setting them to zero
    and re-optimizing the remaining columns.
    Returns (U, V, W, rank) if successful.
    """
    rank = U.shape[1]
    U = U.copy(); V = V.copy(); W = W.copy()
    
    for attempt in range(rank - 49):  # Don't go below ~49
        if deadline and time.time() > deadline:
            break
        if rank <= 49:
            break
            
        # Try each column to remove
        best_col = -1
        best_U, best_V, best_W = None, None, None
        
        for col in range(rank):
            if deadline and time.time() > deadline:
                break
            
            U_try = U.copy(); V_try = V.copy(); W_try = W.copy()
            U_try[:, col] = 0
            V_try[:, col] = 0
            W_try[:, col] = 0
            
            # Re-optimize remaining columns
            # Remove the zero column
            mask = np.ones(rank, dtype=bool)
            mask[col] = False
            U_sub = U_try[:, mask]
            V_sub = V_try[:, mask]
            W_sub = W_try[:, mask]
            
            # Quick ALS to see if it converges
            new_rank = rank - 1
            U_opt, V_opt, W_opt = _als_reduce(
                n, m, p, U_sub, V_sub, W_sub, new_rank,
                max_iters=50, deadline=deadline, seed=col
            )
            
            # Round and verify
            U_r = _round_matrix(U_opt)
            V_r = _round_matrix(V_opt)
            W_r = _round_matrix(W_opt)
            
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                best_col = col
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_col >= 0:
            U, V, W = best_U, best_V, best_W
            rank = rank - 1
        else:
            break
    
    return U, V, W, rank

def _structured_projection(n, m, p, U_init, V_init, W_init, target_rank, deadline, seed=0):
    """
    Try structured projections that respect the Kronecker block structure.
    The 49 Kronecker terms form a 7x7 grid; the 16 naive terms are separate.
    """
    rng = np.random.RandomState(seed)
    init_rank = U_init.shape[1]
    
    if target_rank >= init_rank:
        return U_init.copy(), V_init.copy(), W_init.copy()
    
    # Strategy: project within the 49-term Kronecker block and the 16-term block separately
    # Kronecker block: indices 0..48, Naive block: indices 49..64
    
    kr_rank = min(49, target_rank)
    naive_rank = max(0, target_rank - kr_rank)
    
    if kr_rank < 49:
        P_kr = rng.randn(49, kr_rank)
        Q, _ = np.linalg.qr(P_kr)
        P_kr = Q[:, :kr_rank]
    else:
        P_kr = np.eye(49)
    
    if naive_rank < 16:
        P_naive = rng.randn(16, naive_rank)
        Q, _ = np.linalg.qr(P_naive)
        P_naive = Q[:, :naive_rank]
    else:
        P_naive = np.eye(16)
    
    P = np.zeros((init_rank, target_rank))
    P[:49, :kr_rank] = P_kr
    if naive_rank > 0:
        P[49:, kr_rank:kr_rank+naive_rank] = P_naive
    
    U = U_init @ P
    V = V_init @ P
    W = W_init @ P
    
    return U, V, W

def _exact_polish(U, V, W, n, m, p, max_iters=200, deadline=None):
    """High-precision alternating least squares polish."""
    U = U.copy(); V = V.copy(); W = W.copy()
    rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        for i in range(n):
            for j in range(m):
                num = np.zeros(rank)
                den = np.zeros(rank)
                for k in range(p):
                    num += V[j*p+k] * W[k*n+i]
                    den += (V[j*p+k] * W[k*n+i])**2
                mask = den > 1e-15
                if np.any(mask):
                    U[i*m+j, mask] = num[mask] / den[mask]
        
        for j in range(m):
            for k in range(p):
                num = np.zeros(rank)
                den = np.zeros(rank)
                for i in range(n):
                    num += U[i*m+j] * W[k*n+i]
                    den += (U[i*m+j] * W[k*n+i])**2
                mask = den > 1e-15
                if np.any(mask):
                    V[j*p+k, mask] = num[mask] / den[mask]
        
        for k in range(p):
            for i in range(n):
                num = np.zeros(rank)
                den = np.zeros(rank)
                for j in range(m):
                    num += U[i*m+j] * V[j*p+k]
                    den += (U[i*m+j] * V[j*p+k])**2
                mask = den > 1e-15
                if np.any(mask):
                    W[k*n+i, mask] = num[mask] / den[mask]
        
        # Check convergence
        if it % 20 == 19:
            Ur = U.reshape(n, m, rank)
            Vr = V.reshape(m, p, rank)
            Wr = W.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
            if np.max(np.abs(T - 1.0)) < 1e-12:
                break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Build rank-65 baseline
    U65, V65, W65 = _construct_rank_65(n, m, p)
    
    # Verify baseline
    if not _verify_fast(U65, V65, W65, n, m, p):
        # Fallback: schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U65.copy(), V65.copy(), W65.copy()
    best_rank = 65
    
    # 2. Try rank reduction from 64 down
    for target_rank in range(64, 54, -1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy A: Random projection + ALS
        for seed in range(12):
            if time.time() > deadline or found:
                break
            U_opt, V_opt, W_opt = _als_reduce(
                n, m, p, U65, V65, W65, target_rank,
                max_iters=200, deadline=deadline, seed=seed
            )
            U_r = _round_matrix(U_opt)
            V_r = _round_matrix(V_opt)
            W_r = _round_matrix(W_opt)
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                found = True
                break
        
        # Strategy B: Structured projection
        if not found and time.time() < deadline:
            for seed in range(8):
                if time.time() > deadline:
                    break
                U_sp, V_sp, W_sp = _structured_projection(
                    n, m, p, U65, V65, W65, target_rank, deadline, seed
                )
                U_opt, V_opt, W_opt = _als_reduce(
                    n, m, p, U_sp, V_sp, W_sp, target_rank,
                    max_iters=150, deadline=deadline, seed=100+seed
                )
                U_r = _round_matrix(U_opt)
                V_r = _round_matrix(V_opt)
                W_r = _round_matrix(W_opt)
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        # Strategy C: Perturbed initialization
        if not found and time.time() < deadline:
            for seed in range(6):
                if time.time() > deadline:
                    break
                rng = np.random.RandomState(seed)
                U_p = U65 + rng.randn(n*m, 65) * 0.1
                V_p = V65 + rng.randn(m*p, 65) * 0.1
                W_p = W65 + rng.randn(p*n, 65) * 0.1
                U_opt, V_opt, W_opt = _als_reduce(
                    n, m, p, U_p, V_p, W_p, target_rank,
                    max_iters=150, deadline=deadline, seed=200+seed
                )
                U_r = _round_matrix(U_opt)
                V_r = _round_matrix(V_opt)
                W_r = _round_matrix(W_opt)
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
    
    # 3. Try peeling columns from the best found
    if best_rank < 65 and time.time() < deadline:
        U_peel, V_peel, W_peel, peel_rank = _try_peel_columns(
            n, m, p, best_U.astype(float), best_V.astype(float), best_W.astype(float), deadline
        )
        if peel_rank < best_rank:
            best_U, best_V, best_W = U_peel, V_peel, W_peel
            best_rank = peel_rank
    
    # 4. Final polish
    if time.time() < deadline:
        U_fin = best_U.astype(float)
        V_fin = best_V.astype(float)
        W_fin = best_W.astype(float)
        U_fin, V_fin, W_fin = _exact_polish(U_fin, V_fin, W_fin, n, m, p, max_iters=100, deadline=deadline)
        U_r = _round_matrix(U_fin)
        V_r = _round_matrix(V_fin)
        W_r = _round_matrix(W_fin)
        if _verify_fast(U_r, V_r, W_r, n, m, p):
            best_U, best_V, best_W = U_r, V_r, W_r
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
