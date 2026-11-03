"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses block-based Strassen decomposition as baseline (rank 65 via 4x4x4+naive, or 72 via 2x2 blocks),
then attempts refinement via ALS with rounding and alternative block structures.
Falls back to schoolbook (rank 80) if all else fails."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64, checking tolerance."""
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
    """Compute the L2 error of the decomposition efficiently."""
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
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W.
    U indices: (i,j) -> i*2+j (4 entries)
    V indices: (j,k) -> j*2+k (4 entries)
    W indices: (k,i) -> k*2+i (4 entries)
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _strassen_construct(n, m, p):
    """Construct the Strassen-based decomposition using 2x2 blocks.
    For 4x4 * 4x5:
    - Split into 2x2 blocks
    - Use Strassen (rank 7) for 2x2x2 multiplications
    - Use Strassen + naive (rank 11) for 2x2x3 multiplications
    
    Total: 4 * 7 + 4 * 11 = 28 + 44 = 72
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    strassen_terms = _strassen_2x2x2_terms()

    def add_strassen_2x2_2(ai, aj, bj, bk):
        """Add 7 Strassen terms for A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+2] -> C[ai:ai+2, bk:bk+2]"""
        for u_sparse, v_sparse, w_sparse in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
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
        """Add 11 terms: Strassen for first 2 cols + naive for 3rd col.
        A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+3] -> C[ai:ai+2, bk:bk+3]
        """
        add_strassen_2x2_2(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)

    # C00 (2x2) = A00*B00 + A01*B10
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    
    # C10 (2x2) = A10*B00 + A11*B10
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    
    # C01 (2x3) = A00*B01 + A01*B11
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    
    # C11 (2x3) = A10*B01 + A11*B11
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product embedding.
    
    Splits 4x4x5 into 4x4x4 (handled by Kronecker-embedded Strassen, rank 49)
    plus 4x4x1 (handled naively, rank 16).
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # For 4x4x4: use Kronecker product of two 2x2x2 Strassen decompositions
    # The 4x4x4 tensor = sum over (i1,i2,j1,j2,k1,k2) A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*2+k2] * C[k1*2+k2, i1*2+i2]
    # Using Kronecker: U = U1 ⊗ U2, V = V1 ⊗ V2, W = W1 ⊗ W2
    # where (U1,V1,W1) and (U2,V2,W2) are 2x2x2 Strassen decompositions
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # Kronecker product for U: index (i1*2+i2, j1*2+j2) -> (i1*2+i2)*m + (j1*2+j2)
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * m + j] = val1 * val2
            
            # Kronecker product for V: index (j1*2+j2, k1*2+k2) -> (j1*2+j2)*p + (k1*2+k2)
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * p + k] = val1 * val2
            
            # Kronecker product for W: index (k1*2+k2, i1*2+i2) -> (k1*2+k2)*n + (i1*2+i2)
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] = val1 * val2
            
            add_term(u, v, w)
    
    # Add naive terms for the 5th column (k=4): A[*,j] * B[j,4] -> C[4, i]
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i * m + j] = 1
            v = np.zeros(m*p); v[j * p + 4] = 1
            w = np.zeros(p*n); w[4 * n + i] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=100):
    """Try to find a rank-r decomposition using alternating least squares with rounding.
    
    Returns (U, V, W) as lists if found, else None.
    """
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
    
    # Precompute index arrays for vectorized operations
    U_idx = np.arange(n*m).reshape(n, m)
    V_idx = np.arange(m*p).reshape(m, p)
    W_idx = np.arange(p*n).reshape(p, n)
    
    def _als_iteration():
        """One full ALS iteration: optimize U, then V, then W."""
        # Optimize U: for each (i,j), solve A @ U[i*m+j,:] = ones(p)
        # A[k,:] = V[j*p+k,:] * W[k*n+i,:] (element-wise product), k=0..p-1
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V: for each (j,k), solve A @ V[j*p+k,:] = ones(n)
        # A[i,:] = U[i*m+j,:] * W[k*n+i,:] (element-wise product), i=0..n-1
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W: for each (k,i), solve A @ W[k*n+i,:] = ones(m)
        # A[j,:] = U[i*m+j,:] * V[j*p+k,:] (element-wise product), j=0..m-1
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
        
        # Compute error efficiently
        err = _compute_error(U, V, W, n, m, p, rank)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    # Polish: run additional iterations with tighter convergence
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
    
    # Try multiple rounding strategies
    candidates = []
    
    # Strategy 1: Round to nearest half-integer
    U_r1 = _round_matrix_to_half_int(best_U)
    V_r1 = _round_matrix_to_half_int(best_V)
    W_r1 = _round_matrix_to_half_int(best_W)
    candidates.append((U_r1, V_r1, W_r1))
    
    # Strategy 2: Round to nearest integer
    U_r2 = np.round(best_U)
    V_r2 = np.round(best_V)
    W_r2 = np.round(best_W)
    candidates.append((U_r2, V_r2, W_r2))
    
    # Strategy 3: Try scaling - round 2x values to integers, then divide by 2
    # This is equivalent to half-integer rounding but try different base points
    for offset in [0, 0.25, -0.25]:
        U_r3 = _round_matrix_to_half_int(best_U + offset)
        V_r3 = _round_matrix_to_half_int(best_V + offset)
        W_r3 = _round_matrix_to_half_int(best_W + offset)
        candidates.append((U_r3, V_r3, W_r3))
    
    # Verify each candidate
    for Uc, Vc, Wc in candidates:
        if _verify_fast(Uc.tolist(), Vc.tolist(), Wc.tolist(), n, m, p, atol=1e-9):
            return Uc.tolist(), Vc.tolist(), Wc.tolist()
    
    # If rounding fails, try a final polish with the rounded values as starting point
    # and see if we can get closer to exact
    U_f = U_r1.copy()
    V_f = V_r1.copy()
    W_f = W_r1.copy()
    
    for it in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        
        # Check if we're close to half-integers
        U_close = np.allclose(U, np.round(U*2)/2.0, atol=1e-6)
        V_close = np.allclose(V, np.round(V*2)/2.0, atol=1e-6)
        W_close = np.allclose(W, np.round(W*2)/2.0, atol=1e-6)
        
        if U_close and V_close and W_close:
            U_r = _round_matrix_to_half_int(U)
            V_r = _round_matrix_to_half_int(V)
            W_r = _round_matrix_to_half_int(W)
            if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    """Try to reduce the rank starting from a baseline decomposition."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    # Try reducing rank by dropping columns and re-optimizing
    for target_rank in range(rank - 1, max(40, rank - 25), -1):
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        for init_seed in range(4):
            if time.time() > deadline:
                break
            
            if init_seed == 0:
                U_init = Ua[:, :target_rank]
                V_init = Va[:, :target_rank]
                W_init = Wa[:, :target_rank]
            elif init_seed == 1:
                U_init = Ua[:, rank-target_rank:]
                V_init = Va[:, rank-target_rank:]
                W_init = Wa[:, rank-target_rank:]
            elif init_seed == 2:
                np.random.seed(init_seed * 100 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            else:
                # Random perturbation of a subset
                np.random.seed(999 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx] + np.random.randn(n*m, target_rank) * 0.1
                V_init = Va[:, idx] + np.random.randn(m*p, target_rank) * 0.1
                W_init = Wa[:, idx] + np.random.randn(p*n, target_rank) * 0.1
            
            result = _als_decompose(n, m, p, target_rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=200, deadline=deadline, polish_iters=50)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result

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
    deadline = start_time + 145  # Safety margin: evaluator kills at 150s
    
    n, m, p = DIMENSIONS
    
    # Try both baseline constructions and use the better one
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    U_2x2, V_2x2, W_2x2 = _strassen_construct(n, m, p)
    
    # Verify both
    kron_ok = _verify_fast(U_kron, V_kron, W_kron, n, m, p, atol=1e-9)
    twox2_ok = _verify_fast(U_2x2, V_2x2, W_2x2, n, m, p, atol=1e-9)
    
    if not kron_ok and not twox2_ok:
        return _schoolbook()
    
    # Use the lower-rank valid baseline
    if kron_ok and (not twox2_ok or len(U_kron[0]) <= len(U_2x2[0])):
        U_base, V_base, W_base = U_kron, V_kron, W_kron
    else:
        U_base, V_base, W_base = U_2x2, V_2x2, W_2x2
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try to reduce rank from the baseline - give it more time
    if time.time() < deadline - 50:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 50)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try random initializations at various ranks
    # Start from best_rank-1 and go down
    for rank in range(best_rank - 1, max(45, best_rank - 30), -1):
        if time.time() > deadline - 25:
            break
        
        found = False
        # More seeds for better coverage
        for seed in range(20):
            if time.time() > deadline - 25:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=500, 
                                    deadline=deadline - 25, polish_iters=100)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            # If we found a decomposition at this rank, try to go lower
            continue
        else:
            # If we couldn't find one at this rank, stop trying lower ranks
            # (assuming rank is monotonically harder)
            break
    
    return best_result
# EVOLVE-BLOCK-END
