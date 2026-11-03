"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Combines Strassen block decomposition with SVD-guided dense search and ALS refinement."""
import time
import numpy as np
from scipy import linalg

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

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W."""
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _strassen_construct(n, m, p):
    """Construct the Strassen-based decomposition using 2x2 blocks. Rank 72."""
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
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product embedding."""
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
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _build_multiplication_operator(n, m, p):
    """Build the linear operator for matrix multiplication: maps vec(B) to vec(A*B).
    For fixed A, this is a linear map from R^{m*p} to R^{n*p}.
    But we want the tensor structure. Instead, we build the 'multiplication tensor'
    as a 3D array T[i,j,k] = 1 for all valid (i,j,k).
    
    More usefully, we can think of the problem as: find minimal rank r such that
    T = sum_r u_r ⊗ v_r ⊗ w_r where T[i,j,k] = 1.
    
    The SVD approach: reshape T into a matrix and find its rank.
    T as (n*m) x p matrix has rank p (all ones in each column block).
    T as n x (m*p) matrix has rank n.
    T as (n*p) x m matrix has rank m.
    
    The tensor rank is what we seek. We use the SVD to get good initializations.
    """
    T = np.ones((n, m, p))
    
    # Matrix unfoldings for SVD
    T_123 = T.reshape(n*m, p)  # (n*m) x p
    T_132 = T.transpose(0, 2, 1).reshape(n*p, m)  # (n*p) x m
    T_213 = T.transpose(1, 0, 2).reshape(m*n, p)  # (m*n) x p
    
    return T, T_123, T_132, T_213

def _svd_based_initialization(n, m, p, rank, seed=42):
    """Use SVD of matrix unfoldings to create good initializations for ALS."""
    np.random.seed(seed)
    
    T, T_123, T_132, T_213 = _build_multiplication_operator(n, m, p)
    
    # SVD of each unfolding
    U_123, S_123, Vt_123 = np.linalg.svd(T_123, full_matrices=False)
    U_132, S_132, Vt_132 = np.linalg.svd(T_132, full_matrices=False)
    U_213, S_213, Vt_213 = np.linalg.svd(T_213, full_matrices=False)
    
    # Truncate to desired rank
    r = min(rank, T_123.shape[1], T_132.shape[1], T_213.shape[1])
    
    # Initialize from SVD components
    U_init = U_123[:, :r].copy()
    V_init = U_132[:, :r].copy()
    W_init = U_213[:, :r].copy()
    
    # Scale to balance the factors
    if r > 0:
        S_123_r = S_123[:r]
        S_132_r = S_132[:r]
        S_213_r = S_213[:r]
        
        # Normalize so that product of singular values is balanced
        prod_S = np.prod(np.array([S_123_r, S_132_r, S_213_r]).T, axis=0)
        if np.any(prod_S > 0):
            cube_root = np.cbrt(prod_S)
            for i in range(r):
                if cube_root[i] > 1e-10:
                    U_init[:, i] /= cube_root[i]
                    V_init[:, i] /= cube_root[i]
                    W_init[:, i] /= cube_root[i]
    
    # Add small random perturbation to break symmetry
    U_init += np.random.randn(n*m, r) * 0.01
    V_init += np.random.randn(m*p, r) * 0.01
    W_init += np.random.randn(p*n, r) * 0.01
    
    return U_init, V_init, W_init

def _polar_decomposition_approach(n, m, p, rank, seed=42):
    """Use polar decomposition of the multiplication operator to find rank-1 bilinear forms.
    
    The idea: vectorize the matrix multiplication as a linear map.
    For each output position (i,k), the bilinear form is sum_j A[i,j]*B[j,k].
    We can think of this as: for each (i,k), we need a rank-1 approximation of the
    4x4 matrix E_{i,k} (with 1 at position (i,k) in the output).
    
    Actually, let's think differently. The multiplication tensor T[i,j,k]=1 can be
    decomposed as sum_r u_r[i,j] * v_r[j,k] * w_r[k,i].
    
    The polar approach: consider the map M: R^{16} -> R^{20} defined by
    M(vec(A)) = vec(A * B) for fixed B, or more precisely, the bilinear map.
    
    Let's use the fact that T = 1_{4x4x5} can be written as sum of rank-1 tensors.
    The minimal rank is what we seek. We use the SVD of the "slice" matrices to
    guide the decomposition.
    """
    np.random.seed(seed)
    
    # Build the tensor and its slices
    T = np.ones((n, m, p))
    
    # For each fixed j, T[:,j,:] is an n x p matrix of ones
    # For each fixed i, T[i,:,:] is an m x p matrix of ones  
    # For each fixed k, T[:,:,k] is an n x m matrix of ones
    
    # The key insight: T[i,j,k] = 1 = (1/sqrt(m)) * sum over some basis
    # We can write 1_{n x m x p} = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    
    # Use the fact that the all-ones matrix of size a x b has rank 1.
    # So T[:,j,:] has rank 1 for each j, and T[i,:,:] has rank 1 for each i.
    
    # A simple decomposition: T[i,j,k] = (1/sqrt(m)) * sum_{j'} e_i e_{j'}^T ⊗ e_{j'} e_k^T ⊗ ...
    # This gives rank m*p = 20, but we can do better.
    
    # Better: use the structure. T[i,j,k] = 1 for all i,j,k.
    # We can write: T[i,j,k] = sum_{r=1}^R a_r[i] * b_r[j] * c_r[k] * d_r[i,k]
    # But we need the specific form U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
    
    # Let's try a direct construction using the SVD of the "flattened" tensor.
    T_flat = T.reshape(n*m, p)  # 16 x 5, all ones
    U_svd, S_svd, Vt_svd = np.linalg.svd(T_flat, full_matrices=False)
    
    # The SVD of all-ones 16x5 matrix: rank 1, with U = (1/sqrt(16)) * 1_16, 
    # S = sqrt(80), Vt = (1/sqrt(5)) * 1_5^T
    
    # For a rank-r decomposition, we need to split this rank-1 structure
    # across the three modes.
    
    # Initialize with a structured guess
    U_init = np.zeros((n*m, rank))
    V_init = np.zeros((m*p, rank))
    W_init = np.zeros((p*n, rank))
    
    # Use a "cyclic" pattern inspired by the tensor structure
    for r in range(rank):
        # Distribute the "ones" across the three factors
        phase = 2 * np.pi * r / rank
        
        for i in range(n):
            for j in range(m):
                U_init[i*m+j, r] = np.cos(phase * (i + j)) / np.sqrt(m)
        
        for j in range(m):
            for k in range(p):
                V_init[j*p+k, r] = np.cos(phase * (j + k)) / np.sqrt(p)
        
        for k in range(p):
            for i in range(n):
                W_init[k*n+i, r] = np.cos(phase * (k + i)) / np.sqrt(n)
    
    # Normalize to ensure the product sums to 1
    # Compute the current tensor and scale
    T_current = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U_init.reshape(n, m, rank)[:, :, r]
        Vr = V_init.reshape(m, p, rank)[:, :, r]
        Wr = W_init.reshape(p, n, rank)[:, :, r]
        for j in range(m):
            T_current[:, j, :] += np.outer(Ur[:, j], Vr[j, :]) * Wr.T
    
    # Scale W to match
    mean_val = np.mean(T_current)
    if abs(mean_val) > 1e-10:
        W_init *= 1.0 / mean_val
    
    return U_init, V_init, W_init

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=100):
    """Try to find a rank-r decomposition using alternating least squares with rounding."""
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
    
    def _compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        return np.sqrt(err)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        err = _compute_error(U, V, W)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    # Polish with tighter convergence
    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        
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
        
        err = _compute_error(U, V, W)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-14:
            break
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    
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
    
    for target_rank in range(rank - 1, max(40, rank - 25), -1):
        if time.time() > deadline:
            break
        
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
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try both baseline constructions
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    U_2x2, V_2x2, W_2x2 = _strassen_construct(n, m, p)
    
    kron_ok = _verify_fast(U_kron, V_kron, W_kron, n, m, p, atol=1e-9)
    twox2_ok = _verify_fast(U_2x2, V_2x2, W_2x2, n, m, p, atol=1e-9)
    
    if not kron_ok and not twox2_ok:
        return _schoolbook()
    
    if kron_ok and (not twox2_ok or len(U_kron[0]) <= len(U_2x2[0])):
        U_base, V_base, W_base = U_kron, V_kron, W_kron
    else:
        U_base, V_base, W_base = U_2x2, V_2x2, W_2x2
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try to reduce rank from the baseline
    if time.time() < deadline - 50:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 50)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try SVD-based and polar decomposition initializations at various ranks
    for rank in range(best_rank - 1, max(45, best_rank - 30), -1):
        if time.time() > deadline - 25:
            break
        
        found = False
        
        # SVD-based initialization
        for seed in range(8):
            if time.time() > deadline - 25:
                break
            U_init, V_init, W_init = _svd_based_initialization(n, m, p, rank, seed=seed)
            result = _als_decompose(n, m, p, rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=300, deadline=deadline - 25, polish_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
        
        # Polar decomposition initialization
        for seed in range(8):
            if time.time() > deadline - 25:
                break
            U_init, V_init, W_init = _polar_decomposition_approach(n, m, p, rank, seed=seed)
            result = _als_decompose(n, m, p, rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=300, deadline=deadline - 25, polish_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
        
        # Random initialization as fallback
        for seed in range(10):
            if time.time() > deadline - 25:
                break
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=300, 
                                    deadline=deadline - 25, polish_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
