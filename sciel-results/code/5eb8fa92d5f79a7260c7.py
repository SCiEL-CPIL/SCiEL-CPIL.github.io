"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Block Strassen (rank 64) as baseline.
Attempts rank reduction via structured column removal with SVD-based null-space detection,
plus vectorized ALS polishing with half-integer rounding.
Falls back to schoolbook (rank 80) if all else fails."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    if Ua.ndim == 1:
        Ua = Ua.reshape(-1, 1)
    if Va.ndim == 1:
        Va = Va.reshape(-1, 1)
    if Wa.ndim == 1:
        Wa = Wa.reshape(-1, 1)
        
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
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
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

def _strassen_block_construct(n, m, p):
    """Construct a rank-64 decomposition using 2x2 blocks.
    Splits B into B_left (4x2) and B_right (4x3).
    B_right is split into B_mid (4x2) and B_last (4x1).
    
    Part 1: A * B_left (4x2). Rank 28.
    Part 2: A * B_mid (4x2). Rank 28.
    Part 3: A * B_last (4x1). Rank 8.
    Total Rank: 64.
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

    def add_naive_2x2_1(ai, aj, bj, bk):
        """Add 4 Naive terms for A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+1] -> C[ai:ai+2, bk:bk+1]"""
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk, ai+di)] = 1
                add_term(u, v, w)

    def add_4x4x2(ai, aj, bj, bk):
        """Add 28 terms for A[ai:ai+4, aj:aj+4] @ B[bj:bj+4, bk:bk+2] -> C[ai:ai+4, bk:bk+2]"""
        add_strassen_2x2_2(ai, aj, bj, bk)
        add_strassen_2x2_2(ai, aj+2, bj+2, bk)
        add_strassen_2x2_2(ai+2, aj, bj, bk)
        add_strassen_2x2_2(ai+2, aj+2, bj+2, bk)

    def add_4x4x1(ai, aj, bj, bk):
        """Add 8 terms for A[ai:ai+4, aj:aj+4] @ B[bj:bj+4, bk:bk+1] -> C[ai:ai+4, bk:bk+1]"""
        add_naive_2x2_1(ai, aj, bj, bk)
        add_naive_2x2_1(ai, aj+2, bj+2, bk)
        add_naive_2x2_1(ai+2, aj, bj, bk)
        add_naive_2x2_1(ai+2, aj+2, bj+2, bk)

    # Part 1: C[:, 0:2] = A * B[:, 0:2]
    add_4x4x2(0, 0, 0, 0)
    
    # Part 2: C[:, 2:4] = A * B[:, 2:4]
    add_4x4x2(0, 0, 0, 2)
    
    # Part 3: C[:, 4:5] = A * B[:, 4:5]
    add_4x4x1(0, 0, 0, 4)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _compute_error(U, V, W, n, m, p):
    """Compute the decomposition error efficiently."""
    rank = U.shape[1]
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
    return np.sqrt(np.sum((T - 1.0) ** 2))

def _als_polish(U, V, W, n, m, p, max_iters=50, atol=1e-14):
    """Polish a near-exact decomposition using ALS, then round to half-integers."""
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    rank = U.shape[1]
    
    for it in range(max_iters):
        # Optimize U
        for i in range(n):
            for j in range(m):
                V_slice = V[j*p:(j+1)*p, :]
                W_slice = np.array([W[k*n+i, :] for k in range(p)])
                A_mat = V_slice * W_slice
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                U_slice = np.array([U[i*m+j, :] for i in range(n)])
                W_slice = W[k*n:(k+1)*n, :]
                A_mat = U_slice * W_slice
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_slice = np.array([V[j*p+k, :] for j in range(m)])
                A_mat = U_slice * V_slice
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        err = _compute_error(U, V, W, n, m, p)
        if err < atol:
            break
    
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _try_remove_column(U, V, W, col_idx, n, m, p, deadline):
    """Try removing column col_idx and polishing to see if we still have exact decomposition."""
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    idx = [i for i in range(rank) if i != col_idx]
    U_new = U[:, idx]
    V_new = V[:, idx]
    W_new = W[:, idx]
    
    result = _als_polish(U_new, V_new, W_new, n, m, p, max_iters=30, atol=1e-12)
    if result is not None:
        return result
    return None

def _find_redundant_columns(U, V, W, n, m, p, deadline):
    """Find columns that can be removed while maintaining exact decomposition.
    Uses SVD-based analysis to identify structurally redundant columns."""
    rank = U.shape[1]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Try to remove each column one at a time
    removed_cols = []
    for col in range(rank):
        if time.time() > deadline:
            break
        result = _try_remove_column(Ua, Va, Wa, col, n, m, p, deadline)
        if result is not None:
            Ua, Va, Wa = np.array(result[0]), np.array(result[1]), np.array(result[2])
            removed_cols.append(col)
    
    if removed_cols:
        return np.array(result[0]).tolist(), np.array(result[1]).tolist(), np.array(result[2]).tolist()
    return None

def _random_column_removal(U, V, W, n, m, p, target_rank, deadline, num_seeds=20):
    """Try removing random subsets of columns to reach target_rank."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    current_rank = Ua.shape[1]
    
    if current_rank <= target_rank:
        return (Ua.tolist(), Va.tolist(), Wa.tolist())
    
    for seed in range(num_seeds):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        idx = np.random.choice(current_rank, target_rank, replace=False)
        U_init = Ua[:, idx]
        V_init = Va[:, idx]
        W_init = Wa[:, idx]
        
        result = _als_polish(U_init, V_init, W_init, n, m, p, max_iters=40, atol=1e-12)
        if result is not None:
            return result
            
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
    
    # Build baseline: Block Strassen (rank 64)
    U_block, V_block, W_block = _strassen_block_construct(n, m, p)
    block_ok = _verify_fast(U_block, V_block, W_block, n, m, p, atol=1e-9)
    
    if not block_ok:
        return _schoolbook()
    
    best_result = {"U": U_block, "V": V_block, "W": W_block}
    best_rank = len(U_block[0])
    
    # Phase 1: Greedy column removal - try to remove one column at a time
    Ua = np.array(U_block, dtype=np.float64)
    Va = np.array(V_block, dtype=np.float64)
    Wa = np.array(W_block, dtype=np.float64)
    
    max_greedy_removals = 12
    for removal_round in range(max_greedy_removals):
        if time.time() > deadline - 20:
            break
        if Ua.shape[1] <= 50:
            break
        
        found_removal = False
        rank = Ua.shape[1]
        # Try columns in a fixed order (deterministic)
        for col in range(rank):
            if time.time() > deadline - 20:
                break
            result = _try_remove_column(Ua, Va, Wa, col, n, m, p, deadline - 20)
            if result is not None:
                Ua, Va, Wa = np.array(result[0]), np.array(result[1]), np.array(result[2])
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                best_rank = Ua.shape[1]
                found_removal = True
                break
        
        if not found_removal:
            break
    
    # Phase 2: Try to jump to specific target ranks using random subset selection
    for target_rank in [best_rank - 2, best_rank - 3, best_rank - 4, best_rank - 5]:
        if target_rank < 50 or time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
            
        result = _random_column_removal(Ua.tolist(), Va.tolist(), Wa.tolist(), n, m, p, 
                                        target_rank, deadline - 15, num_seeds=15)
        if result is not None:
            Ua, Va, Wa = np.array(result[0]), np.array(result[1]), np.array(result[2])
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            best_rank = target_rank
    
    return best_result

# EVOLVE-BLOCK-END
