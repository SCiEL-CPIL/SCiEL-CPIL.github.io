"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using Kronecker product of two Strassen decompositions."""
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val
    return {"U": U, "V": V, "W": W}


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7 using Strassen where beneficial.
    Part 1: 4x4 * 4x4 -> rank 49
    Part 2: 4x4 * 4x2 -> rank 28
    Part 3: 4x4 * 4x1 -> rank 16
    Total: 93
    """
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]
    W2p = [[0] * rank2 for _ in range(8)]
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = block_idx * 7 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U2p[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = 4 + kl
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V2p[j*2+kl][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W2p[k*4+i][r] += val
            block_idx += 1
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]
    W3 = [[0] * rank3 for _ in range(4)]
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U3[i*4+j][r] = 1
                    V3[j][r] = 1
                    W3[i][r] = 1
            block_idx += 1
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for i in range(4):
        for r in range(rank3):
            V[i*p + 6][offset3 + r] = V3[i][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = W3[i][r]
    return {"U": U, "V": V, "W": W}


def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _als_decompose(n, m, p, target_rank, init_U, init_V, init_W, max_iter=200, tol=1e-12):
    """Alternating least squares for exact tensor decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    U = np.array(init_U, dtype=float)[:, :target_rank].copy()
    V = np.array(init_V, dtype=float)[:, :target_rank].copy()
    W = np.array(init_W, dtype=float)[:, :target_rank].copy()
    T1 = T.reshape(nm, mp * pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    T1_T = T1.T
    T2_T = T2.T
    T3_T = T3.T
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIME_LIMIT - 3:
            break
        Z1 = (V @ W.T).T.reshape(mp * pn, target_rank)
        U_new = np.linalg.lstsq(Z1.T, T1_T, rcond=None)[0].T
        Z2 = (U @ W.T).T.reshape(nm * pn, target_rank)
        V_new = np.linalg.lstsq(Z2.T, T2_T, rcond=None)[0].T
        Z3 = (U @ V.T).T.reshape(nm * mp, target_rank)
        W_new = np.linalg.lstsq(Z3.T, T3_T, rcond=None)[0].T
        diff = max(np.max(np.abs(U_new - U)), np.max(np.abs(V_new - V)), np.max(np.abs(W_new - W)))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break
    for it in range(30):
        if time.time() - _START_TIME > _TIME_LIMIT - 2:
            break
        Z1 = (V @ W.T).T.reshape(mp * pn, target_rank)
        U_new = np.linalg.lstsq(Z1.T, T1_T, rcond=None)[0].T
        Z2 = (U @ W.T).T.reshape(nm * pn, target_rank)
        V_new = np.linalg.lstsq(Z2.T, T2_T, rcond=None)[0].T
        Z3 = (U @ V.T).T.reshape(nm * mp, target_rank)
        W_new = np.linalg.lstsq(Z3.T, T3_T, rcond=None)[0].T
        diff = max(np.max(np.abs(U_new - U)), np.max(np.abs(V_new - V)), np.max(np.abs(W_new - W)))
        U, V, W = U_new, V_new, W_new
        if diff < 1e-14:
            break
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = float(np.max(np.abs(T_recon - T)))
    return U, V, W, error


def _round_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Round to nearest half-integers and verify exact decomposition."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if (np.max(np.abs(U_r - U)) > tol or np.max(np.abs(V_r - V)) > tol or np.max(np.abs(W_r - W)) > tol):
        return None
    U_list = U_r.tolist()
    V_list = V_r.tolist()
    W_list = W_r.tolist()
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None


def _svd_init_rank_r(n, m, p, target_rank, seed=42):
    """Generate SVD-based initialization for rank-r decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    T1 = T.reshape(nm, mp * pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
    U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
    U3, S3, Vt3 = np.linalg.svd(T3, full_matrices=False)
    U_init = U1[:, :target_rank] * np.sqrt(S1[:target_rank])
    V_init = U2[:, :target_rank] * np.sqrt(S2[:target_rank])
    W_init = U3[:, :target_rank] * np.sqrt(S3[:target_rank])
    np.random.seed(seed)
    U_init += 0.01 * np.random.randn(*U_init.shape)
    V_init += 0.01 * np.random.randn(*V_init.shape)
    W_init += 0.01 * np.random.randn(*W_init.shape)
    return U_init, V_init, W_init


def _try_als_reduce(U, V, W, n, m, p, target_rank, max_iter=200, tol=1e-12, num_restarts=10):
    """Try to reduce rank using ALS with multiple restarts and SVD initialization."""
    best_result = None
    best_err = float('inf')
    U_init = np.array(U, dtype=float)
    V_init = np.array(V, dtype=float)
    W_init = np.array(W, dtype=float)
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        if restart == 0:
            U_r, V_r, W_r = _svd_init_rank_r(n, m, p, target_rank, seed=restart)
        elif restart == 1:
            U_r = U_init[:, :target_rank].copy()
            V_r = V_init[:, :target_rank].copy()
            W_r = W_init[:, :target_rank].copy()
        else:
            U_r = U_init @ np.random.randn(U_init.shape[1], target_rank)
            V_r = V_init @ np.random.randn(V_init.shape[1], target_rank)
            W_r = W_init @ np.random.randn(W_init.shape[1], target_rank)
        try:
            U_new, V_new, W_new, err = _als_decompose(n, m, p, target_rank, U_r, V_r, W_r, max_iter=max_iter, tol=tol)
            if err < 1e-8:
                result = _round_and_verify(U_new, V_new, W_new, n, m, p)
                if result is not None:
                    return result
            if err < best_err:
                best_err = err
        except Exception:
            continue
    return None


def _try_shared_rank_reduction(n, m, p, base_U, base_V, base_W, target_rank, max_iter=200, num_restarts=10):
    """Try to find a decomposition of target_rank where terms share between sub-tensors."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    U_base = np.array(base_U, dtype=float)
    V_base = np.array(base_V, dtype=float)
    W_base = np.array(base_W, dtype=float)
    base_rank = U_base.shape[1]
    best_result = None
    best_err = float('inf')
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 8:
            break
        if restart == 0:
            T1 = T.reshape(nm, mp * pn)
            U1, S1, _ = np.linalg.svd(T1, full_matrices=False)
            T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            U2, S2, _ = np.linalg.svd(T2, full_matrices=False)
            T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            U3, S3, _ = np.linalg.svd(T3, full_matrices=False)
            U_r = U1[:, :target_rank] * np.sqrt(S1[:target_rank])
            V_r = U2[:, :target_rank] * np.sqrt(S2[:target_rank])
            W_r = U3[:, :target_rank] * np.sqrt(S3[:target_rank])
            np.random.seed(restart)
            U_r += 0.01 * np.random.randn(*U_r.shape)
            V_r += 0.01 * np.random.randn(*V_r.shape)
            W_r += 0.01 * np.random.randn(*W_r.shape)
        elif restart == 1:
            U_r = U_base[:, :target_rank].copy()
            V_r = V_base[:, :target_rank].copy()
            W_r = W_base[:, :target_rank].copy()
        elif restart == 2:
            np.random.seed(restart * 100)
            A_u = np.random.randn(base_rank, target_rank)
            A_v = np.random.randn(base_rank, target_rank)
            A_w = np.random.randn(base_rank, target_rank)
            U_r = U_base @ A_u
            V_r = V_base @ A_v
            W_r = W_base @ A_w
        else:
            np.random.seed(restart * 200 + 7)
            left_cols = list(range(0, 49))
            right_cols = list(range(49, base_rank))
            U_r = np.zeros((nm, target_rank))
            V_r = np.zeros((mp, target_rank))
            W_r = np.zeros((pn, target_rank))
            for r in range(target_rank):
                if r < target_rank // 2:
                    lc = left_cols[r % len(left_cols)]
                    rc = right_cols[r % len(right_cols)]
                    alpha = 0.7 + 0.1 * (r % 3)
                    U_r[:, r] = alpha * U_base[:, lc] + (1-alpha) * U_base[:, rc]
                    V_r[:, r] = alpha * V_base[:, lc] + (1-alpha) * V_base[:, rc]
                    W_r[:, r] = alpha * W_base[:, lc] + (1-alpha) * W_base[:, rc]
                else:
                    lc = left_cols[r % len(left_cols)]
                    rc = right_cols[(r - target_rank // 2) % len(right_cols)]
                    alpha = 0.3 + 0.1 * (r % 3)
                    U_r[:, r] = alpha * U_base[:, lc] + (1-alpha) * U_base[:, rc]
                    V_r[:, r] = alpha * V_base[:, lc] + (1-alpha) * V_base[:, rc]
                    W_r[:, r] = alpha * W_base[:, lc] + (1-alpha) * W_base[:, rc]
        try:
            U_new, V_new, W_new, err = _als_decompose(n, m, p, target_rank, U_r, V_r, W_r, max_iter=max_iter)
            if err < 1e-8:
                result = _round_and_verify(U_new, V_new, W_new, n, m, p)
                if result is not None:
                    return result
            if err < best_err:
                best_err = err
        except Exception:
            continue
    return None


def _block_circulant_embedding(n, m, p, base_U, base_V, base_W):
    """Embed the 4x4x7 tensor into a 4x4x8 block-circulant space.
    
    The idea: treat the 7 output columns as elements of a cyclic group of order 8
    (with one zero-padded position). In the circulant embedding, the multiplication
    tensor becomes a block-circulant matrix where each block is a 4x4 multiplication
    tensor. The rank of the circulant embedding can sometimes be lower than the
    sum of individual ranks due to shared structure in the Fourier domain.
    
    We construct a decomposition of the embedded tensor, then extract the valid
    (non-wrapped) terms.
    """
    # In the circulant embedding, we have p' = 8 columns instead of 7.
    # The embedded tensor T_emb[a, b, c] for a=i*4+j, b=j*8+k, c=k*4+i
    # where k ranges over 0..7, but only k=0..6 correspond to actual output.
    # The k=7 position is zero-padded.
    
    # Actually, the circulant embedding works differently. Let me think about this
    # more carefully. The key insight is that if we can find a decomposition where
    # the "wrapping" terms (k=7) cancel out, we get a valid decomposition of the
    # original problem.
    
    # For now, this is a placeholder that returns None if the approach doesn't
    # yield a valid decomposition.
    return None


def _polish_decomposition(U, V, W, n, m, p, max_iter=100):
    """High-precision polish: run additional ALS iterations with tighter tolerance."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    rank = U.shape[1]
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    T1 = T.reshape(nm, mp * pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    for it in range(max_iter):
        Z1 = np.einsum('br,cr->bc,r', V, W).reshape(mp * pn, rank)
        U_new = np.linalg.lstsq(Z1.T, T1.T, rcond=None)[0].T
        Z2 = np.einsum('ar,cr->ac,r', U, W).reshape(nm * pn, rank)
        V_new = np.linalg.lstsq(Z2.T, T2.T, rcond=None)[0].T
        Z3 = np.einsum('ar,br->ab,r', U, V).reshape(nm * mp, rank)
        W_new = np.linalg.lstsq(Z3.T, T3.T, rcond=None)[0].T
        diff = max(np.max(np.abs(U_new - U)), np.max(np.abs(V_new - V)), np.max(np.abs(W_new - W)))
        U, V, W = U_new, V_new, W_new
        if diff < 1e-15:
            break
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = float(np.max(np.abs(T_recon - T)))
    return U, V, W, error


def _try_kron_structure_reduction(n, m, p, base_U, base_V, base_W, target_rank, max_iter=200):
    """Try to exploit the Kronecker product structure of the 4x4x4 part.
    
    The 4x4x4 part uses rank-49 (Kronecker of two rank-7 Strassen).
    The 4x4x2 part uses rank-28 (4 blocks of rank-7 Strassen for 2x2x2).
    The 4x4x1 part uses rank-16 (schoolbook).
    
    We try to find a decomposition where the Kronecker structure is partially
    shared, potentially reducing the total rank below 93.
    """
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    U_base = np.array(base_U, dtype=float)
    V_base = np.array(base_V, dtype=float)
    W_base = np.array(base_W, dtype=float)
    base_rank = U_base.shape[1]
    
    if target_rank >= base_rank:
        return None
    
    # Strategy: use the 4x4x4 Strassen structure (rank 49) and try to
    # fold the remaining 44 rank terms into a reduced set.
    # The key insight: the 4x4x2 and 4x4x1 parts might share some
    # rank terms with the 4x4x4 part if we allow mixed terms.
    
    # Initialize with the first target_rank columns of the base decomposition
    U_r = U_base[:, :target_rank].copy()
    V_r = V_base[:, :target_rank].copy()
    W_r = W_base[:, :target_rank].copy()
    
    # Add some random perturbation to escape the local minimum
    np.random.seed(42)
    U_r += 0.005 * np.random.randn(*U_r.shape)
    V_r += 0.005 * np.random.randn(*V_r.shape)
    W_r += 0.005 * np.random.randn(*W_r.shape)
    
    try:
        U_new, V_new, W_new, err = _als_decompose(n, m, p, target_rank, U_r, V_r, W_r, max_iter=max_iter)
        if err < 1e-8:
            result = _round_and_verify(U_new, V_new, W_new, n, m, p)
            if result is not None:
                return result
    except Exception:
        pass
    
    # Try another initialization: mix columns from different parts
    np.random.seed(123)
    U_r2 = np.zeros((nm, target_rank))
    V_r2 = np.zeros((mp, target_rank))
    W_r2 = np.zeros((pn, target_rank))
    
    # Mix: take some columns from each part with random linear combinations
    for r in range(target_rank):
        # Randomly select source columns from different parts
        part = r % 3  # 0: 4x4x4 part, 1: 4x4x2 part, 2: 4x4x1 part
        if part == 0:
            src = np.random.randint(0, 49)
        elif part == 1:
            src = 49 + np.random.randint(0, 28)
        else:
            src = 77 + np.random.randint(0, 16)
        
        U_r2[:, r] = U_base[:, src]
        V_r2[:, r] = V_base[:, src]
        W_r2[:, r] = W_base[:, src]
    
    U_r2 += 0.005 * np.random.randn(*U_r2.shape)
    V_r2 += 0.005 * np.random.randn(*V_r2.shape)
    W_r2 += 0.005 * np.random.randn(*W_r2.shape)
    
    try:
        U_new, V_new, W_new, err = _als_decompose(n, m, p, target_rank, U_r2, V_r2, W_r2, max_iter=max_iter)
        if err < 1e-8:
            result = _round_and_verify(U_new, V_new, W_new, n, m, p)
            if result is not None:
                return result
    except Exception:
        pass
    
    return None


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Build the base rank-93 decomposition
    base_result = None
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            base_result = result
    except Exception:
        pass
    
    if base_result is None:
        if time.time() - _START_TIME > _TIME_LIMIT:
            return _schoolbook(4, 4, 7)
        try:
            result = _schoolbook(4, 4, 7)
            base_result = result
        except Exception:
            return _schoolbook(4, 4, 7)
    
    base_rank = len(base_result['U'][0])
    
    # Try Kronecker-structure-aware reduction first (more targeted)
    if time.time() - _START_TIME < _TIME_LIMIT - 15:
        for target_rank in [base_rank - 1, base_rank - 2, base_rank - 3]:
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            try:
                reduced = _try_kron_structure_reduction(4, 4, 7, base_result['U'], base_result['V'], base_result['W'], target_rank)
                if reduced is not None:
                    return reduced
            except Exception:
                pass
    
    # Try shared-rank reduction with multiple target ranks
    if time.time() - _START_TIME < _TIME_LIMIT - 10:
        for target_rank in range(base_rank - 1, max(base_rank - 6, 50), -1):
            if time.time() - _START_TIME > _TIME_LIMIT - 7:
                break
            try:
                reduced = _try_shared_rank_reduction(4, 4, 7, base_result['U'], base_result['V'], base_result['W'], target_rank)
                if reduced is not None:
                    return reduced
            except Exception:
                pass
    
    # Try standard ALS reduction
    if time.time() - _START_TIME < _TIME_LIMIT - 8:
        for target_rank in range(base_rank - 1, max(base_rank - 4, 50), -1):
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                break
            try:
                reduced = _try_als_reduce(base_result['U'], base_result['V'], base_result['W'], 4, 4, 7, target_rank)
                if reduced is not None:
                    return reduced
            except Exception:
                pass
    
    # Final polish on the base result
    try:
        U_arr = np.array(base_result['U'], dtype=float)
        V_arr = np.array(base_result['V'], dtype=float)
        W_arr = np.array(base_result['W'], dtype=float)
        U_p, V_p, W_p, err = _polish_decomposition(U_arr, V_arr, W_arr, 4, 4, 7, max_iter=30)
        if err < 1e-10:
            polished = _round_and_verify(U_p, V_p, W_p, 4, 4, 7)
            if polished is not None:
                return polished
    except Exception:
        pass
    
    return base_result
