import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def _target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _snap_half_int(arr):
    return np.round(arr * 2) / 2.0

def _recon(U, V, W):
    # sum_r U[:,r] V[:,r] W[:,r]
    return np.einsum('ar,br,cr->abc', U, V, W)

def _verify(U, V, W, target, atol=1e-8):
    return np.allclose(_recon(U, V, W), target, atol=atol)

def _als_block(U, V, W, T, max_iters, deadline, snap_every=4):
    """Alternating least squares on a small block tensor T (a,b,c)."""
    a, b, c = T.shape
    R = U.shape[1]
    U = U.copy(); V = V.copy(); W = W.copy()
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Update U: for each i, solve M_VW @ U[i] = T[i]
        M_VW = np.einsum('br,cr->bcr', V, W).reshape(b*c, R)
        for i in range(a):
            if time.time() > deadline:
                break
            try:
                U[i, :] = np.linalg.lstsq(M_VW, T[i].flatten(), rcond=None)[0]
            except Exception:
                pass
        # Update V
        M_UW = np.einsum('ar,cr->acr', U, W).reshape(a*c, R)
        for j in range(b):
            if time.time() > deadline:
                break
            tgt = np.zeros(a*c)
            for i in range(a):
                for k in range(c):
                    if T[i, j, k] != 0:
                        tgt[i*c + k] = T[i, j, k]
            try:
                V[j, :] = np.linalg.lstsq(M_UW, tgt, rcond=None)[0]
            except Exception:
                pass
        # Update W
        M_UV = np.einsum('ar,br->abr', U, V).reshape(a*b, R)
        for k in range(c):
            if time.time() > deadline:
                break
            tgt = np.zeros(a*b)
            for i in range(a):
                for j in range(b):
                    if T[i, j, k] != 0:
                        tgt[i*b + j] = T[i, j, k]
            try:
                W[k, :] = np.linalg.lstsq(M_UV, tgt, rcond=None)[0]
            except Exception:
                pass
        if snap_every and (it + 1) % snap_every == 0:
            U = _snap_half_int(U)
            V = _snap_half_int(V)
            W = _snap_half_int(W)
    return U, V, W

def _block_tensor_3x2x2(n, mb, pb):
    """Build the 3 x mb x pb sub-tensor for block (mb, pb) of a 3 x m x p multiplication.
    Here the block multiplies a 3 x mb matrix by a mb x pb matrix -> 3 x pb.
    We treat it as a standalone tensor T[i, j, k] = 1 for the 'diagonal' pattern.
    Actually for a sub-block, the multiplication is: C[i, k] = sum_j A[i, j] B[j, k].
    The tensor for this block is T[i, j, k] = 1 for all (i, j, k) in the block? No.
    The tensor T[i, j, k] = 1 means the term A[i,j]*B[j,k] contributes to C[i,k].
    So for a full block, T[i, j, k] = 1 for all i in [0,n), j in [0,mb), k in [0,pb).
    """
    T = np.zeros((n, mb, pb))
    for i in range(n):
        for j in range(mb):
            for k in range(pb):
                T[i, j, k] = 1.0
    return T

def _schoolbook_block(n, mb, pb):
    rank = n * mb * pb
    U = np.zeros((n*mb, rank))
    V = np.zeros((mb*pb, rank))
    W = np.zeros((pb*n, rank))
    col = 0
    for i in range(n):
        for j in range(mb):
            for k in range(pb):
                U[i*mb + j, col] = 1
                V[j*pb + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _eliminate_rank(U, V, W, target, deadline):
    """Remove linearly-dependent columns (rank elimination)."""
    R = U.shape[1]
    keep = []
    # Build a combined matrix to check column dependence via the tensor.
    # Simpler: try removing each column and re-checking exactness.
    for r in range(R):
        if time.time() > deadline:
            break
        # skip column r
        idx = [x for x in range(R) if x != r]
        if not idx:
            continue
        U2 = U[:, idx]
        V2 = V[:, idx]
        W2 = W[:, idx]
        if _verify(U2, V2, W2, target):
            U, V, W = U2, V2, W2
            R = U.shape[1]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    target = _target_tensor(n, m, p)

    # Schoolbook global
    rank_schoolbook = n * m * p
    U_init = np.zeros((n*m, rank_schoolbook))
    V_init = np.zeros((m*p, rank_schoolbook))
    W_init = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i*m+j, col] = 1
                V_init[j*p+k, col] = 1
                W_init[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    best_rank = rank_schoolbook

    random.seed(42)
    np.random.seed(42)

    # Block decomposition: split m into 2 blocks of 2, p into 4 blocks of 2.
    # Each block is 3 x 2 x 2. Solve each independently, then concatenate.
    mb_list = [2, 2]
    pb_list = [2, 2, 2, 2]

    # For each block, try to find a low-rank exact decomposition.
    # 3x2x2 schoolbook rank = 12. Try to reduce.
    block_solutions = []
    for bi, (mb, pb) in enumerate(zip(mb_list, pb_list)):
        if time.time() > deadline - 10:
            block_solutions.append(_schoolbook_block(n, mb, pb))
            continue
        T_block = _block_tensor_3x2x2(n, mb, pb)
        U_b, V_b, W_b = _schoolbook_block(n, mb, pb)
        best_b_rank = n * mb * pb
        best_b = (U_b.copy(), V_b.copy(), W_b.copy())

        # Try ranks from 11 down to 4
        for tr in range(best_b_rank - 1, 3, -1):
            if time.time() > deadline - 10:
                break
            found = False
            for restart in range(6):
                if time.time() > deadline - 10:
                    break
                U_r = np.random.randn(n*mb, tr) * 0.5
                V_r = np.random.randn(mb*pb, tr) * 0.5
                W_r = np.random.randn(pb*n, tr) * 0.5
                U_r, V_r, W_r = _als_block(U_r, V_r, W_r, T_block, max_iters=25, deadline=deadline - 5)
                U_s, V_s, W_s = _snap_half_int(U_r), _snap_half_int(V_r), _snap_half_int(W_r)
                if _verify(U_s, V_s, W_s, T_block):
                    best_b = (U_s, V_s, W_s)
                    best_b_rank = tr
                    found = True
                    break
                if _verify(U_r, V_r, W_r, T_block):
                    best_b = (U_r, V_r, W_r)
                    best_b_rank = tr
                    found = True
                    break
            if found:
                break

        # Final polish: tight tolerance
        if time.time() < deadline - 5:
            U_b, V_b, W_b = _als_block(best_b[0], best_b[1], best_b[2], T_block, max_iters=15, deadline=deadline - 3, snap_every=0)
            U_s, V_s, W_s = _snap_half_int(best_b[0]), _snap_half_int(best_b[1]), _snap_half_int(best_b[2])
            if _verify(U_s, V_s, W_s, T_block):
                best_b = (U_s, V_s, W_s)
            # rank elimination
            best_b = _eliminate_rank(best_b[0], best_b[1], best_b[2], T_block, deadline - 2)
        block_solutions.append(best_b)

    # Concatenate block solutions
    U_cat = np.hstack([b[0] for b in block_solutions])
    V_cat = np.hstack([b[1] for b in block_solutions])
    W_cat = np.hstack([b[2] for b in block_solutions])

    # But concatenation only works if blocks are independent in the global tensor.
    # They are independent because each block occupies disjoint (i,j,k) regions.
    # Verify the concatenated result against the full target.
    if _verify(U_cat, V_cat, W_cat, target):
        if U_cat.shape[1] < best_rank:
            best_rank = U_cat.shape[1]
            best_U, best_V, best_W = U_cat, V_cat, W_cat
    else:
        # Fallback: global ALS search (keep current program's approach as backup)
        for target_rank in range(90, 47, -2):
            if time.time() > deadline - 15:
                break
            for restart in range(10):
                if time.time() > deadline - 15:
                    break
                U = np.random.randn(n*m, target_rank) * 0.5
                V = np.random.randn(m*p, target_rank) * 0.5
                W = np.random.randn(p*n, target_rank) * 0.5
                U, V, W = _als_block(U, V, W, target, max_iters=30, deadline=deadline - 10, snap_every=5)
                U_s, V_s, W_s = _snap_half_int(U), _snap_half_int(V), _snap_half_int(W)
                if _verify(U_s, V_s, W_s, target):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_s, V_s, W_s
                        break
                elif _verify(U, V, W, target):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                        break

    # Final global rank elimination
    if time.time() < deadline - 3:
        best_U, best_V, best_W = _eliminate_rank(best_U, best_V, best_W, target, deadline - 1)
        best_rank = best_U.shape[1]

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    return {"U": U_list, "V": V_list, "W": W_list}
