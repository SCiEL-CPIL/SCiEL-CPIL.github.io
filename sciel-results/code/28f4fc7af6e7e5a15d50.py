import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    random.seed(42)
    np.random.seed(42)

    # Precompute index arrays for vectorized verification
    idx_i = np.repeat(np.arange(n), m*p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n*m)
    U_idx = idx_i * m + idx_j
    V_idx = idx_j * p + idx_k
    W_idx = idx_k * n + idx_i

    def verify_fast(U, V, W):
        vals = (U[U_idx] * V[V_idx] * W[W_idx]).sum(axis=1)
        return np.all(np.abs(vals - 1.0) < 1e-9)

    def residuals_fast(U, V, W):
        vals = (U[U_idx] * V[V_idx] * W[W_idx]).sum(axis=1)
        return vals - 1.0

    # ---- Strassen 2x2 * 2x2 (rank 7) ----
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[0,2]=1; W2[1,2]=1
    U2[0,3]=1; U2[3,3]=-1; V2[1,3]=1; V2[3,3]=1; W2[0,3]=1; W2[3,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[1,4]=1; W2[3,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[2,5]=1; W2[0,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[1,6]=1; W2[2,6]=1

    # ---- 4x4 * 4x4 via Kronecker Strassen (rank 49) ----
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    for t1 in range(7):
        for t2 in range(7):
            col = t1*7 + t2
            for i1 in range(2):
                if U2[i1,t1]!=0:
                    for i2 in range(2):
                        if U2[i2,t2]!=0:
                            U44[i1*2+i2, col] = U2[i1,t1]*U2[i2,t2]
            for i1 in range(2):
                if V2[i1,t1]!=0:
                    for i2 in range(2):
                        if V2[i2,t2]!=0:
                            V44[i1*2+i2, col] = V2[i1,t1]*V2[i2,t2]
            for i1 in range(2):
                if W2[i1,t1]!=0:
                    for i2 in range(2):
                        if W2[i2,t2]!=0:
                            W44[i1*2+i2, col] = W2[i1,t1]*W2[i2,t2]

    # ---- 4x4 * 4x1 (rank 4) ----
    U41 = np.zeros((16, 4))
    V41 = np.zeros((20, 4))
    W41 = np.zeros((20, 4))
    for r in range(4):
        for i in range(4):
            for j in range(4):
                if i == r:
                    U41[i*4+j, r] = 1.0
                    W41[16+i, r] = 1.0
        for j in range(4):
            V41[j*5+4, r] = 1.0

    # ---- Combine into 4x4 * 4x5 (rank 53) ----
    rank = 53
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            V[j*5+k, 0:49] = V44[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W44[k*4+i, 0:49]
    U[:, 49:53] = U41
    V[:, 49:53] = V41
    W[:, 49:53] = W41

    if not verify_fast(U, V, W):
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0; col+=1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Fast drop check ----
    def try_drop_columns_fast(U, V, W, rank):
        changed = True
        while changed:
            changed = False
            for c in range(rank-1, -1, -1):
                if time.time() > deadline:
                    return U, V, W, rank
                mask = np.ones(rank, dtype=bool); mask[c] = False
                if verify_fast(U[:, mask], V[:, mask], W[:, mask]):
                    U, V, W = U[:, mask], V[:, mask], W[:, mask]
                    rank -= 1
                    changed = True
        return U, V, W, rank

    def try_drop_two_fast(U, V, W, rank):
        if time.time() > deadline:
            return U, V, W, rank
        for c1 in range(rank):
            for c2 in range(c1+1, rank):
                if time.time() > deadline:
                    return U, V, W, rank
                mask = np.ones(rank, dtype=bool)
                mask[c1] = False; mask[c2] = False
                if verify_fast(U[:, mask], V[:, mask], W[:, mask]):
                    return U[:, mask], V[:, mask], W[:, mask], rank-2
        return U, V, W, rank

    U, V, W, rank = try_drop_columns_fast(U, V, W, rank)
    U, V, W, rank = try_drop_two_fast(U, V, W, rank)

    # ---- Guided local search ----
    entry_set = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    def local_search_guided(U, V, W, rank, iters):
        for it in range(iters):
            if time.time() > deadline:
                break
            target = it % rank
            U_t = U[:, target]; V_t = V[:, target]; W_t = W[:, target]
            contrib = U_t[U_idx] * V_t[V_idx] * W_t[W_idx]
            if np.all(np.abs(contrib) < 1e-12):
                mask = np.ones(rank, dtype=bool); mask[target] = False
                if verify_fast(U[:, mask], V[:, mask], W[:, mask]):
                    return U[:, mask], V[:, mask], W[:, mask], rank-1
                continue
            n_perturb = 1 + (it % 3)
            orig = []
            for _ in range(n_perturb):
                which = (it + _) % 3
                mat = U if which==0 else (V if which==1 else W)
                r = (it * 7 + _ * 13 + which * 31) % mat.shape[0]
                orig.append((which, r, target, mat[r, target]))
                mat[r, target] = entry_set[(it + _ + which) % len(entry_set)]
            mask = np.ones(rank, dtype=bool); mask[target] = False
            if verify_fast(U[:, mask], V[:, mask], W[:, mask]):
                return U[:, mask], V[:, mask], W[:, mask], rank-1
            for which, r, c, val in orig:
                mat = U if which==0 else (V if which==1 else W)
                mat[r, c] = val
            if it % 2 == 0:
                other_col = (target + 1) % rank
                orig2 = []
                for _ in range(3):
                    which = (it + _) % 3
                    mat = U if which==0 else (V if which==1 else W)
                    r = (it * 11 + _ * 17 + which * 23) % mat.shape[0]
                    col = target if _ < 2 else other_col
                    orig2.append((which, r, col, mat[r, col]))
                    mat[r, col] = entry_set[(it + _ + which * 2) % len(entry_set)]
                mask = np.ones(rank, dtype=bool); mask[target] = False
                if verify_fast(U[:, mask], V[:, mask], W[:, mask]):
                    return U[:, mask], V[:, mask], W[:, mask], rank-1
                for which, r, c, val in orig2:
                    mat = U if which==0 else (V if which==1 else W)
                    mat[r, c] = val
        return U, V, W, rank

    iters = 8000
    U, V, W, rank = local_search_guided(U, V, W, rank, iters)
    U, V, W, rank = try_drop_columns_fast(U, V, W, rank)
    U, V, W, rank = try_drop_two_fast(U, V, W, rank)

    if rank >= best_rank:
        U, V, W, rank = best_U, best_V, best_W, best_rank

    if not verify_fast(U, V, W):
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0; col+=1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
