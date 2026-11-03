import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0
    np.random.seed(20240517)

    def verify(U, V, W):
        # U:(nm,r) V:(mp,r) W:(pn,r); returns exactness bool
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        if r == 0:
            return False
        T = np.tensordot(U, V, axes=(1, 1))          # (nm, mp, r)
        TW = T * W.T                                  # (nm, mp, r)
        S = TW.sum(axis=2)                            # (nm, mp)
        idx = np.arange(n * m)
        ok = np.all(np.abs(S[idx, idx] - 1.0) < 1e-9)
        off = np.ones(n * m, dtype=bool)
        off[idx] = False
        ok = ok and np.all(np.abs(S[off]) < 1e-9)
        return bool(ok)

    def residual(U, V, W):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        if r == 0:
            return float(n * m * p)
        T = np.tensordot(U, V, axes=(1, 1))
        TW = T * W.T
        S = TW.sum(axis=2)
        idx = np.arange(n * m)
        res = float(np.sum((S - 1.0) ** 2))
        off = np.ones(n * m, dtype=bool)
        off[idx] = False
        res += float(np.sum(S[off] ** 2))
        return res

    # ---- Strassen 2x2 * 2x2 (rank 7) ----
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[0,2]=1; W2[1,2]=1
    U2[0,3]=1; U2[3,3]=-1; V2[1,3]=1; V2[3,3]=1; W2[0,3]=1; W2[3,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[1,4]=1; W2[3,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[2,5]=1; W2[0,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[1,6]=1; W2[2,6]=1

    # ---- 4x4 * 4x4 via Kronecker Strassen (rank 49) ----
    U44 = np.zeros((16, 49)); V44 = np.zeros((16, 49)); W44 = np.zeros((16, 49))
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
    U41 = np.zeros((16, 4)); V41 = np.zeros((20, 4)); W41 = np.zeros((20, 4))
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
    U = np.zeros((16, rank)); V = np.zeros((20, rank)); W = np.zeros((20, rank))
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

    if not verify(U, V, W):
        # fallback: schoolbook (always valid)
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0; col+=1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_res = 0.0

    def drop_pass(U, V, W, rank):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for c in range(rank-1, -1, -1):
                if time.time() > deadline:
                    return U, V, W, rank
                mask = np.ones(rank, dtype=bool); mask[c] = False
                if verify(U[:, mask], V[:, mask], W[:, mask]):
                    U, V, W = U[:, mask], V[:, mask], W[:, mask]
                    rank -= 1
                    changed = True
        return U, V, W, rank

    U, V, W, rank = drop_pass(U, V, W, rank)
    if rank < best_rank:
        best_rank = rank; best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # ---- structured coordinate descent on the smooth residual ----
    def coord_descent(U, V, W, rank, iters):
        for it in range(iters):
            if time.time() > deadline:
                break
            if rank == 0:
                break
            r = np.random.randint(0, rank)
            best_step = None
            base = residual(U, V, W)
            # try small moves on one of the three factors at column r
            for mat, rows in ((U, n*m), (V, m*p), (W, p*n)):
                if time.time() > deadline:
                    break
                for delta in (0.5, -0.5, 1.0, -1.0):
                    if time.time() > deadline:
                        break
                    rr = np.random.randint(0, rows)
                    old = mat[rr, r]
                    mat[rr, r] = old + delta
                    val = residual(U, V, W)
                    mat[rr, r] = old
                    if val < base - 1e-12:
                        base = val
                        best_step = (mat, rr, r, old, old + delta)
            if best_step is not None:
                mat, rr, r, old, new = best_step
                mat[rr, r] = new
                if base < best_res:
                    best_res = base
                if base < 1e-9:
                    if verify(U, V, W):
                        U, V, W, rank = drop_pass(U, V, W, rank)
                        if rank < best_rank:
                            best_rank = rank
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        return U, V, W, rank

    # ---- random perturbation + drop ----
    entry_set = [-1.0, -0.5, 0.0, 0.5, 1.0]
    def random_perturb(U, V, W, rank, iters):
        for it in range(iters):
            if time.time() > deadline:
                break
            target = np.random.randint(0, rank)
            n_perturb = np.random.randint(1, 4)
            orig = []
            for _ in range(n_perturb):
                which = np.random.randint(0, 3)
                mat = U if which == 0 else (V if which == 1 else W)
                rr = np.random.randint(0, mat.shape[0])
                c = np.random.randint(0, rank)
                orig.append((mat, rr, c, mat[rr, c]))
                mat[rr, c] = np.random.choice(entry_set)
            mask = np.ones(rank, dtype=bool); mask[target] = False
            if verify(U[:, mask], V[:, mask], W[:, mask]):
                U, V, W = U[:, mask], V[:, mask], W[:, mask]
                rank -= 1
                return U, V, W, rank
            for mat, rr, c, val in orig:
                mat[rr, c] = val
        return U, V, W, rank

    # ---- rank-reduction: try to express a target column as combo of others ----
    def try_rank_reduce(U, V, W, rank):
        # For each target column, attempt to find coefficients making its
        # tensor contribution cancel. Solve linear system in the free entries.
        if rank < 2:
            return U, V, W, rank
        target = np.random.randint(0, rank)
        # We allow adjusting entries of the OTHER columns to absorb target.
        # Simple heuristic: random linear combination test.
        keep = [c for c in range(rank) if c != target]
        # try a few random coefficient vectors
        for trial in range(20):
            if time.time() > deadline:
                return U, V, W, rank
            # pick a random subset of columns to scale
            coeffs = np.random.choice([0.0, 0.5, 1.0, -1.0], size=len(keep))
            Uc = U.copy(); Vc = V.copy(); Wc = W.copy()
            # add target's contribution into keep columns
            Uc[:, keep] += np.outer(U[:, target], coeffs)
            Vc[:, keep] += np.outer(V[:, target], coeffs)
            Wc[:, keep] += np.outer(W[:, target], coeffs)
            mask = np.ones(rank, dtype=bool); mask[target] = False
            Uc2 = Uc[:, mask]; Vc2 = Vc[:, mask]; Wc2 = Wc[:, mask]
            if verify(Uc2, Vc2, Wc2):
                return Uc2, Vc2, Wc2, rank - 1
        return U, V, W, rank

    # ---- main search loop ----
    while time.time() < deadline - 5.0:
        # coordinate descent for a few iters
        U, V, W, rank = coord_descent(U, V, W, 60)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_res = residual(U, V, W)
        # random perturb
        U, V, W, rank = random_perturb(U, V, W, 30)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_res = residual(U, V, W)
        # rank reduce
        U, V, W, rank = try_rank_reduce(U, V, W, rank)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_res = residual(U, V, W)
        # drop pass
        U, V, W, rank = drop_pass(U, V, W, rank)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_res = residual(U, V, W)
        if best_rank <= 50:
            break

    U, V, W, rank = best_U, best_V, best_W, best_rank
    if not verify(U, V, W):
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1.0; V[j*p+k,col]=1.0; W[k*n+i,col]=1.0; col+=1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
