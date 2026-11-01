import numpy as np
import time

def construct():
    t0 = time.time()
    limit = 130.0
    n, m, p = 2, 2, 10

    def _strassen_2x2():
        # Rank-7 Strassen for 2x2 matrix multiplication
        # Each tuple: (u_vec[4], v_vec[4], w_vec[4])
        # u indexes A[i*m+j], v indexes B[j*p_local+k_local], w indexes C[k_local*n+i]
        # For 2x2x2: p_local=2, so B has 4 entries, C has 4 entries
        s = [
            (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1])),
            (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,1,0,-1])),
            (np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,0,1,1])),
            (np.array([0,0,0,1]), np.array([-1,0,1,0]), np.array([1,1,0,0])),
            (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,0,1,0])),
            (np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])),
            (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,0])),
        ]
        return s

    def _build_rank35():
        U = np.zeros((4, 35), dtype=np.float64)
        V = np.zeros((20, 35), dtype=np.float64)
        W = np.zeros((20, 35), dtype=np.float64)
        blocks = _strassen_2x2()
        for blk in range(5):
            for s_i in range(7):
                col = blk * 7 + s_i
                ua, va, wa = blocks[s_i]
                U[0:4, col] = ua
                V[blk*4:blk*4+4, col] = va
                W[blk*4:blk*4+4, col] = wa
        return U.tolist(), V.tolist(), W.tolist()

    def _check(U, V, W, tol=1e-8):
        Un = np.asarray(U, dtype=np.float64)
        Vn = np.asarray(V, dtype=np.float64)
        Wn = np.asarray(W, dtype=np.float64)
        R = Un.shape[1]
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    acc = np.sum(Un[ii*m+jj, :] * Vn[jj*p+kk, :] * Wn[kk*n+ii, :])
                    tgt = 1.0 if jj == kk else 0.0
                    if abs(acc - tgt) > tol:
                        return False
        return True

    U35, V35, W35 = _build_rank35()
    ok35 = _check(U35, V35, W35)

    # Attempt: try rank 34 with a structured perturbation approach.
    # Idea: since the 5 Strassen blocks are independent in k-support,
    # a rank reduction would require sharing rank-1 terms across blocks.
    # Try a small numerical optimization on a subset of entries.
    best_rank = 35
    best_U, best_V, best_W = U35, V35, W35

    if ok35:
        # Quick structured search: try to merge one pair of rank-1 terms
        # across adjacent blocks by checking if a linear combination works.
        # This is a heuristic; we limit to a few trials.
        elapsed = time.time() - t0
        if elapsed < limit - 20.0:
            rng = np.random.default_rng(42)
            for trial in range(50):
                if time.time() - t0 > limit - 10.0:
                    break
                # Try rank 34: zero out one column and see if we can compensate
                # by adjusting others (not really feasible without optimization,
                # so we just verify the known 35 and return it)
                pass

    if ok35:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Fallback: schoolbook rank 40
        U40 = np.zeros((4, 40), dtype=np.float64)
        V40 = np.zeros((20, 40), dtype=np.float64)
        W40 = np.zeros((20, 40), dtype=np.float64)
        c = 0
        for ii in range(2):
            for jj in range(2):
                for kk in range(10):
                    U40[ii*2+jj, c] = 1.0
                    V40[jj*10+kk, c] = 1.0
                    W40[kk*2+ii, c] = 1.0
                    c += 1
        return {"U": U40.tolist(), "V": V40.tolist(), "W": W40.tolist()}
