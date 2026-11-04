"""Matrix multiplication 5x5x5 rank minimization search."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]


# EVOLVE-BLOCK-START
def construct():
    start_wall = time.time()
    HARD_DEADLINE = 142.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_trivial = n * m * p  # 125

    # Schoolbook construction
    U0 = np.zeros((n * m, rank_trivial))
    V0 = np.zeros((m * p, rank_trivial))
    W0 = np.zeros((p * n, rank_trivial))
    for idx in range(rank_trivial):
        i = idx // (m * p)
        r = idx % (m * p)
        j = r // p
        k = r % p
        U0[i * m + j, idx] = 1
        V0[j * p + k, idx] = 1
        W0[k * n + i, idx] = 1

    # Verify schoolbook
    assert np.max(np.abs(U0 @ np.transpose(V0) - np.transpose(W0))) < 1e-9

    def deadline_ok():
        return time.time() < HARD_DEADLINE

    # Helper: check validity of factor set
    def factors_valid(Uf, Vf, Wf, tol=2e-9):
        if Uf.shape[0] != n * m or Vf.shape[0] != m * p or Wf.shape[0] != p * n:
            return False
        if not (Uf.shape[1] == Vf.shape[1] == Wf.shape[1]):
            return False
        L = Uf @ np.transpose(Vf)
        R = np.transpose(Wf)
        d = np.max(np.abs(L - R))
        return d < tol

    def round_to_half(X):
        return (np.round(2.0 * X) / 2.0)

    def polish(Uf, Vf, Wf, max_iter=120):
        # Coordinate gradient descent with backtracking
        U = Uf.copy().astype(float)
        V = Vf.copy().astype(float)
        W = Wf.copy().astype(float)
        r = U.shape[1]
        t = 0.05
        step = 0
        while step < max_iter and deadline_ok():
            step += 1
            # gradient of 0.5||UV^T - W^T||^2
            G = (U @ V.T) - W.T
            # d_U: (n*m, r), G (n*m, p*n) -> G @ W -> wait: loss L = ||U V^T - W^T||_F^2
            # dU = G @ W ? Let me derive: L = sum_{a,c} (U V^T - W^T)[a,c]^2
            # dU = 2 (U V^T - W^T) W ... actually dL/dU = 2 (G) @ W^T?
            # F = U V^T, G = F - W^T, dF/dU = V ⊗ I. dL = 2 <G, dF> = 2 <G, dU V^T> = 2 <G V, dU>
            # So dU = 2 G V
            # dV = 2 G^T U
            # dW = 2 G W^T?  W appears in -W^T, dL/dW^T = -2G, dW = -2 W^T? Let me be careful:
            # L = ||UV^T - W^T||_F^2. d(L)/dW^T = -2(UV^T - W^T) = -2G. dL/dW = -2G^T.
            dU = G @ V
            dV = G.T @ U
            dW = -G.T  # gradient wrt W of L = -2G^T, but we use direction G^T for descent on L
            # For W, minimizing L means moving W against gradient of L wrt W.
            # grad_W(L) = -2 G^T, so descent direction = 2 G^T.
            # For U,V: grad(L) = 2G V and 2G^T U, descent = -grad.
            # Use line search per coordinate; we do simple step with scaling.

            def try_step(U_, V_, W_, du, dv, dw, alpha):
                return (U_ - alpha * du), (V_ - alpha * dv), (W_ + alpha * dw)  # W gradient is -G^T -> descent +G^T

            cur_U, cur_V, cur_W = U, V, W
            Gcur = cur_U @ cur_V.T - cur_W.T
            Lcur = np.sum(Gcur * Gcur)
            improved = False
            for it in range(12):
                U_, V_, W_ = try_step(cur_U, cur_V, cur_W, dU, dV, dW, alpha := t)
                G_ = U_ @ V_.T - W_.T
                L_ = np.sum(G_ * G_)
                if L_ < Lcur * (1 - 1e-10) - 1e-12:
                    U, V, W = U_, V_, W_
                    dU = G_ @ V
                    dV = G_.T @ U
                    dW = -G_
                    Lcur = L_
                    improved = True
                    break
                t *= 0.5
                if t < 1e-12:
                    break
            if not improved:
                t = max(t, 0.01)
        return U, V, W

    def random_half_int(rmax, lo=-8.0, hi=8.0, sparsity=0.8):
        A = np.zeros((rmax,))
        nz = max(1, int(sparsity * rmax))
        idx = np.random.choice(rmax, size=nz, replace=False)
        A[idx] = np.random.choice(np.arange(lo * 2 + 1, hi * 2 + 1) / 2.0, size=nz)
        return A

    best_U, best_V, best_W, best_rank = U0.copy(), V0.copy(), W0.copy(), rank_trivial
    best_score = 0.0  # 1.0 corresponds to rank 125; score = 125/rank
    best_res = (best_U, best_V, best_W)

    rng = np.random.RandomState(12345)

    # Stage 1: coordinate elimination attempts (rank r from 124 down)
    # Try to express one schoolbook term as linear combination of others with half-integer coeffs.
    # This is a structured search; limited attempts to fit deadline.
    r_target_min = 100
    if deadline_ok():
        for r in range(rank_trivial - 1, max(rank_trivial - 8, r_target_min) - 1, -1):
            if not deadline_ok():
                break
            Uf = U0[:, :r].copy()
            Vf = V0[:, :r].copy()
            Wf = W0[:, :r].copy()
            # For each removed column c in [r..rank_trivial-1], try solve U_c = Uf @ a, V_c = Vf @ b, W_c = Wf @ d
            # where a,b,d length r, half-integers. Try least squares then round, then verify.
            success = True
            # We attempt greedy: for c from r to rank-1, find a,b,d; if fail, stop this r
            for c in range(r, rank_trivial):
                if not deadline_ok():
                    success = False
                    break
                Uc = U0[:, c: c + 1]
                Vc = V0[:, c: c + 1]
                Wc = W0[:, c: c + 1]
                # solve for a, b, d
                a, *_ = np.linalg.lstsq(Uf.T @ Uf, Uf.T @ Uc, rcond=None)
                b, *_ = np.linalg.lstsq(Vf.T @ Vf, Vf.T @ Vc, rcond=None)
                d, *_ = np.linalg.lstsq(Wf.T @ Wf, Wf.T @ Wc, rcond=None)
                # try rounding a,b,d to half-integers (several scales)
                ok = False
                for a_r in [a, np.round(a), np.round(2*a)/2, np.rint(a)]:
                    for b_r in [b, np.round(b), np.round(2*b)/2, np.rint(b)]:
                        for d_r in [d, np.round(d), np.round(2*d)/2, np.rint(d)]:
                            errU = np.linalg.norm(Uf @ a_r - Uc)
                            errV = np.linalg.norm(Vf @ b_r - Vc)
                            errW = np.linalg.norm(Wf @ d_r - Wc)
                            if max(errU, errV, errW) < 1e-7:
                                ok = True
                                break
                        if ok: break
                    if ok: break
                if not ok:
                    # random perturb around LSQ
                    for _ in range(30):
                        if not deadline_ok():
                            break
                        a_r = a + 0.5 * rng.randn(*a.shape)
                        b_r = b + 0.5 * rng.randn(*b.shape)
                        d_r = d + 0.5 * rng.randn(*d.shape)
                        a_r = np.round(2*a_r)/2
                        b_r = np.round(2*b_r)/2
                        d_r = np.round(2*d_r)/2
                        errU = np.linalg.norm(Uf @ a_r - Uc)
                        errV = np.linalg.norm(Vf @ b_r - Vc)
                        errW = np.linalg.norm(Wf @ d_r - Wc)
                        if max(errU, errV, errW) < 1e-7:
                            ok = True
                            break
                if not ok:
                    success = False
                    break
            if success and factors_valid(Uf, Vf, Wf):
                best_rank = r
                best_U, best_V, best_W = Uf.copy(), Vf.copy(), Wf.copy()
                break

    # Stage 2: random half-integer factor search for ranks below current best
    r_search_start = max(best_rank - 1, 95)
    rng2 = np.random.RandomState(987654321)
    for target_rank in range(r_search_start, 89, -1):
        if not deadline_ok():
            break
        if target_rank >= best_rank:
            continue
        # Number of random initializations to try this rank
        n_trials = 600
        found = False
        for trial in range(n_trials):
            if not deadline_ok():
                break
            Uf = np.zeros((n * m, target_rank))
            Vf = np.zeros((m * p, target_rank))
            Wf = np.zeros((p * n, target_rank))
            for col in range(target_rank):
                # random sparse half-int columns with varying density
                dens = rng2.choice([0.5, 0.6, 0.7, 0.8, 1.0])
                Uf[:, col] = random_half_int(target_rank, lo=-6, hi=6, sparsity=dens)
                Vf[:, col] = random_half_int(target_rank, lo=-6, hi=6, sparsity=dens)
                Wf[:, col] = random_half_int(target_rank, lo=-6, hi=6, sparsity=dens)
                # Ensure at least one nonzero per col
                if np.all(Uf[:, col] == 0): Uf[0, col] = 0.5
                if np.all(Vf[:, col] == 0): Vf[0, col] = 0.5
                if np.all(Wf[:, col] == 0): Wf[0, col] = 0.5

            # Quick validity check without polish
            if not factors_valid(Uf, Vf, Wf, tol=5e-8):
                continue
            # If valid at loose tol, polish
            Up, Vp, Wp = polish(Uf, Vf, Wf, max_iter=60)
            if factors_valid(Up, Vp, Wp):
                # round to half-integers
                Ur = round_to_half(Up)
                Vr = round_to_half(Vp)
                Wr = round_to_half(Wp)
                if factors_valid(Ur, Vr, Wr):
                    best_rank = target_rank
                    best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
                    found = True
                    break
        if found:
            break

    # Stage 3: local perturbation search around best if rank improved
    if best_rank < rank_trivial and deadline_ok():
        # try to reduce by 1 more via coordinate replacement
        r_try = best_rank - 1
        if r_try >= 90:
            n_trials = 200
            found = False
            for trial in range(n_trials):
                if not deadline_ok():
                    break
                Uf = best_U[:, :r_try].copy()
                Vf = best_V[:, :r_try].copy()
                Wf = best_W[:, :r_try].copy()
                # random perturb last column and a few random entries to half-integers
                Uf[:, r_try - 1] = random_half_int(r_try, lo=-6, hi=6, sparsity=0.7)
                Vf[:, r_try - 1] = random_half_int(r_try, lo=-6, hi=6, sparsity=0.7)
                Wf[:, r_try - 1] = random_half_int(r_try, lo=-6, hi=6, sparsity=0.7)
                if np.all(Uf[:, r_try - 1] == 0): Uf[0, r_try - 1] = 0.5
                if np.all(Vf[:, r_try - 1] == 0): Vf[0, r_try - 1] = 0.5
                if np.all(Wf[:, r_try - 1] == 0): Wf[0, r_try - 1] = 0.5
                Up, Vp, Wp = polish(Uf, Vf, Wf, max_iter=120)
                Ur = round_to_half(Up)
                Vr = round_to_half(Vp)
                Wr = round_to_half(Wp)
                if factors_valid(Ur, Vr, Wr):
                    best_rank = r_try
                    best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
                    found = True
                    break
            if not found and deadline_ok():
                # Also try structured: replace a column with a sum of two others
                # (linear combination identity) - limited random
                for _ in range(50):
                    if not deadline_ok():
                        break
                    r2 = best_rank
                    Uf = best_U.copy()
                    Vf = best_V.copy()
                    Wf = best_W.copy()
                    c = rng2.randint(0, r2 - 1)
                    e1 = rng2.randint(0, r2 - 1)
                    e2 = rng2.randint(0, r2 - 1)
                    if e1 == e2 or e1 == c or e2 == c:
                        continue
                    a = rng2.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                    b = rng2.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                    Uf[:, c] = a * Uf[:, e1] + b * Uf[:, e2]
                    Vf[:, c] = a * Vf[:, e1] + b * Vf[:, e2]
                    Wf[:, c] = a * Wf[:, e1] + b * Wf[:, e2]
                    # Then polish
                    Up, Vp, Wp = polish(Uf, Vf, Wf, max_iter=80)
                    Ur = round_to_half(Up)
                    Vr = round_to_half(Vp)
                    Wr = round_to_half(Wp)
                    if factors_valid(Ur, Vr, Wr):
                        # Verify rank did not increase (could be equal)
                        best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
                        break

    # Final verification
    if not factors_valid(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_trivial

    # Ensure integer/half-integer
    best_U = round_to_half(best_U)
    best_V = round_to_half(best_V)
    best_W = round_to_half(best_W)
    if not factors_valid(best_U, best_V, best_W):
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_trivial

    return {
        "U": np.asarray(best_U).tolist(),
        "V": np.asarray(best_V).tolist(),
        "W": np.asarray(best_W).tolist(),
    }
# EVOLVE-BLOCK-END
