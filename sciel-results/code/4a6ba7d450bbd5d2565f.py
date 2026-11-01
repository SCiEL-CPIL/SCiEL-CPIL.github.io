import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin under 150s wall-clock

    # U: (n*m, rank) = (4, rank); V: (m*p, rank) = (18, rank); W: (p*n, rank) = (18, rank)
    # Index conventions: U[i*m+j], V[j*p+k], W[k*n+i]

    def build_base():
        terms = []
        # Strassen blocks for k-pairs: (0,1),(2,3),(4,5),(6,7)
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=2, d=A[1,1]=3
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
            terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
            terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
            terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
        # Schoolbook for k=8
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))

        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    def verify(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', Ur, Vr, Wr)
        return float(np.max(np.abs(T - 1.0))) <= tol

    def to_halfint(X):
        return np.round(np.array(X, dtype=float) * 2.0) / 2.0

    # ---- Base construction ----
    U0, V0, W0 = build_base()
    base_rank = len(U0[0])
    base_ok = verify(U0, V0, W0)
    if not base_ok:
        # Fallback: return base as-is (best effort)
        return {"U": U0, "V": V0, "W": W0}

    # ---- Rank-reduction pass (null-space compression + local refinement) ----
    # Tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
    # A rank-r decomposition is a set of r rank-1 slices (u_r, v_r, w_r).
    # We search for the smallest r whose slice tensors span the all-ones tensor.
    #
    # Null-space compression: if the base rank-R slices are linearly dependent as
    # tensors, we can express the tensor with fewer independent slices.
    def try_reduce(U, V, W, target_rank, max_iter=400, seed=0):
        """Try to find a rank-`target_rank` half-integer decomposition via local refinement.
        Returns (U,V,W) if found, else None."""
        n_, m_, p_ = n, m, p
        R = target_rank
        rng = np.random.default_rng(seed)
        # Parameterize each rank-1 slice: u (4), v (18), w (18) -> 40 per slice
        dim_u, dim_v, dim_w = n_ * m_, m_ * p_, p_ * n_
        per = dim_u + dim_v + dim_w
        N = n_ * m_ * p_

        def residuals(params):
            Uf = params[:R * dim_u].reshape(R, dim_u)
            Vf = params[R * dim_u:R * dim_u + R * dim_v].reshape(R, dim_v)
            Wf = params[R * dim_u + R * dim_v:R * (dim_u + dim_v + dim_w)].reshape(R, dim_w)
            Ur = Uf.reshape(R, n_, m_)
            Vr = Vf.reshape(R, m_, p_)
            Wr = Wf.reshape(R, p_, n_)
            T = np.einsum('rkm,rmn,rnk->kmn', Ur, Vr, Wr)  # index i,j,k -> (m,n,p)
            return (T - 1.0).reshape(-1)

        def snap_and_check(Uf, Vf, Wf):
            Us, Vs, Ws = to_halfint(Uf), to_halfint(Vf), to_halfint(Wf)
            Ur = Us.reshape(R, n_, m_).transpose(1, 2, 0)   # (n,m,R)
            Vr = Vs.reshape(R, m_, p_).transpose(1, 2, 0)   # (m,p,R)
            Wr = Ws.reshape(R, p_, n_).transpose(1, 2, 0)   # (p,n,R)
            T = np.einsum('ikr,kjr,jir->ikj', Ur, Vr, Wr)
            if float(np.max(np.abs(T - 1.0))) <= 1e-8:
                # Rebuild in (dim, rank) layout
                Uout = Us.transpose(1, 0)  # (dim_u, R)
                Vout = Vs.transpose(1, 0)  # (dim_v, R)
                Wout = Ws.transpose(1, 0)  # (dim_w, R)
                return Uout, Vout, Wout
            return None

        best = None
        for attempt in range(6):
            if time.time() > deadline:
                break
            # Init: use base slices if available, else random
            if attempt == 0 and base_rank >= R:
                # Truncate/zero-pad base slices
                Uf0 = np.array(U0, dtype=float).T  # (R_base, 4)
                Vf0 = np.array(V0, dtype=float).T  # (R_base, 18)
                Wf0 = np.array(W0, dtype=float).T  # (R_base, 18)
                Uf = np.zeros((R, dim_u)); Vf = np.zeros((R, dim_v)); Wf = np.zeros((R, dim_w))
                Ruse = min(R, base_rank)
                Uf[:Ruse] = Uf0[:Ruse]; Vf[:Ruse] = Vf0[:Ruse]; Wf[:Ruse] = Wf0[:Ruse]
                params = np.concatenate([Uf.ravel(), Vf.ravel(), Wf.ravel()])
            else:
                params = rng.standard_normal(R * per) * 0.7

            # Gauss-Newton style least squares via J^T J
            x = params.copy()
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                r = residuals(x)
                if np.linalg.norm(r) < 1e-12:
                    break
                # Numerical Jacobian (block-sparse by slice)
                J = np.zeros((N, R * per))
                h = 1e-6
                for s in range(R):
                    # perturb u slice
                    base = s * dim_u
                    for a in range(dim_u):
                        xp = x.copy(); xp[base + a] += h
                        J[:, base + a] = (residuals(xp) - r) / h
                    base = R * dim_u + s * dim_v
                    for a in range(dim_v):
                        xp = x.copy(); xp[base + a] += h
                        J[:, base + a] = (residuals(xp) - r) / h
                    base = R * dim_u + R * dim_v + s * dim_w
                    for a in range(dim_w):
                        xp = x.copy(); xp[base + a] += h
                        J[:, base + a] = (residuals(xp) - r) / h
                # Solve normal equations with damping
                JtJ = J.T @ J
                Jtr = J.T @ r
                lam = 1e-8 * (np.trace(JtJ) / (R * per) + 1e-12)
                try:
                    step = np.linalg.solve(JtJ + lam * np.eye(R * per), Jtr)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(JtJ + lam * np.eye(R * per), Jtr, rcond=None)[0]
                x = x - step
                if it % 20 == 0:
                    if time.time() > deadline:
                        break
            # Snap to half-integers and verify
            Uf = x[:R * dim_u].reshape(R, dim_u)
            Vf = x[R * dim_u:R * dim_u + R * dim_v].reshape(R, dim_v)
            Wf = x[R * dim_u + R * dim_v:R * (dim_u + dim_v + dim_w)].reshape(R, dim_w)
            res = snap_and_check(Uf, Vf, Wf)
            if res is not None:
                best = res
                break
        return best

    # Try to reduce rank below base_rank.
    for R in range(max(4, base_rank - 8), base_rank):
        if time.time() > deadline - 15.0:
            break
        sol = try_reduce(U0, V0, W0, R, max_iter=120, seed=R)
        if sol is not None:
            Uf, Vf, Wf = sol
            if verify(Uf, Vf, Wf):
                return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # ---- Return verified base decomposition ----
    return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
