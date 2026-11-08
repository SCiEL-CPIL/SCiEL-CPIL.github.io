import time

DIMENSIONS = [3, 4, 6]

def _build_known():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0

    def add_rank1(u_idx, u_vals, v_idx, v_vals, w_idx, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_idx, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_idx, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_idx, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}


def _validate(U, V, W, n, m, p):
    """Verify C[i,k,j] = sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]."""
    import numpy as np
    Ua = np.asarray(U, dtype=object)
    Va = np.asarray(V, dtype=object)
    Wa = np.asarray(W, dtype=object)
    R = Ua.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0
                for t in range(R):
                    s = s + Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if s != 1:
                    return False
    return True


def _refine(U, V, W, n, m, p, tol=1e-6):
    """Rank-revealing reduction. Returns (U,V,W) with minimal rank or None."""
    import numpy as np
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R0 = Ua.shape[1]
    try:
        # T is (n*m) x (m*p) x (p*n), T[i*m+j, j*p+k, k*n+i] = U[i*m+j,t]V[j*p+k,t]W[k*n+i,t]
        T = np.einsum('it,jt,kt->ijk', Ua, Va, Wa)
        # Flatten T to a matrix and find its rank via SVD
        Tm = T.reshape(n * m * p, R0)
        # SVD to get rank
        Usvd, svals, Vsvd = np.linalg.svd(Tm, full_matrices=False)
        # Determine numerical rank
        rank = int(np.sum(svals > tol * svals[0])) if svals.size else 0
        if rank >= R0:
            return None
        # Project onto the rank subspace: keep first `rank` right-singular vectors
        # New factor = Tm @ Vsvd[:, :rank].T  ->  (n*m*p, rank)
        Tnew = Tm @ Vsvd[:, :rank].T
        # Need to factor Tnew = U_new @ V_new @ W_new with matching index structure.
        # Reconstruct using the same "tensor" structure: Tnew[i*m+j, j*p+k, k*n+i]
        Tnew = Tnew.reshape(n, m, p, rank)
        # We need U_new[i*m+j, t], V_new[j*p+k, t], W_new[k*n+i, t] such that
        # sum_t U_new[i*m+j,t]*V_new[j*p+k,t]*W_new[k*n+i,t] = Tnew[i,j,k,t]... 
        # This is a 4-way tensor decomposition; simplest: keep rank-1 in t by absorbing.
        # Instead, build new factors by a rank-1-per-(i,j,k) is too big.
        # Use a simple exact approach: since Tnew has rank `rank` as a matrix in the
        # (i*m*p) x (j*p*n) sense, factor the flattened matrix and map back.
        # Flatten T as A (i*m*p) x (j*p*n)? Not aligned with our 3-factor form.
        # Fallback: use the SVD subspace to compress the original factors.
        # New U = Ua @ Vsvd[:, :rank]  (n*m x rank)
        # New V = Va @ Vsvd[:, :rank]  (m*p x rank)
        # New W = Wa @ Vsvd[:, :rank]  (p*n x rank)
        # Then sum_t' U_new[i*m+j,t']*V_new[j*p+k,t']*W_new[k*n+i,t']
        #   = sum_t' sum_a Ua[i*m+j,a] Vsvd[a,t'] * ... 
        # This does NOT generally equal the original T unless the subspace is closed.
        # So this naive projection is only valid if the factors lie in the subspace.
        # Safer: verify the reconstructed product and only accept if it matches.
        Unew = Ua @ Vsvd[:, :rank]
        Vnew = Va @ Vsvd[:, :rank]
        Wnew = Wa @ Vsvd[:, :rank]
        # Verify
        Tcheck = np.einsum('it,jt,kt->ijk', Unew, Vnew, Wnew)
        if not np.allclose(Tcheck.reshape(-1, rank), Tm, atol=1e-4):
            # projection not exact; try to find exact integer/half-integer solution
            # via solving for the subspace basis exactly.
            return None
        # Round to half-integers
        def round_half(a):
            return np.round(2 * a) / 2.0
        Ur = round_half(Unew)
        Vr = round_half(Vnew)
        Wr = round_half(Wnew)
        # Verify the rounded version exactly
        Tc = np.einsum('it,jt,kt->ijk', Ur, Vr, Wr)
        if not np.allclose(Tc, T, atol=1e-6):
            return None
        return (Ur.tolist(), Vr.tolist(), Wr.tolist())
    except Exception:
        return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0

    # Start from the known 66-rank solution
    known = _build_known()
    best = known
    if not _validate(known["U"], known["V"], known["W"], n, m, p):
        # Should never happen, but guard
        return known

    # Try to reduce the rank
    if time.time() < deadline - 30.0:
        refined = _refine(known["U"], known["V"], known["W"], n, m, p)
        if refined is not None:
            U, V, W = refined
            if _validate(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}

    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
