import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145  # 5s safety margin before 150s kill


def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, n, m, p, tol=1e-10):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_target = _build_target_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _check_rank1_2sparse(nm, mp, pn, a1, b1, c1, a2, b2, c2):
    """Check if a 2-sparse tensor with entries at (a1,b1,c1) and (a2,b2,c2), both = 1,
    is rank 1. Returns (u, v, w) if rank 1, else None."""
    # For rank 1: u[a1]*v[b1]*w[c1] = 1, u[a2]*v[b2]*w[c2] = 1
    # and all other entries must be 0.
    # If a1 != a2, b1 != b2, c1 != c2: impossible (would need 4 nonzeros)
    # If two indices are equal, we can try.
    
    u = np.zeros(nm)
    v = np.zeros(mp)
    w = np.zeros(pn)
    
    # Case 1: a1 == a2
    if a1 == a2:
        u[a1] = 1.0
        # Need v[b1]*w[c1] = 1 and v[b2]*w[c2] = 1
        # If b1 == b2 and c1 == c2: trivial (same point)
        # If b1 == b2: need c1 == c2
        # If c1 == c2: need b1 == b2
        # Otherwise: set v[b1]=1, w[c1]=1, v[b2]=1, w[c2]=1
        # But then u[a1]*v[b1]*w[c1] = 1*1*1 = 1 OK
        # and u[a1]*v[b2]*w[c2] = 1*1*1 = 1 OK
        # But we need to check no other entries are nonzero.
        # With u having one nonzero, v having two, w having two:
        # The tensor has nonzeros at (a1, b1, c1), (a1, b1, c2), (a1, b2, c1), (a1, b2, c2)
        # That's 4 nonzeros unless some indices coincide.
        # So we need either b1==b2 or c1==c2.
        if b1 == b2:
            if c1 != c2:
                return None  # Would have 2 nonzeros at (a1,b1,c1) and (a1,b1,c2), but c1!=c2 means different
            # Same point
            u[a1] = 1.0
            v[b1] = 1.0
            w[c1] = 1.0
            return (u, v, w)
        elif c1 == c2:
            if b1 != b2:
                return None
            u[a1] = 1.0
            v[b1] = 1.0
            w[c1] = 1.0
            return (u, v, w)
        else:
            # b1 != b2 and c1 != c2: 4 nonzeros, not rank 1
            return None
    
    # Case 2: b1 == b2
    if b1 == b2:
        v[b1] = 1.0
        if a1 == a2:
            if c1 != c2:
                return None
            u[a1] = 1.0
            v[b1] = 1.0
            w[c1] = 1.0
            return (u, v, w)
        elif c1 == c2:
            if a1 != a2:
                return None
            u[a1] = 1.0
            v[b1] = 1.0
            w[c1] = 1.0
            return (u, v, w)
        else:
            # a1 != a2 and c1 != c2: 4 nonzeros
            return None
    
    # Case 3: c1 == c2
    if c1 == c2:
        w[c1] = 1.0
        if a1 == a2:
            if b1 != b2:
                return None
            u[a1] = 1.0
            v[b1] = 1.0
            w[c1] = 1.0
            return (u, v, w)
        elif b1 == b2:
            if a1 != a2:
                return None
            u[a1] = 1.0
            v[b1] = 1.0
            w[c1] = 1.0
            return (u, v, w)
        else:
            # a1 != a2 and b1 != b2: 4 nonzeros
            return None
    
    # No shared index: 2 nonzeros at completely different positions
    # A rank-1 tensor with 2 nonzeros must have them sharing at least one index
    return None


def _try_merge_pairs(n, m, p):
    """
    Try to find a rank (n*m*p - 1) decomposition by merging two schoolbook terms
    whose sum is rank 1.
    """
    if time.time() > _DEADLINE - 30:
        return None

    n_terms = n * m * p
    target_rank = n_terms - 1  # 23
    nm, mp, pn = n * m, m * p, p * n

    # Generate all schoolbook terms as (i,j,k) triples
    terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    # For each pair, check if the SUM of their rank-1 tensors is rank 1
    for idx1 in range(n_terms):
        if time.time() > _DEADLINE - 10:
            return None
        i1, j1, k1 = terms[idx1]
        a1 = i1 * m + j1
        b1 = j1 * p + k1
        c1 = k1 * n + i1
        for idx2 in range(idx1 + 1, n_terms):
            i2, j2, k2 = terms[idx2]
            a2 = i2 * m + j2
            b2 = j2 * p + k2
            c2 = k2 * n + i2

            # Check if the sum (2-sparse, both entries = 1) is rank 1
            result = _check_rank1_2sparse(nm, mp, pn, a1, b1, c1, a2, b2, c2)
            if result is not None:
                u_vec, v_vec, w_vec = result
                # Verify the merged term
                T_check = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
                T_expected = np.zeros((nm, mp, pn))
                T_expected[a1, b1, c1] = 1.0
                T_expected[a2, b2, c2] = 1.0
                if np.allclose(T_check, T_expected, atol=1e-10):
                    # Build the rank-23 decomposition
                    U = np.zeros((nm, target_rank))
                    V = np.zeros((mp, target_rank))
                    W = np.zeros((pn, target_rank))

                    col = 0
                    for t_idx, (ti, tj, tk) in enumerate(terms):
                        if t_idx == idx1 or t_idx == idx2:
                            continue
                        U[ti * m + tj, col] = 1
                        V[tj * p + tk, col] = 1
                        W[tk * n + ti, col] = 1
                        col += 1

                    # Add the merged term
                    U[:, col] = u_vec
                    V[:, col] = v_vec
                    W[:, col] = w_vec

                    if _verify(U, V, W, n, m, p, tol=1e-9):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None


def _try_rank_23(n, m, p):
    """
    Attempt to find a rank 23 decomposition using structured approach.
    """
    if time.time() > _DEADLINE - 20:
        return None

    # First try the merge-pairs approach
    res = _try_merge_pairs(n, m, p)
    if res is not None:
        return res

    # Try a parameterized search: start from schoolbook, perturb two terms
    # and solve for parameters that maintain the tensor equality.
    
    n_terms = n * m * p
    target_rank = 23

    T_target = _build_target_tensor(n, m, p)

    # Use scipy if available, otherwise fall back to schoolbook
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    if time.time() > _DEADLINE - 10:
        return None

    def objective(params):
        r = target_rank
        nm, mp, pn = n * m, m * p, p * n
        U = params[:r*nm].reshape(r, nm).T  # (nm, r)
        V = params[r*nm:r*nm+r*mp].reshape(r, mp).T  # (mp, r)
        W = params[r*nm+r*mp:].reshape(r, pn).T  # (pn, r)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_recon - T_target) ** 2)

    # Initialize with schoolbook (24 terms) truncated to 23
    rng = np.random.default_rng(123)
    nm, mp, pn = n * m, m * p, p * n

    for attempt in range(5):
        if time.time() > _DEADLINE - 5:
            return None

        # Initialize: use 23 of the 24 schoolbook terms, perturb the last
        U_init = np.zeros((nm, target_rank))
        V_init = np.zeros((mp, target_rank))
        W_init = np.zeros((pn, target_rank))

        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        for c in range(22):
            i, j, k = terms[c]
            U_init[i * m + j, c] = 1
            V_init[j * p + k, c] = 1
            W_init[k * n + i, c] = 1

        # For the 23rd term, use a random perturbation of the 24th term
        i23, j23, k23 = terms[23]
        U_init[i23 * m + j23, 22] = 1.0
        V_init[j23 * p + k23, 22] = 1.0
        W_init[k23 * n + i23, 22] = 1.0

        # Add small random perturbation
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        params += rng.standard_normal(params.shape) * 0.01

        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < 1e-8:
                r = target_rank
                U_opt = res.x[:r*nm].reshape(r, nm).T
                V_opt = res.x[r*nm:r*nm+r*mp].reshape(r, mp).T
                W_opt = res.x[r*nm+r*mp:].reshape(r, pn).T
                if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-9):
                    return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        except Exception:
            continue

    return None


def construct():
    n, m, p = 2, 3, 4

    # Try rank 23
    res = _try_rank_23(n, m, p)
    if res is not None:
        return res

    # Fallback: Schoolbook
    return _schoolbook(n, m, p)
