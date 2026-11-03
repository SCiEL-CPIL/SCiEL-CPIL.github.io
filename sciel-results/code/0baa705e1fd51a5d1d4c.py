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


def _try_merge_pairs(n, m, p):
    """
    Try to find a rank (n*m*p - 1) decomposition by merging two schoolbook terms.
    Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if their difference tensor
    is rank 1 (decomposable).
    """
    if time.time() > _DEADLINE - 30:
        return None

    n_terms = n * m * p
    target_rank = n_terms - 1  # 23

    # Generate all schoolbook terms as (i,j,k) triples
    terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    # For each pair, check if the difference of their rank-1 tensors is itself rank 1
    # Term (i,j,k) corresponds to tensor T_{ijk} with:
    #   T_{ijk}[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # The difference T_{i1j1k1} - T_{i2j2k2} has at most 4 nonzero entries.
    # It is rank 1 iff those entries form a rank-1 pattern.

    # A 3-tensor with support on at most 4 points is rank 1 iff
    # the support forms a "rectangle" in the appropriate sense.
    # Specifically, if the two terms share exactly one index value, the difference
    # has 2 nonzero entries and is rank 1.

    # Let's check: two terms (i1,j1,k1) and (i2,j2,k2)
    # Their difference is nonzero at:
    #   (a1,b1,c1) = (i1*m+j1, j1*p+k1, k1*n+i1) with value +1
    #   (a2,b2,c2) = (i2*m+j2, j2*p+k2, k2*n+i2) with value -1
    # This 2-sparse tensor is rank 1 iff (a1,b1,c1) and (a2,b2,c2) are "aligned"
    # in the sense that the tensor is decomposable.

    # A 2-sparse 3-tensor is rank 1 iff the two support points lie on a line
    # in the tensor space, which for a difference of two rank-1 tensors means
    # the two original rank-1 tensors must be "compatible".

    # Actually, let's just compute it numerically for each pair.
    for idx1 in range(n_terms):
        if time.time() > _DEADLINE - 10:
            return None
        i1, j1, k1 = terms[idx1]
        for idx2 in range(idx1 + 1, n_terms):
            i2, j2, k2 = terms[idx2]

            # Build the difference tensor
            T_diff = np.zeros((n * m, m * p, p * n))
            a1 = i1 * m + j1
            b1 = j1 * p + k1
            c1 = k1 * n + i1
            a2 = i2 * m + j2
            b2 = j2 * p + k2
            c2 = k2 * n + i2
            T_diff[a1, b1, c1] = 1.0
            T_diff[a2, b2, c2] = -1.0

            # Check if T_diff is rank 1 by trying to decompose it
            # A tensor is rank 1 iff it equals u ⊗ v ⊗ w for some vectors.
            # For a 2-sparse tensor, this is equivalent to checking if the
            # two nonzero entries are "compatible".

            # Simple check: if the two points share the same a, or same b, or same c,
            # the tensor might be rank 1. Let's verify numerically.
            # Try to find u, v, w such that u[a1]*v[b1]*w[c1] = 1 and u[a2]*v[b2]*w[c2] = -1
            # and all other entries are 0.

            # Since the tensor is 2-sparse, if it's rank 1, the support must be
            # "degenerate". Let's check: if a1==a2, then we need v[b1]*w[c1] = 1/(u[a1])
            # and v[b2]*w[c2] = -1/(u[a1]), so v[b1]*w[c1] = -v[b2]*w[c2].
            # This is possible if b1!=b2 and c1!=c2, or if one of them is equal.

            # Let's just try a numerical rank-1 decomposition.
            # For a 2-sparse tensor, rank 1 is equivalent to the two entries being
            # related in a specific way. Let's use the unfolding approach.

            # Unfold along first mode: T_diff_unfold[a, (b,c)]
            # The rank of the tensor equals the rank of any unfolding (for rank 1).
            # So check if the matricization has rank 1.

            # Matricize: shape (n*m, m*p * p*n)
            mat = T_diff.reshape(n * m, m * p * p * n)
            # Compute rank via SVD
            try:
                s = np.linalg.svd(mat, compute_uv=False)
                # Count significant singular values
                num_nonzero = np.sum(s > 1e-10)
            except:
                num_nonzero = 2

            if num_nonzero <= 1:
                # The difference is rank 1, so we can merge these two terms!
                # Construct the merged decomposition
                # The merged term is T_{i1j1k1} + T_{i2j2k2} = 2*T_{i1j1k1} - T_diff
                # But T_diff is rank 1, say T_diff = u ⊗ v ⊗ w
                # So T_{i1j1k1} + T_{i2j2k2} = 2*e_{i1j1k1} - u⊗v⊗w
                # This is not directly helpful. Instead:
                # We want to express the sum as a single rank-1 tensor.
                # But the sum of two rank-1 tensors is generally rank 2.
                # If the difference is rank 1, then sum = 2*term1 - diff, which is rank ≤ 2.
                # We need the SUM to be rank 1, not the difference.

                # Let me reconsider. If T_diff = T1 - T2 is rank 1, then T1 = T2 + T_diff.
                # T1 and T2 are both rank 1. T_diff is rank 1.
                # We want to replace T1 and T2 with a single term.
                # T1 + T2 = 2*T1 - T_diff. This is not rank 1 in general.

                # Actually, I made an error. We want to merge T1 and T2 into ONE term.
                # That means we need T1 + T2 to be rank 1, which is NOT what we checked.
                # We checked T1 - T2 is rank 1, which is different.

                # Let me check if T1 + T2 is rank 1 instead.
                T_sum = np.zeros((n * m, m * p, p * n))
                T_sum[a1, b1, c1] = 1.0
                T_sum[a2, b2, c2] = 1.0
                mat_sum = T_sum.reshape(n * m, m * p * p * n)
                try:
                    s_sum = np.linalg.svd(mat_sum, compute_uv=False)
                    num_nonzero_sum = np.sum(s_sum > 1e-10)
                except:
                    num_nonzero_sum = 2

                if num_nonzero_sum <= 1:
                    # T1 + T2 is rank 1! We can merge them.
                    # Find the decomposition of T_sum
                    # T_sum = u ⊗ v ⊗ w
                    # Use SVD of the matricization
                    U_svd, s_vals, Vt_svd = np.linalg.svd(mat_sum, full_matrices=False)
                    if s_vals[0] > 1e-10:
                        # The rank-1 approximation is u = U_svd[:,0] * s_vals[0], 
                        # and the rest is in Vt_svd
                        # But we need exact decomposition.
                        # For a rank-1 tensor, the matricization has rank 1,
                        # so u = mat_sum[:,0] / mat_sum[0,0] if mat_sum[0,0] != 0
                        # Let's find the decomposition directly.
                        
                        # Find nonzero entries
                        nonzero = np.argwhere(T_sum != 0)
                        if len(nonzero) == 2:
                            # Two nonzero entries at (a1,b1,c1) and (a2,b2,c2), both = 1
                            # For rank 1: u[a1]*v[b1]*w[c1] = 1, u[a2]*v[b2]*w[c2] = 1
                            # And all other entries 0.
                            # This requires a1==a2 or b1==b2 or c1==c2 for the zero pattern to work.
                            
                            a1v, b1v, c1v = nonzero[0]
                            a2v, b2v, c2v = nonzero[1]
                            
                            # Check if they share an index
                            if a1v == a2v:
                                # u[a1v] * v[b1v] * w[c1v] = 1
                                # u[a1v] * v[b2v] * w[c2v] = 1
                                # So v[b1v]*w[c1v] = v[b2v]*w[c2v]
                                # Set u[a1v]=1, v[b1v]=1, w[c1v]=1, v[b2v]=1, w[c2v]=1
                                u_vec = np.zeros(n * m)
                                v_vec = np.zeros(m * p)
                                w_vec = np.zeros(p * n)
                                u_vec[a1v] = 1.0
                                v_vec[b1v] = 1.0
                                w_vec[c1v] = 1.0
                                v_vec[b2v] = 1.0
                                w_vec[c2v] = 1.0
                                # Verify
                                T_check = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
                                if np.allclose(T_check, T_sum, atol=1e-10):
                                    # Build the rank-23 decomposition
                                    U = np.zeros((n * m, target_rank))
                                    V = np.zeros((m * p, target_rank))
                                    W = np.zeros((p * n, target_rank))
                                    
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
                    continue

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

    # Try: take schoolbook, remove term idx, and try to express it as a
    # linear combination of the remaining 23 rank-1 tensors.
    # This is equivalent to checking if term_idx is in the span of the others.
    # But rank-1 tensors don't form a linear space, so this is not straightforward.

    # Alternative: use the fact that we can add a small perturbation to
    # two schoolbook terms and see if the total still equals T_target.
    # Let terms r1 and r2 be modified:
    #   U[:,r1] = e_{a1} + t*u, V[:,r1] = e_{b1} + t*v, W[:,r1] = e_{c1} + t*w
    #   U[:,r2] = e_{a2} - t*u, V[:,r2] = e_{b2} - t*v, W[:,r2] = e_{c2} - t*w
    # where u,v,w are chosen so that the extra terms cancel.

    # The extra contribution from r1 is:
    #   t*(e_{a1}⊗e_{b1}⊗(t*w) + t*(e_{a1}⊗v⊗e_{c1}) + t*(t*u⊗e_{b1}⊗e_{c1})
    #     + t*(t*u⊗v⊗e_{c1}) + t*(e_{a1}⊗t*v⊗t*w) + t*(t*u⊗e_{b1}⊗t*w)
    #     + t*(t*u⊗t*v⊗e_{c1}) + t^3*(u⊗v⊗w))
    # This is getting complicated. Let's just use a numerical optimization
    # with a structured initialization.

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
        params = np.zeros(target_rank * (nm + mp + pn))
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
