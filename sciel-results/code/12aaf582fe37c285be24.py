import numpy as np
import time

_START = time.time()
def _time_left():
    return 150.0 - (time.time() - _START)

def _build_tensor(n, m, p):
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _terms_schoolbook(n, m, p):
    """48 rank-1 terms (a_i_j)(b_j_k)(c_k_i)."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = np.zeros(n*m); V = np.zeros(m*p); W = np.zeros(p*n)
                U[i*m + j] = 1
                V[j*p + k] = 1
                W[k*n + i] = 1
                terms.append((U, V, W))
    return terms

def _verify(U, V, W, n, m, p, tol=1e-6):
    r = len(U[0])
    Ua = np.array(U, dtype=np.float64).reshape(n, m, r)
    Va = np.array(V, dtype=np.float64).reshape(m, p, r)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
    target = np.ones((n, m, p), dtype=np.float64)
    return float(np.max(np.abs(T - target))) < tol

def _construct_strassen(n, m, p):
    """Original rank-44 hybrid construction (guaranteed valid)."""
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # m1 = (a+d)(e+h)
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            # m2 = (c+d)e
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            W[(b_c*2+1)*n + 1][col] = -1
            col += 1
            # m3 = a(f-h)
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            # m4 = d(g-e)
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            col += 1
            # m5 = (a+b)h
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            col += 1
            # m6 = (c-a)(e+f)
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            # m7 = (b-d)(g+h)
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            col += 1
            # Trivial for row i=2
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    return U, V, W

def _to_lists(U, V, W):
    return (np.array(U).astype(np.int64).tolist(),
            np.array(V).astype(np.int64).tolist(),
            np.array(W).astype(np.int64).tolist())

def _search(n, m, p):
    """Try to find a decomposition of rank < 44 by merging terms."""
    best = _construct_strassen(n, m, p)
    best_rank = 44
    best_verified = _verify(best[0], best[1], best[2], n, m, p)
    if not best_verified:
        # should not happen, but fall back to schoolbook if needed
        T = _build_tensor(n, m, p)
        terms = _terms_schoolbook(n, m, p)
        best = ([t[0][i*m+j] if False else 0 for i in range(n*m)] for _ in range(1))
        return None

    # Convert to numpy matrices
    U0 = np.array(best[0], dtype=np.float64)
    V0 = np.array(best[1], dtype=np.float64)
    W0 = np.array(best[2], dtype=np.float64)
    r = U0.shape[1]
    T_target = np.ones((n, m, p), dtype=np.float64)

    def residual(Um, Vm, Wm):
        Ua = Um.reshape(n, m, -1)
        Va = Vm.reshape(m, p, -1)
        Wa = Wm.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
        return float(np.max(np.abs(T - T_target)))

    # Build the 48 schoolbook term matrices
    terms = _terms_schoolbook(n, m, p)
    n_terms = len(terms)

    def merge_terms(Um, Vm, Wm, a, b, alpha, beta, tol=1e-8):
        """Replace terms a and b with their linear combination alpha*term_a + beta*term_b.
        Returns (new_Um, new_Vm, new_Wm, new_r) if combination is valid (sum reconstructs).
        The combined term is alpha*Ua + beta*Ua' (elementwise) etc.
        We remove one of the two columns and replace it with the sum."""
        # combined = alpha*col_a + beta*col_b (elementwise)
        new_col_U = alpha*Um[:, a] + beta*Um[:, b]
        new_col_V = alpha*Vm[:, a] + beta*Vm[:, b]
        new_col_W = alpha*Wm[:, a] + beta*Wm[:, b]
        # Keep term a, set term b to zero, then set term a to combined
        new_Um = Um.copy()
        new_Vm = Vm.copy()
        new_Wm = Wm.copy()
        new_Um[:, a] = new_col_U
        new_Um[:, b] = 0
        new_Vm[:, a] = new_col_V
        new_Vm[:, b] = 0
        new_Wm[:, a] = new_col_W
        new_Wm[:, b] = 0
        return new_Um, new_Vm, new_Wm

    # Search for valid merges: for each pair (a,b) and each (alpha,beta) in a small set,
    # check if merging reduces the residual to near-zero.
    # This is expensive, so limit to a small candidate set and a time budget.
    alphas = [1, -1, 0.5, 2]
    betas  = [1, -1, 0.5, 2, 0]
    candidate_pairs = [(a, b) for a in range(r) for b in range(a+1, r)]

    # Sort pairs by "potential": prefer pairs where the combined term might be sparser
    # or where the two terms overlap in support.
    def pair_score(a, b):
        # Overlap in U, V, W supports
        s = 0
        s += np.sum((U0[:, a] != 0) & (U0[:, b] != 0))
        s += np.sum((V0[:, a] != 0) & (V0[:, b] != 0))
        s += np.sum((W0[:, a] != 0) & (W0[:, b] != 0))
        return s

    candidate_pairs.sort(key=lambda ab: -pair_score(ab[0], ab[1]))

    # Limit number of pairs to try based on time
    max_pairs = min(len(candidate_pairs), 500)
    tried = 0
    for (a, b) in candidate_pairs[:max_pairs]:
        if tried % 20 == 0 and _time_left() < 20:
            break
        for alpha in alphas:
            for beta in betas:
                if tried % 50 == 0 and _time_left() < 15:
                    break
                tried += 1
                new_Um, new_Vm, new_Wm = merge_terms(U0, V0, W0, a, b, float(alpha), float(beta))
                res = residual(new_Um, new_Vm, new_Wm)
                if res < 1e-6:
                    # Valid merge! Rank is now r (one column is zero, but we can drop it)
                    # Actually we removed one column's content, so effective rank is r-1
                    # Let's construct the new matrices with r-1 columns
                    # Find which column is now zero
                    new_U = new_Um[:, a].copy()  # combined
                    # Build list of columns: all except b (since b is zero)
                    keep = [c for c in range(r) if c != b]
                    new_Uf = new_Um[:, keep]
                    new_Vf = new_Vm[:, keep]
                    new_Wf = new_Wm[:, keep]
                    new_r = new_Uf.shape[1]
                    if new_r < best_rank:
                        best = (new_Uf.tolist(), new_Vf.tolist(), new_Wf.tolist())
                        best_rank = new_r
                        U0, V0, W0 = new_Uf, new_Vf, new_Wf
                        r = new_r
                        # Restart search with new base
                        break
            if _time_left() < 10:
                break
        if _time_left() < 10:
            break

    return best

def construct():
    n, m, p = 3, 4, 4
    # Fallback: guaranteed rank-44
    U, V, W = _construct_strassen(n, m, p)
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but ensure we return something
        return {"U": U, "V": V, "W": W}

    # Try to improve
    if _time_left() > 30:
        result = _search(n, m, p)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Final: return the best verified result
    return {"U": U, "V": V, "W": W}
