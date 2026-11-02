import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _schoolbook(n, m, p):
    """Exact rank n*m*p decomposition (guaranteed fallback)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for t, (i, j, k) in enumerate(
        ((i, j, k) for i in range(n) for j in range(m) for k in range(p))
    ):
        U[i * m + j, t] = 1.0
        V[j * p + k, t] = 1.0
        W[k * n + i, t] = 1.0
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check sum_t U[:,t] * V[:,t] * W[:,t] == T (the matmul tensor)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    recon = np.einsum('at,bt,ct->abc', U, V, W)
    return float(np.max(np.abs(recon - T))) <= tol


def _round_half(U, V, W):
    """Round entries to nearest half-integer and verify."""
    Ur = np.round(U * 2.0) / 2.0
    Vr = np.round(V * 2.0) / 2.0
    Wr = np.round(W * 2.0) / 2.0
    n, m, p = DIMENSIONS
    if _verify(Ur, Vr, Wr, n, m, p, tol=1e-4):
        return Ur, Vr, Wr
    return None


def _grappe_search(n, m, p, target_rank, seed=0, iters=2000, lr=0.02, deadline=None):
    """Grappe-style alternating minimization + rank deflation."""
    rng = np.random.default_rng(seed)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    U = rng.standard_normal((n * m, target_rank)) * 0.3
    V = rng.standard_normal((m * p, target_rank)) * 0.3
    W = rng.standard_normal((p * n, target_rank)) * 0.3

    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        # Alternate: fix two, optimize the third via least squares
        # Optimize U: for each t, U[:,t] = (V[:,t] * W[:,t])^T contracted with T
        # T reshaped: T[a,b,c]. We need sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
        # For fixed V,W: U[:,t] should satisfy sum_t U[a,t]*(V[b,t]*W[c,t]) = T[a,b,c]
        # This is: U @ diag(V[b,t]*W[c,t]) ... actually per t:
        # sum_t U[a,t] * (V[b,t]*W[c,t]) = T[a,b,c]
        # For each t, define M_t[a, (b,c)] = V[b,t]*W[c,t], then U[:,t] = argmin ||U[:,t] M_t - T_t||
        # But T is shared. Simpler: U[:,t] = (V[:,t] * W[:,t]) outer with T flattened? 
        # Actually: sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
        # => U[:,t] = (T_{a,b,c} weighted by V[b,t]W[c,t]) ... 
        # Least squares: minimize ||sum_t U[:,t] (V[:,t]*W[:,t])^T - T||_F
        # For fixed V,W, U[:,t] = argmin sum_t ... 
        # Gradient: U[:,t] should be (T * (V[:,t]*W[:,t]) summed over b,c) normalized
        # Simpler alternating: 
        # U[:,t] = (T reshaped) contracted: for each t, U[:,t] = einsum('abc,bt,ct->at', T, V, W) / norm
        U_new = np.einsum('abc,bt,ct->at', T, V, W)
        # Normalize columns to prevent blowup
        norms = np.linalg.norm(U_new, axis=0, keepdims=True)
        norms[norms < 1e-12] = 1.0
        U_new = U_new / norms
        U = U_new
        
        V_new = np.einsum('abc,at,ct->bt', T, U, W)
        norms = np.linalg.norm(V_new, axis=0, keepdims=True)
        norms[norms < 1e-12] = 1.0
        V = V_new / norms
        
        W_new = np.einsum('abc,at,bt->ct', T, U, V)
        norms = np.linalg.norm(W_new, axis=0, keepdims=True)
        norms[norms < 1e-12] = 1.0
        W = W_new / norms
        
        err = np.max(np.abs(np.einsum('at,bt,ct->abc', U, V, W) - T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-8:
            break

    # Try to deflate rank: remove columns with small contribution
    # Keep only top-rank columns by contribution
    contribs = np.abs(np.einsum('at,bt,ct->t', U, V, W))  # rough
    # Actually, just try rounding at current rank
    rounded = _round_half(best_U, best_V, best_W)
    if rounded is not None:
        return rounded
    
    # Try deflation: zero out weakest column, re-optimize
    for col_idx in range(target_rank - 1, 0, -1):
        if deadline is not None and time.time() > deadline:
            break
        U2 = best_U[:, :col_idx]
        V2 = best_V[:, :col_idx]
        W2 = best_W[:, :col_idx]
        # Quick re-optimize
        for _ in range(200):
            U2 = np.einsum('abc,bt,ct->at', T, V2, W2)
            norms = np.linalg.norm(U2, axis=0, keepdims=True)
            norms[norms < 1e-12] = 1.0
            U2 = U2 / norms
            V2 = np.einsum('abc,at,ct->bt', T, U2, W2)
            norms = np.linalg.norm(V2, axis=0, keepdims=True)
            norms[norms < 1e-12] = 1.0
            V2 = V2 / norms
            W2 = np.einsum('abc,at,bt->ct', T, U2, V2)
            norms = np.linalg.norm(W2, axis=0, keepdims=True)
            norms[norms < 1e-12] = 1.0
            W2 = W2 / norms
        
        rounded = _round_half(U2, V2, W2)
        if rounded is not None:
            return rounded
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 96
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Grappe search for lower ranks
    # Known: rank(4x4x4) <= 49 (Strassen), so 4x4x6 might be <= 49+32=81 or lower
    # Try ranks: 80, 75, 70, 65, 60, 55, 50
    for target_rank in [80, 75, 70, 65, 60, 55, 50, 45]:
        if time.time() > deadline - 30:
            break
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            result = _grappe_search(n, m, p, target_rank, seed=seed, iters=1500, deadline=deadline - 25)
            if result is not None:
                Ur, Vr, Wr = result
                r = Ur.shape[1]
                if r < best_rank:
                    best_rank = r
                    U, V, W = Ur, Vr, Wr
                break  # Found a valid decomposition at this rank, try lower
    
    # Final: use the best found
    # Ensure shapes are correct and entries are half-integers
    def to_numeric(mat):
        out = []
        for row in mat:
            new_row = []
            for x in row:
                xf = float(x)
                if abs(xf - round(xf)) < 1e-9:
                    new_row.append(int(round(xf)))
                else:
                    new_row.append(xf)
            out.append(new_row)
        return out
    
    return {"U": to_numeric(U), "V": to_numeric(V), "W": to_numeric(W)}
