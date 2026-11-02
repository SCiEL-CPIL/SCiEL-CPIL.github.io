"""Matrix multiplication 2x3x7: extended search with random restarts."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]


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


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W (as float arrays of shape (n*m, r), (m*p, r), (p*n, r))
    exactly decompose the multiplication tensor (within integer/half-integer exactness)."""
    r = U.shape[1]
    # Build T_3 = W @ U^T  (shape (p*n, n*m))
    T3 = W @ U.T
    ok = True
    for i in range(n):
        for k in range(p):
            expected = 1.0 if k == i else 0.0
            actual = T3[k * n + i, i * m]
            if abs(actual - expected) > 1e-6:
                ok = False
    # Also verify T_2 = U @ V^T
    T2 = U @ V.T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T2[i * m + j, j * p + k]
                if abs(actual - expected) > 1e-6:
                    ok = False
    # Verify T_1 = V @ W^T
    T1 = V @ W.T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T1[i * m + j, k * n + i]
                if abs(actual - expected) > 1e-6:
                    ok = False
    return ok


def _to_list(mat):
    out = []
    for row in mat:
        r = []
        for val in row:
            # Snap to half-integer
            hv = np.round(val * 2.0) / 2.0
            if abs(hv - round(hv)) < 1e-9:
                r.append(int(round(hv)))
            else:
                r.append(float(hv))
        out.append(r)
    return out


def _peel_random(n, m, p, max_rank, seed=0):
    """Greedy rank-1 peeling with random perturbations."""
    rng = np.random.default_rng(seed)
    
    T3 = np.zeros((p * n, n * m))
    for i in range(n):
        for k in range(p):
            T3[k * n + i, i * m] = 1.0
            
    # Add small random noise to break symmetries
    T3 += rng.normal(0, 1e-4, T3.shape)

    terms = []
    scale = 3.0 ** 0.25
    for r in range(max_rank):
        U, s, Vt = np.linalg.svd(T3, full_matrices=False)
        if len(s) == 0 or s[0] <= 1e-9:
            break
        u_vec = Vt[0, :]
        w_vec = U[:, 0]
        
        T2 = np.zeros((n * m, m * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T2[i * m + j, j * p + k] = 1.0
        
        v_vec = T2 @ u_vec / scale
        
        nu = np.linalg.norm(u_vec)
        nv = np.linalg.norm(v_vec)
        nw = np.linalg.norm(w_vec)
        u_vec = u_vec / nu
        v_vec = v_vec / nv
        w_vec = w_vec / nw
        
        # Update residual
        T3 = T3 - scale * np.outer(w_vec, u_vec)
        
        terms.append((scale, u_vec, v_vec, w_vec))
        
    return terms


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0

    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    min_rank = 21
    max_rank = min(best_rank - 1, 41)

    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for each rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            try:
                terms = _peel_random(n, m, p, r, seed=attempt)
                if len(terms) < r:
                    continue
                
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                
                for c in range(r):
                    a, u, v, w = terms[c]
                    U[:, c] = np.round(2.0 * u) / 2.0
                    V[:, c] = np.round(2.0 * v) / 2.0
                    W[:, c] = np.round(2.0 * w) / 2.0

                if _verify(U, V, W, n, m, p):
                    return {
                        "U": _to_list(U),
                        "V": _to_list(V),
                        "W": _to_list(W),
                    }
            except Exception:
                continue

    return best

# EVOLVE-BLOCK-END
