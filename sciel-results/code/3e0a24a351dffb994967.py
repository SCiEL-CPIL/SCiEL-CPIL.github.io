# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0  # seconds safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _verify(U, V, W, tol=1e-7):
    """Check exact decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _rank24_decomposition():
    """
    Construct a rank-24 decomposition for 3x3 matrix multiplication.
    Uses a structured approach: partition into 2x2 + 1x1 blocks.
    
    A = [[A11, a1], [a2, a3]]  where A11 is 2x2, a1/a2 are 2x1, a3 is 1x1
    B = [[B11, b1], [b2, b3]]  where B11 is 2x2, b1/b2 are 2x1, b3 is 1x1
    
    C = A*B:
    C11 = A11*B11 + a1*b2   (2x2)
    C12 = A11*b1 + a1*b3    (2x1)
    C21 = a2*B11 + a3*b2    (1x2)
    C22 = a2*b1 + a3*b3     (1x1)
    
    Use Strassen (rank 7) for A11*B11 -> contributes to C11
    Schoolbook for all other terms:
    - a1*b2: 2*1*1 = 2
    - A11*b1: 2*2*1 = 4
    - a1*b3: 2*1*1 = 2
    - a2*B11: 1*2*2 = 4
    - a3*b2: 1*1*2 = 2
    - a2*b1: 1*2*1 = 2
    - a3*b3: 1*1*1 = 1
    Total: 7 + 2+4+2+4+2+2+1 = 24
    """
    n, m, p = 3, 3, 3
    nm, mp, pn = n*m, m*p, p*n
    rank = 24
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    # Strassen for A11*B11 (indices: A rows 0,1; cols 0,1; B rows 0,1; cols 0,1)
    # A11 = A[0:2, 0:2], B11 = B[0:2, 0:2], contributes to C[0:2, 0:2]
    
    # Strassen M1 = (A11+A22)(B11+B22) where in 2x2 context:
    # A[0,0]+A[1,1] and B[0,0]+B[1,1]
    # Contributes to C[0,0] and C[1,1]
    U[0, col] = 1.0; U[3, col] = 1.0  # A[0,0], A[1,1]
    V[0, col] = 1.0; V[3, col] = 1.0  # B[0,0], B[1,1]
    W[0, col] = 1.0; W[4, col] = 1.0  # C[0,0], C[1,1]
    col += 1
    
    # M2 = (A21+A22)B11 = (A[1,0]+A[1,1]) * B[0,0]
    # Contributes to C[1,0]
    U[3, col] = 1.0; U[4, col] = 1.0  # A[1,0], A[1,1]
    V[0, col] = 1.0                    # B[0,0]
    W[3, col] = 1.0                    # C[1,0]
    col += 1
    
    # M3 = A11(B21-B22) = A[0,0] * (B[1,0]-B[1,1])
    # Contributes to C[0,1] and C[1,1]
    U[0, col] = 1.0                    # A[0,0]
    V[2, col] = 1.0; V[3, col] = -1.0  # B[1,0], -B[1,1]
    W[1, col] = 1.0; W[4, col] = 1.0   # C[0,1], C[1,1]
    col += 1
    
    # M4 = A22(B12-B11) = A[1,1] * (B[0,1]-B[0,0])
    # Contributes to C[0,0] and C[1,0]
    U[3, col] = 1.0                    # A[1,1]
    V[1, col] = 1.0; V[0, col] = -1.0  # B[0,1], -B[0,0]
    W[0, col] = 1.0; W[3, col] = 1.0   # C[0,0], C[1,0]
    col += 1
    
    # M5 = (A11+A12)B22 = (A[0,0]+A[0,1]) * B[1,1]
    # Contributes to -C[0,0] and C[0,1]
    U[0, col] = 1.0; U[1, col] = 1.0  # A[0,0], A[0,1]
    V[3, col] = 1.0                    # B[1,1]
    W[0, col] = -1.0; W[1, col] = 1.0  # -C[0,0], C[0,1]
    col += 1
    
    # M6 = (A21-A11)(B11+B12) = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # Contributes to C[1,1]
    U[3, col] = 1.0; U[0, col] = -1.0  # A[1,0], -A[0,0]
    V[0, col] = 1.0; V[1, col] = 1.0   # B[0,0], B[0,1]
    W[4, col] = 1.0                    # C[1,1]
    col += 1
    
    # M7 = (A12-A22)(B21+B22) = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # Contributes to C[0,0]
    U[1, col] = 1.0; U[3, col] = -1.0  # A[0,1], -A[1,1]
    V[2, col] = 1.0; V[3, col] = 1.0   # B[1,0], B[1,1]
    W[0, col] = 1.0                    # C[0,0]
    col += 1
    
    # Now schoolbook terms for the cross-block products:
    # a1*b2: A[0,2]*B[2,0] -> C[0,0]; A[0,2]*B[2,1] -> C[0,1]
    # Wait, I need to be careful with indices.
    # A is 3x3, B is 3x3, C is 3x3
    # A[i,j] * B[j,k] -> C[i,k]
    
    # Let me redefine: 
    # Block A11 = A[0:2, 0:2], a1 = A[0:2, 2] (col 2), a2 = A[2, 0:2] (row 2), a3 = A[2,2]
    # Block B11 = B[0:2, 0:2], b1 = B[0:2, 2] (col 2), b2 = B[2, 0:2] (row 2), b3 = B[2,2]
    
    # C11 = A11*B11 + a1*b2:
    #   A11*B11 done via Strassen above
    #   a1*b2: A[0,2]*B[2,0] -> C[0,0]; A[1,2]*B[2,0] -> C[1,0]
    #          A[0,2]*B[2,1] -> C[0,1]; A[1,2]*B[2,1] -> C[1,1]
    # That's 4 terms, not 2. Let me recount.
    
    # Actually a1 is 2x1 (column), b2 is 1x2 (row), so a1*b2 is 2x2 = 4 terms.
    # Similarly for other cross terms.
    
    # Let me just do full schoolbook for all non-A11*B11 terms:
    # Total schoolbook terms: 27 - 4 = 23 (since A11*B11 has 4 terms replaced by 7 Strassen)
    # So total rank = 7 + 23 = 30. That's worse than 27!
    
    # I made an error. Strassen replaces 4 multiplications with 7, so it's worse for 2x2.
    # Strassen is only beneficial when you have larger blocks.
    
    # For 3x3, the known optimal is rank 23. Let me use a different approach.
    # Return schoolbook for now.
    return _schoolbook()


def _try_solve_for_W(U, V, rank):
    """Given U (nm x rank) and V (mp x rank), solve linear system for W (pn x rank)."""
    n, m, p = 3, 3, 3
    num_vars = (p * n) * rank
    A = np.zeros((n * m * p, num_vars))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    var_idx = (k * n + i) * rank + r
                    A[eq, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                eq += 1
    W_flat, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=None)
    W = W_flat.reshape((p * n, rank))
    return W


def _try_rank_fixed_UV(rank, seed=0):
    """Fix U,V as first `rank` schoolbook columns, solve for W, verify."""
    n, m, p = 3, 3, 3
    U_s, V_s, W_s = _schoolbook()
    if rank > U_s.shape[1]:
        return None
    U = U_s[:, :rank].copy()
    V = V_s[:, :rank].copy()
    W = _try_solve_for_W(U, V, rank)
    if _verify(U, V, W):
        return U, V, W
    return None


def _general_search(rank, seed=0):
    """
    General search allowing U,V,W to vary. Uses scipy least-squares on a nonlinear system,
    with a final exact verification.
    """
    n, m, p = 3, 3, 3
    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    nm, mp, pn = n * m, m * p, p * n

    def residuals(flat):
        U = flat[:nm * rank].reshape(nm, rank)
        V = flat[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = flat[nm * rank + mp * rank:].reshape(pn, rank)
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    for attempt in range(8):
        if _time_left() < 8.0:
            break
        # Initialize near schoolbook-like random
        U0 = rng.normal(size=(nm, rank))
        V0 = rng.normal(size=(mp, rank))
        W0 = rng.normal(size=(pn, rank))
        # Scale to make products ~O(1)
        scale = 1.0 / (rank ** 0.5)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        flat0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        try:
            sol = least_squares(residuals, flat0, method='lm', max_nfev=2000)
        except Exception:
            continue
        U = sol.x[:nm * rank].reshape(nm, rank)
        V = sol.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = sol.x[nm * rank + mp * rank:].reshape(pn, rank)
        if _verify(U, V, W, tol=1e-6):
            return U, V, W
    return None


def construct():
    """
    Find a rank-minimal exact decomposition for 3x3 matrix multiplication.
    Tries decreasing ranks; returns the best valid one found before deadline.
    """
    n, m, p = 3, 3, 3

    # Fallback: schoolbook rank 27 (always valid)
    U_s, V_s, W_s = _schoolbook()
    best = (U_s, V_s, W_s)  # rank 27

    # Try ranks from 23 down
    for rank in [23, 22, 21, 20, 19]:
        if _time_left() < 15.0:
            break
        res = _try_rank_fixed_UV(rank)
        if res is not None:
            U, V, W = res
            if rank < best[0].shape[1]:
                best = (U, V, W)
            continue
        if _time_left() < 20.0:
            break
        res2 = _general_search(rank)
        if res2 is not None:
            U, V, W = res2
            if rank < best[0].shape[1]:
                best = (U, V, W)
            continue

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# EVOLVE-BLOCK-END
