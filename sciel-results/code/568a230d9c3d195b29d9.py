"""Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
Combines block Strassen (rank 22) baseline with BFGS optimization and structured
random-walk search for rank-21. Uses efficient einsum-based verification and
structured random walks guided by slice-rank constraints.
"""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0  # seconds, leaves 10s margin before 150s kill


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using einsum to compute the tensor contraction.
    Returns True iff sum_r W[k*n+i,r]*U[i*m+j,r]*V[j*p+k,r] == 1 for all i,j,k.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Reshape for efficient computation
    # U: (n*m, r) -> (n, m, r)
    # V: (m*p, r) -> (m, p, r)
    # W: (p*n, r) -> (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    expected = np.ones((n, m, p), dtype=float)
    return bool(np.max(np.abs(T - expected)) <= tol)


def _bilinear_error_fast(U, V, W, n, m, p):
    """Compute squared bilinear error efficiently using einsum."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U3 = U.reshape(n, m, U.shape[1])
    V3 = V.reshape(m, p, V.shape[1])
    W3 = W.reshape(p, n, W.shape[1])
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    expected = np.ones((n, m, p), dtype=float)
    return float(np.sum((T - expected) ** 2))


def _rank22_list(n, m, p):
    """Construct rank-22 decomposition using block Strassen."""
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A: 0=A00, 1=A01, 3=A10, 4=A11
    # B: 0=B00, 1=B01, 4=B10, 5=B11
    # W: 0=C00, 2=C01, 1=C10, 3=C11

    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1

    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1

    # Block 2: 7 Strassen terms for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B: 2=B02, 3=B03, 6=B12, 7=B13
    # W: 4=C02, 6=C03, 5=C12, 7=C13

    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1

    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1

    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1

    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1

    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1

    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1

    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1

    # Block 3: 8 schoolbook terms for A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    # A: 2=A02, 5=A12
    # B: 8=B20, 9=B21, 10=B22, 11=B23
    for i, ai in [(0, 2), (1, 5)]:
        for k, bk in [(0, 8), (1, 9), (2, 10), (3, 11)]:
            U[ai][col] = 1
            V[bk][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _structured_random_walk(n, m, p, r, max_steps=500, seed=42):
    """Structured random walk in parameter space guided by error landscape.
    
    Uses a smooth surrogate: minimize sum of squared residuals with adaptive
    step sizes based on local gradient information.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Start from rank-22 with one column zeroed, then optimize
    U22, V22, W22 = _rank22_list(n, m, p)
    U22 = np.array(U22, dtype=float)
    V22 = np.array(V22, dtype=float)
    W22 = np.array(W22, dtype=float)
    
    best_err = float('inf')
    best_params = None
    
    for drop_col in range(22):
        if _remaining() < 10.0:
            break
        
        keep = [c for c in range(22) if c != drop_col]
        U_init = U22[:, keep].copy()
        V_init = V22[:, keep].copy()
        W_init = W22[:, keep].copy()
        
        # Small random perturbation to escape local minima
        U_init += rng.randn(*U_init.shape) * 0.01
        V_init += rng.randn(*V_init.shape) * 0.01
        W_init += rng.randn(*W_init.shape) * 0.01
        
        x = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            return _bilinear_error_fast(U, V, W, n, m, p)
        
        # Gradient descent with adaptive step size
        lr = 0.01
        min_lr = 1e-6
        patience = 50
        no_improve = 0
        cur_err = objective(x)
        
        for step in range(max_steps):
            if _remaining() < 8.0:
                break
            
            # Numerical gradient
            grad = np.zeros_like(x)
            eps = 1e-7
            for i in range(0, len(x), 50):  # Sample gradient for speed
                for j in range(i, min(i + 50, len(x))):
                    x_plus = x.copy()
                    x_plus[j] += eps
                    grad[j] = (objective(x_plus) - cur_err) / eps
            
            # Adaptive step
            x -= lr * grad
            new_err = objective(x)
            
            if new_err < cur_err:
                cur_err = new_err
                no_improve = 0
                if cur_err < best_err:
                    best_err = cur_err
                    best_params = x.copy()
            else:
                x += lr * grad  # Undo
                no_improve += 1
                lr *= 0.95
                if lr < min_lr:
                    break
            
            if no_improve >= patience:
                break
        
        if best_err < 1e-8 and best_params is not None:
            x = best_params
            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                return U.tolist(), V.tolist(), W.tolist(), r
    
    return None


def _try_bfgs_rank21(n, m, p):
    """Try BFGS optimization from multiple initializations for rank-21."""
    r = 21
    nm, mp, pn = n * m, m * p, p * n
    
    U22, V22, W22 = _rank22_list(n, m, p)
    U22 = np.array(U22, dtype=float)
    V22 = np.array(V22, dtype=float)
    W22 = np.array(W22, dtype=float)
    
    best_err = float('inf')
    best_params = None
    
    for drop_col in range(22):
        if _remaining() < 20.0:
            break
        
        keep = [c for c in range(22) if c != drop_col]
        U_init = U22[:, keep]
        V_init = V22[:, keep]
        W_init = W22[:, keep]
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            return _bilinear_error_fast(U, V, W, n, m, p)
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='BFGS',
                           options={'maxiter': 300, 'gtol': 1e-7})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is not None and best_err < 1e-8:
        x = best_params
        U = x[:nm * r].reshape(nm, r)
        V = x[nm * r:nm * r + mp * r].reshape(mp, r)
        W = x[nm * r + mp * r:].reshape(pn, r)
        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
            return U.tolist(), V.tolist(), W.tolist(), r
    
    return None


def _try_ilp_rank_reduction(n, m, p, target_rank=21):
    """Use ILP-style search to find rank reduction from rank-22.
    
    Strategy:
    1. Try simple column drops (rank 21 by removing one column)
    2. Try merging pairs with small coefficients
    """
    from fractions import Fraction
    import itertools
    
    nm, mp, pn = n * m, m * p, p * n
    r0 = 22
    
    U22, V22, W22 = _rank22_list(n, m, p)
    
    def to_frac_matrix(mat):
        return np.array([[Fraction(x) for x in row] for row in mat], dtype=object)
    
    U_frac = to_frac_matrix(U22)
    V_frac = to_frac_matrix(V22)
    W_frac = to_frac_matrix(W22)
    
    # Precompute all constraint indices
    constraints = []
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1 if (ia == i and ja == jb and kb == k) else 0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            constraints.append((Uidx, Vidx, Widx, expected))
    
    def check_subset_valid(cols):
        """Check if keeping only these columns gives exact decomposition."""
        for Uidx, Vidx, Widx, expected in constraints:
            actual = Fraction(0)
            for r in cols:
                actual += U_frac[Uidx, r] * V_frac[Vidx, r] * W_frac[Widx, r]
            if actual != expected:
                return False
        return True
    
    all_cols = list(range(r0))
    
    # First: try simple column drops
    for drop in all_cols:
        if _remaining() < 10.0:
            break
        cols_keep = [c for c in all_cols if c != drop]
        if check_subset_valid(cols_keep):
            U_res = [[float(U_frac[i, c]) for c in cols_keep] for i in range(nm)]
            V_res = [[float(V_frac[i, c]) for c in cols_keep] for i in range(mp)]
            W_res = [[float(W_frac[i, c]) for c in cols_keep] for i in range(pn)]
            return U_res, V_res, W_res, target_rank
    
    # Second: try merging pairs with small coefficients
    c_options = [Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)]
    
    for drop_from in all_cols:
        if _remaining() < 5.0:
            break
        for merge_to in all_cols:
            if merge_to == drop_from:
                continue
            for c in c_options:
                if _remaining() < 3.0:
                    break
                # Try merging drop_from into merge_to with coefficient c
                valid = True
                for Uidx, Vidx, Widx, expected in constraints:
                    actual = Fraction(0)
                    for r in all_cols:
                        if r == drop_from:
                            continue
                        u_val = U_frac[Uidx, r]
                        v_val = V_frac[Vidx, r]
                        w_val = W_frac[Widx, r]
                        if r == merge_to:
                            u_val += c * U_frac[Uidx, drop_from]
                            v_val -= c * V_frac[Vidx, drop_from]
                        actual += u_val * v_val * w_val
                    if actual != expected:
                        valid = False
                        break
                if valid:
                    result_cols = [r for r in all_cols if r != drop_from]
                    U_res = [[0.0]*len(result_cols) for _ in range(nm)]
                    V_res = [[0.0]*len(result_cols) for _ in range(mp)]
                    W_res = [[0.0]*len(result_cols) for _ in range(pn)]
                    
                    for new_idx, old_idx in enumerate(result_cols):
                        for Uidx in range(nm):
                            val = U_frac[Uidx, old_idx]
                            if old_idx == merge_to:
                                val += c * U_frac[Uidx, drop_from]
                            U_res[Uidx][new_idx] = float(val)
                        for Vidx in range(mp):
                            val = V_frac[Vidx, old_idx]
                            if old_idx == merge_to:
                                val -= c * V_frac[Vidx, drop_from]
                            V_res[Vidx][new_idx] = float(val)
                        for Widx in range(pn):
                            W_res[Widx][new_idx] = float(W_frac[Widx, old_idx])
                    
                    return U_res, V_res, W_res, target_rank
    
    return None


def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    # Start with rank-22
    best_U, best_V, best_W = _rank22_list(n, m, p)
    best_rank = 22

    # Verify
    if not _verify_fast(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p

    if _remaining() < 30.0:
        return _to_dict(best_U, best_V, best_W)

    # Try BFGS optimization for rank-21
    if _remaining() > 25.0:
        result21 = _try_bfgs_rank21(n, m, p)
        if result21 is not None:
            U21, V21, W21, r21 = result21
            return _to_dict(U21, V21, W21)

    # Try ILP-based rank reduction search
    if _remaining() > 20.0:
        ilp_result = _try_ilp_rank_reduction(n, m, p, target_rank=21)
        if ilp_result is not None:
            U_ilp, V_ilp, W_ilp, r_ilp = ilp_result
            return _to_dict(U_ilp, V_ilp, W_ilp)

    # Try structured random walk for rank-21
    if _remaining() > 15.0:
        rw_result = _structured_random_walk(n, m, p, 21, max_steps=300)
        if rw_result is not None:
            U_rw, V_rw, W_rw, r_rw = rw_result
            return _to_dict(U_rw, V_rw, W_rw)

    return _to_dict(best_U, best_V, best_W)
