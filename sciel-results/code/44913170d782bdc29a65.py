"""Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
Uses block Strassen (rank 22) as baseline, then searches for rank-21 via
BFGS optimization of the bilinear verification error.
"""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0  # seconds, leaves 10s margin before 150s kill


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Full bilinear verification: for all (ia, ja, jb, kb, i, k),
    sum_r W[k*n+i, r] * U[ia*m+ja, r] * V[jb*p+kb, r] == delta(ia,i)*delta(ja,jb)*delta(kb,k)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]

    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = np.dot(W[Widx, :], U[Uidx, :] * V[Vidx, :])
                            if abs(actual - expected) > tol:
                                return False
    return True


def _bilinear_error(U, V, W, n, m, p):
    """Compute the squared error of the bilinear verification.
    Error = sum over all (ia,ja,jb,kb,i,k) of (actual - expected)^2
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    total_err = 0.0
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = np.dot(W[Widx, :], U[Uidx, :] * V[Vidx, :])
                            total_err += (actual - expected) ** 2
    return total_err


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


def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}


def _gradient_descent_rank21(n, m, p, seed=42):
    """Custom gradient descent with momentum for rank-21 decomposition.
    Uses analytic gradient of the bilinear error."""
    import random
    rng = random.Random(seed)
    
    r = 21
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index arrays for efficiency
    indices = []
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            indices.append((Uidx, Vidx, Widx, expected))
    
    idx_arr = np.array([(u, v, w) for u, v, w, e in indices], dtype=int)
    exp_arr = np.array([e for u, v, w, e in indices], dtype=float)
    
    def compute_error_and_grad(U, V, W):
        """Compute error and gradients for U, V, W."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute all bilinear products
        # For each index (u,v,w): product = sum_r W[w,r]*U[u,r]*V[v,r]
        # We can vectorize this
        
        # Build matrices for fast computation
        # U_mat: (nm, r), V_mat: (mp, r), W_mat: (pn, r)
        
        # For each combination, compute dot(W[w,:], U[u,:]*V[v,:])
        # This is equivalent to: W[w,:] @ (U[u,:] * V[v,:])
        
        # Vectorized: for each (u,v,w) triple
        # U_u = U[idx_u, :]  # (r,)
        # V_v = V[idx_v, :]  # (r,)
        # W_w = W[idx_w, :]  # (r,)
        # product = W_w @ (U_u * V_v)
        
        U_idx = idx_arr[:, 0]
        V_idx = idx_arr[:, 1]
        W_idx = idx_arr[:, 2]
        
        U_sel = U[U_idx, :]  # (num_idx, r)
        V_sel = V[V_idx, :]  # (num_idx, r)
        W_sel = W[W_idx, :]  # (num_idx, r)
        
        products = np.sum(W_sel * U_sel * V_sel, axis=1)  # (num_idx,)
        
        errors = products - exp_arr
        total_err = np.sum(errors ** 2)
        
        # Gradients
        # d_err/dU[u,r] = 2 * sum over indices where U_idx==u of errors * W_sel * V_sel
        # We can compute this efficiently
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # For each index, the contribution to gradients
        # d/dU[u,r]: 2 * errors[i] * W_sel[i,r] * V_sel[i,r]
        # d/dV[v,r]: 2 * errors[i] * W_sel[i,r] * U_sel[i,r]
        # d/dW[w,r]: 2 * errors[i] * U_sel[i,r] * V_sel[i,r]
        
        contrib_U = 2.0 * errors[:, None] * W_sel * V_sel  # (num_idx, r)
        contrib_V = 2.0 * errors[:, None] * W_sel * U_sel  # (num_idx, r)
        contrib_W = 2.0 * errors[:, None] * U_sel * V_sel  # (num_idx, r)
        
        # Accumulate using np.add.at for repeated indices
        np.add.at(grad_U, U_idx, contrib_U)
        np.add.at(grad_V, V_idx, contrib_V)
        np.add.at(grad_W, W_idx, contrib_W)
        
        return total_err, grad_U, grad_V, grad_W
    
    # Initialize from rank-22 with one column dropped
    U22, V22, W22 = _rank22_list(n, m, p)
    U22 = np.array(U22, dtype=float)
    V22 = np.array(V22, dtype=float)
    W22 = np.array(W22, dtype=float)
    
    best_err = float('inf')
    best_result = None
    
    for drop_col in range(22):
        if _remaining() < 20.0:
            break
        
        keep = [c for c in range(22) if c != drop_col]
        U = U22[:, keep].copy()
        V = V22[:, keep].copy()
        W = W22[:, keep].copy()
        
        # Add small random perturbation
        U += rng.gauss(0, 0.01) * np.ones_like(U)
        V += rng.gauss(0, 0.01) * np.ones_like(V)
        W += rng.gauss(0, 0.01) * np.ones_like(W)
        
        # Gradient descent with momentum
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        err, gU, gV, gW = compute_error_and_grad(U, V, W)
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        
        for step in range(500):
            if _remaining() < 15.0:
                break
            
            # Update velocities
            vU = momentum * vU + (1 - momentum) * gU
            vV = momentum * vV + (1 - momentum) * gV
            vW = momentum * vW + (1 - momentum) * gW
            
            # Update parameters
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            err, gU, gV, gW = compute_error_and_grad(U, V, W)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            # Adaptive learning rate
            if err < 1e-10:
                break
            if step % 100 == 0 and step > 0:
                lr *= 0.5
        
        if best_err < 1e-8:
            break
    
    if best_result is not None and best_err < 1e-8:
        U, V, W = best_result
        if _verify_full(U, V, W, n, m, p, tol=1e-6):
            return U.tolist(), V.tolist(), W.tolist(), r
    return None


def _try_rank21_bfgs(n, m, p):
    """Try to find rank-21 decomposition using BFGS optimization."""
    r = 21
    nm, mp, pn = n * m, m * p, p * n
    
    U22, V22, W22 = _rank22_list(n, m, p)
    U22 = np.array(U22, dtype=float)
    V22 = np.array(V22, dtype=float)
    W22 = np.array(W22, dtype=float)
    
    best_err = float('inf')
    best_params = None
    
    for drop_col in range(22):
        if _remaining() < 25.0:
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
            return _bilinear_error(U, V, W, n, m, p)
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='BFGS',
                           options={'maxiter': 200, 'gtol': 1e-6})
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
        if _verify_full(U, V, W, n, m, p, tol=1e-6):
            return U.tolist(), V.tolist(), W.tolist(), r
    return None


def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    # Start with rank-22
    best_U, best_V, best_W = _rank22_list(n, m, p)
    best_rank = 22

    # Verify
    if not _verify_full(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p

    if _remaining() < 30.0:
        return _to_dict(best_U, best_V, best_W)

    # Try gradient descent for rank-21
    result21 = _gradient_descent_rank21(n, m, p, seed=42)
    if result21 is not None:
        U21, V21, W21, r21 = result21
        return _to_dict(U21, V21, W21)
    
    # Try BFGS for rank-21
    if _remaining() > 25.0:
        result21 = _try_rank21_bfgs(n, m, p)
        if result21 is not None:
            U21, V21, W21, r21 = result21
            return _to_dict(U21, V21, W21)
    
    # Try random restarts for rank-21
    if _remaining() > 20.0:
        def _random_rank21_attempt():
            r = 21
            x0 = np.random.randn(nm * r + mp * r + pn * r) * 0.1

            def objective(x):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error(U, V, W, n, m, p)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': 300, 'gtol': 1e-7})
                if res.fun < 1e-10:
                    x = res.x
                    U = x[:nm * r].reshape(nm, r)
                    V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                    W = x[nm * r + mp * r:].reshape(pn, r)
                    if _verify_full(U, V, W, n, m, p, tol=1e-6):
                        return U.tolist(), V.tolist(), W.tolist(), r
            except Exception:
                pass
            return None

        for _ in range(3):
            if _remaining() < 15.0:
                break
            res = _random_rank21_attempt()
            if res is not None:
                U21, V21, W21, r21 = res
                return _to_dict(U21, V21, W21)

    return _to_dict(best_U, best_V, best_W)
