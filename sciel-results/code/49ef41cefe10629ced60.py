"""Fresh design: rank-reduction search from schoolbook with smoothed LS surrogate."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def _to_tensor(U, V, W):
    """Build the full (n*m, m*p, p*n) tensor from factor matrices."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return T


def _verify(U, V, W, tol=1e-9):
    """Check exact decomposition: T == sum_r U_r ⊗ V_r ⊗ W_r and matches schoolbook."""
    n, m, p = DIMENSIONS
    T = _to_tensor(U, V, W)
    # Build target (schoolbook)
    tgt = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T, tgt, atol=tol)


def _solve_column(U, V, T, r):
    """Given U, V fixed, solve for W[:, r] to best match T.
    T = sum_{r'} U_{r'} ⊗ V_{r'} ⊗ W_{r'}. Fix all W except column r, solve LS for W_r."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Residual after removing column r's contribution (unknown W_r)
    T_other = np.zeros_like(T)
    for rp in range(rank):
        if rp == r:
            continue
        T_other += np.einsum('a,b,c->abc', U[:, rp], V[:, rp], W[:, rp])
    residual = T - T_other
    # residual = U_r ⊗ V_r ⊗ W_r
    # Reshape: for each (a,b), residual[a,b,:] = U[a,r]*V[b,r]*W[:,r]
    # So W[:,r] = residual[a,b,:] / (U[a,r]*V[b,r]) — but this is overdetermined.
    # Use LS: vectorize. residual.reshape(-1, p*n) = (U_r ⊗ V_r) outer ... 
    # Actually residual[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    # Let u = U[:,r], v = V[:,r], w = W[:,r]
    # residual = outer(outer(u,v), w)  -- i.e., residual[a,b,c] = u[a]*v[b]*w[c]
    # This is a rank-1 tensor in (a*b) x c space: residual.reshape(n*m, p*n) = (u⊗v) * w^T
    # So we need (u⊗v) * w^T = residual.reshape(n*m, p*n)
    # LS solve: minimize ||(u⊗v) w^T - R|| where R = residual.reshape(n*m, p*n)
    uv = np.kron(U[:, r], V[:, r])  # length n*m * m*p? No, u is n*m, v is m*p
    # u length n*m, v length m*p -> uv length (n*m)*(m*p) which is too big
    # Actually residual is (n*m, m*p, p*n). Reshape to ((n*m)*(m*p), p*n)
    R = residual.reshape((n * m) * (m * p), p * n)
    uv = np.kron(U[:, r], V[:, r])  # length (n*m)*(m*p)
    # Solve: uv * w^T = R  =>  each row i: uv[i]*w = R[i,:]
    # LS: minimize sum_i ||uv[i]*w - R[i,:]||^2
    # This is: (uv^T uv) w = uv^T R  (vector form, w is p*n)
    # More precisely: for each component c: minimize sum_i (uv[i]*w[c] - R[i,c])^2
    # => uv^T uv * w[c] = uv^T R[:,c]
    A = np.dot(uv, uv)
    if abs(A) < 1e-15:
        return None
    w = np.zeros(p * n)
    for c in range(p * n):
        w[c] = np.dot(uv, R[:, c]) / A
    return w


def _solve_two_columns(U, V, W, r1, r2):
    """Solve for W[:, r1] and W[:, r2] jointly given U, V fixed."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = _to_tensor(U, V, W)
    T_other = np.zeros_like(T)
    for rp in range(rank):
        if rp == r1 or rp == r2:
            continue
        T_other += np.einsum('a,b,c->abc', U[:, rp], V[:, rp], W[:, rp])
    residual = T - T_other
    R = residual.reshape((n * m) * (m * p), p * n)
    uv1 = np.kron(U[:, r1], V[:, r1])
    uv2 = np.kron(U[:, r2], V[:, r2])
    # Solve [uv1, uv2] * [w1, w2] = R  (LS)
    A = np.column_stack([uv1, uv2])  # shape ((n*m)*(m*p), 2)
    # For each column c of R: solve A @ [w1[c], w2[c]] = R[:, c]
    w1 = np.zeros(p * n)
    w2 = np.zeros(p * n)
    for c in range(p * n):
        sol, *_ = np.linalg.lstsq(A, R[:, c], rcond=None)
        w1[c] = sol[0]
        w2[c] = sol[1]
    return w1, w2


def _snap(x, tol=1e-7):
    """Snap values close to half-integers to exact half-integers."""
    y = np.array(x, dtype=float)
    snapped = np.round(y * 2) / 2
    if np.allclose(y, snapped, atol=tol):
        return snapped
    # Try integers
    snapped2 = np.round(y)
    if np.allclose(y, snapped2, atol=tol):
        return snapped2
    return y


def construct():
    n, m, p = DIMENSIONS
    rank0 = n * m * p  # 72
    # Build schoolbook
    U = np.zeros((n * m, rank0))
    V = np.zeros((m * p, rank0))
    W = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    best_rank = rank0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Greedy rank reduction: try removing columns one at a time
    current_rank = rank0
    # Work with active column indices
    active = list(range(rank0))

    while current_rank > 1 and time.time() < _DEADLINE:
        # For each active column, try removing it and solving for the best replacement
        # Strategy: remove column c, then re-solve all columns to minimize error
        # Simpler: try removing each column, then do a few LS refinement passes
        best_remove = None
        best_err = np.inf

        for idx in range(len(active)):
            c = active[idx]
            # Remove column c
            new_active = active[:idx] + active[idx + 1:]
            if not new_active:
                break
            nr = len(new_active)
            # Extract current factors for active columns
            Ua = U[:, new_active]
            Va = V[:, new_active]
            Wa = W[:, new_active]

            # Refinement: alternate solving for W (all columns) given U,V, then U given V,W, then V given U,W
            for sweep in range(3):
                # Solve W given U, V
                for r in range(nr):
                    w_new = _solve_column(Ua, Va, T, r)
                    if w_new is not None:
                        Wa[:, r] = w_new
                # Solve U given V, W: T[a,b,c] = U[a,r]*V[b,r]*W[c,r]
                # For each r: U[:,r] * (V[:,r] ⊗ W[:,r]) = T[:, :, r] reshaped
                for r in range(nr):
                    # T[a,b,c] for fixed r: contribution is U[a,r]*V[b,r]*W[c,r]
                    # Residual approach: remove column r, solve
                    T_other = np.zeros_like(T)
                    for rp in range(nr):
                        if rp == r:
                            continue
                        T_other += np.einsum('a,b,c->abc', Ua[:, rp], Va[:, rp], Wa[:, rp])
                    residual = T - T_other
                    R = residual.reshape((m * p) * (p * n), n * m)
                    vw = np.kron(Va[:, r], Wa[:, r])
                    A = np.dot(vw, vw)
                    if abs(A) < 1e-15:
                        continue
                    u = np.zeros(n * m)
                    for c2 in range(n * m):
                        u[c2] = np.dot(vw, R[:, c2]) / A
                    Ua[:, r] = u
                # Solve V given U, W
                for r in range(nr):
                    T_other = np.zeros_like(T)
                    for rp in range(nr):
                        if rp == r:
                            continue
                        T_other += np.einsum('a,b,c->abc', Ua[:, rp], Va[:, rp], Wa[:, rp])
                    residual = T - T_other
                    R = residual.reshape((n * m) * (p * n), m * p)
                    uw = np.kron(Ua[:, r], Wa[:, r])
                    A = np.dot(uw, uw)
                    if abs(A) < 1e-15:
                        continue
                    v = np.zeros(m * p)
                    for c2 in range(m * p):
                        v[c2] = np.dot(uw, R[:, c2]) / A
                    Va[:, r] = v

            # Compute error
            T_new = _to_tensor(Ua, Va, Wa)
            err = np.linalg.norm(T_new - T)
            if err < best_err:
                best_err = err
                best_remove = (idx, c, Ua.copy(), Va.copy(), Wa.copy())

            if time.time() > _DEADLINE - 20:
                break

        if best_remove is None:
            break

        idx, c, Ua, Va, Wa = best_remove
        # Check if error is small enough to attempt exact recovery
        if best_err < 1e-6:
            # Try to snap to half-integers and verify
            Ua_s = _snap(Ua)
            Va_s = _snap(Va)
            Wa_s = _snap(Wa)
            if _verify(Ua_s, Va_s, Wa_s, tol=1e-8):
                # Success! Update
                current_rank -= 1
                best_rank = current_rank
                # Rebuild full matrices
                new_active = active[:idx] + active[idx + 1:]
                U_new = np.zeros((n * m, current_rank))
                V_new = np.zeros((m * p, current_rank))
                W_new = np.zeros((p * n, current_rank))
                for ni, col_idx in enumerate(new_active):
                    U_new[:, ni] = Ua_s[:, ni]
                    V_new[:, ni] = Va_s[:, ni]
                    W_new[:, ni] = Wa_s[:, ni]
                U, V, W = U_new, V_new, W_new
                active = list(range(current_rank))
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                # Can't remove this column exactly; try a different strategy
                # Try solving two columns jointly for better fit
                break
        else:
            break

    # Final: return best verified decomposition
    # Ensure best is verified
    if not _verify(best_U, best_V, best_W, tol=1e-8):
        # Fall back to schoolbook
        U = np.zeros((n * m, rank0))
        V = np.zeros((m * p, rank0))
        W = np.zeros((p * n, rank0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
