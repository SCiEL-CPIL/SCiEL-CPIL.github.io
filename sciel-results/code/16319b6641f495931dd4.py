"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 142.0  # safety margin under 150s wall-clock

    n, m, p = 2, 5, 5
    # Tensor convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)

    def build_schoolbook(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    # Precompute index arrays for vectorized operations
    # For each (i,j,k) triple, store the indices into U, V, W rows
    ijk_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ijk_indices.append((i*m+j, j*p+k, k*n+i))
    u_idx = np.array([x[0] for x in ijk_indices])  # (nmp,)
    v_idx = np.array([x[1] for x in ijk_indices])  # (nmp,)
    w_idx = np.array([x[2] for x in ijk_indices])  # (nmp,)
    nmp = n*m*p

    def compute_residual(U, V, W):
        # residual for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        # Vectorized: for each (i,j,k), dot(U[u_idx], V[v_idx]*W[w_idx]) - 1
        # U[u_idx] has shape (nmp, R), V[v_idx]*W[w_idx] has shape (nmp, R)
        prod = V[v_idx] * W[w_idx]  # (nmp, R)
        res = np.sum(U[u_idx] * prod, axis=1) - 1.0
        return res

    def max_abs_residual(U, V, W):
        return float(np.max(np.abs(compute_residual(U, V, W))))

    # Precompute index maps for fast linear solves
    # For U: variable U[i*m+j, r]. Constraint (i,j,k): sum_r U[i*m+j,r]*(V[j*p+k,r]*W[k*n+i,r]) = 1
    # For V: variable V[j*p+k, r]. Constraint (i,j,k): sum_r V[j*p+k,r]*(U[i*m+j,r]*W[k*n+i,r]) = 1
    # For W: variable W[k*n+i, r]. Constraint (i,j,k): sum_r W[k*n+i,r]*(U[i*m+j,r]*V[j*p+k,r]) = 1

    def solve_for_U(V, W, R):
        U = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                A_block = np.zeros((p, R))
                b_block = np.ones(p)
                for k in range(p):
                    A_block[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u_row = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                    U[i*m+j, :] = u_row
                except np.linalg.LinAlgError:
                    U[i*m+j, :] = 0
        return U

    def solve_for_V(U, W, R):
        V = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A_block = np.zeros((n, R))
                b_block = np.ones(n)
                for i in range(n):
                    A_block[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v_row = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                    V[j*p+k, :] = v_row
                except np.linalg.LinAlgError:
                    V[j*p+k, :] = 0
        return V

    def solve_for_W(U, V, R):
        W = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                A_block = np.zeros((m, R))
                b_block = np.ones(m)
                for j in range(m):
                    A_block[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w_row = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                    W[k*n+i, :] = w_row
                except np.linalg.LinAlgError:
                    W[k*n+i, :] = 0
        return W

    def gauss_newton(U, V, W, max_iter=400, tol=1e-11):
        R = U.shape[1]
        total_vars = U.size + V.size + W.size
        U_flat = U.flatten().copy()
        V_flat = V.flatten().copy()
        W_flat = W.flatten().copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 4:
                break
            U_cur = U_flat.reshape(U.shape)
            V_cur = V_flat.reshape(V.shape)
            W_cur = W_flat.reshape(W.shape)
            res = compute_residual(U_cur, V_cur, W_cur)
            max_res = np.max(np.abs(res))
            if max_res < tol:
                break
            
            # Build Jacobian vectorized
            # J shape: (nmp, total_vars)
            # For each (i,j,k) at index idx:
            #   d/dU[u_idx, r] = V[v_idx, r] * W[w_idx, r]
            #   d/dV[v_idx, r] = U[u_idx, r] * W[w_idx, r]
            #   d/dW[w_idx, r] = U[u_idx, r] * V[v_idx, r]
            U_sel = U_cur[u_idx]  # (nmp, R)
            V_sel = V_cur[v_idx]  # (nmp, R)
            W_sel = W_cur[w_idx]  # (nmp, R)
            
            J_U = V_sel * W_sel  # (nmp, R)
            J_V = U_sel * W_sel  # (nmp, R)
            J_W = U_sel * V_sel  # (nmp, R)
            
            # Build full J by scattering
            J = np.zeros((nmp, total_vars))
            for r in range(R):
                J[:, u_idx*R + r] = J_U[:, r]
                J[:, U.size + v_idx*R + r] = J_V[:, r]
                J[:, U.size + V.size + w_idx*R + r] = J_W[:, r]
            
            try:
                JtJ = J.T @ J
                JtJ += 1e-8 * np.eye(total_vars)
                rhs = -J.T @ res
                delta = np.linalg.solve(JtJ, rhs)
            except np.linalg.LinAlgError:
                try:
                    delta = np.linalg.lstsq(JtJ, -J.T @ res, rcond=None)[0]
                except Exception:
                    break
            
            # Line search
            best_step = 1.0
            best_obj = np.sum(res**2)
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
                U_test = U_flat + alpha * delta[:U.size]
                V_test = V_flat + alpha * delta[U.size:U.size+V.size]
                W_test = W_flat + alpha * delta[U.size+V.size:]
                U_t = U_test.reshape(U.shape)
                V_t = V_test.reshape(V.shape)
                W_t = W_test.reshape(W.shape)
                res_t = compute_residual(U_t, V_t, W_t)
                obj_t = np.sum(res_t**2)
                if obj_t < best_obj:
                    best_obj = obj_t
                    best_step = alpha
            
            U_flat = U_flat + best_step * delta[:U.size]
            V_flat = V_flat + best_step * delta[U.size:U.size+V.size]
            W_flat = W_flat + best_step * delta[U.size+V.size:]
        return U_flat.reshape(U.shape), V_flat.reshape(V.shape), W_flat.reshape(W.shape)

    def alternating(U, V, W, cycles=40):
        R = U.shape[1]
        for c in range(cycles):
            if time.time() > deadline - 6:
                break
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
        return U, V, W

    def snap_to_half_integer(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def try_rank(R, seed=42):
        if time.time() > deadline - 12:
            return None
        np.random.seed(seed)
        best = None
        best_err = float('inf')
        num_inits = 12
        for init_iter in range(num_inits):
            if time.time() > deadline - 6:
                break
            # Mixed initialization strategies
            if init_iter % 3 == 0:
                U = np.random.randn(n*m, R) * 0.5
                V = np.random.randn(m*p, R) * 0.5
                W = np.random.randn(p*n, R) * 0.5
            elif init_iter % 3 == 1:
                # Sparse-ish initialization
                U = np.random.randn(n*m, R) * 0.3
                V = np.random.randn(m*p, R) * 0.3
                W = np.random.randn(p*n, R) * 0.3
                # Add some structure: make first few columns more active
                for r in range(min(3, R)):
                    U[:, r] += 0.5
                    V[:, r] += 0.5
                    W[:, r] += 0.5
            else:
                # Binary-ish initialization
                U = (np.random.rand(n*m, R) > 0.5).astype(float) * 0.8
                V = (np.random.rand(m*p, R) > 0.5).astype(float) * 0.8
                W = (np.random.rand(p*n, R) > 0.5).astype(float) * 0.8
            
            # normalize columns to avoid degenerate scales
            for r in range(R):
                for M in (U, V, W):
                    mx = np.max(np.abs(M[:, r]))
                    if mx > 0:
                        M[:, r] /= mx
            U, V, W = alternating(U, V, W, cycles=40)
            U, V, W = gauss_newton(U, V, W, max_iter=300, tol=1e-10)
            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-9:
                # Try to snap
                Us, Vs, Ws = snap_to_half_integer(U, V, W)
                err_snapped = max_abs_residual(Us, Vs, Ws)
                if err_snapped < 1e-7:
                    return Us, Vs, Ws
                else:
                    return U, V, W
        if best is not None and best_err < 1e-7:
            Us, Vs, Ws = snap_to_half_integer(best[0], best[1], best[2])
            err_snapped = max_abs_residual(Us, Vs, Ws)
            if err_snapped < 1e-7:
                return Us, Vs, Ws
            else:
                return best
        return None

    # Start with schoolbook
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)

    # Try ranks downward from a reasonable bound
    # Theoretical lower bound for 2x5x5 is around 30-35
    for R in range(45, 28, -1):
        if time.time() > deadline - 20:
            break
        found = False
        num_seeds = 15
        for seed in range(num_seeds):
            if time.time() > deadline - 12:
                break
            result = try_rank(R, seed=seed*17 + 5)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            # Try to go even lower
            continue
    
    # Final polish
    if best_rank < n * m * p and time.time() < deadline - 5:
        U_best, V_best, W_best = gauss_newton(U_best, V_best, W_best, max_iter=500, tol=1e-12)
        # Try snapping after polish
        Us, Vs, Ws = snap_to_half_integer(U_best, V_best, W_best)
        err_snapped = max_abs_residual(Us, Vs, Ws)
        if err_snapped < 1e-6:
            U_best, V_best, W_best = Us, Vs, Ws

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
