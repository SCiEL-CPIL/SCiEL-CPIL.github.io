import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin under 150s wall-clock
    
    n, m, p = 2, 5, 5
    
    def time_left():
        return deadline - time.time()
    
    def build_schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
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
    
    def compute_residual(U, V, W):
        R = U.shape[1]
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    res[idx] = np.dot(u_row, v_row * w_row) - 1.0
                    idx += 1
        return res
    
    def max_abs_residual(U, V, W):
        return float(np.max(np.abs(compute_residual(U, V, W))))
    
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
    
    def alternating(U, V, W, cycles=30):
        R = U.shape[1]
        for c in range(cycles):
            if time.time() > deadline - 8:
                break
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
        return U, V, W
    
    def gauss_newton(U, V, W, max_iter=200, tol=1e-10):
        R = U.shape[1]
        total_vars = U.size + V.size + W.size
        for it in range(max_iter):
            if time.time() > deadline - 6:
                break
            res = compute_residual(U, V, W)
            if np.max(np.abs(res)) < tol:
                break
            J = np.zeros((n*m*p, total_vars))
            idx = 0
            for i in range(n):
                for j in range(m):
                    u_idx = i*m+j
                    for k in range(p):
                        v_idx = j*p+k
                        w_idx = k*n+i
                        for r in range(R):
                            J[idx, u_idx*R + r] = V[v_idx, r] * W[w_idx, r]
                            J[idx, U.size + v_idx*R + r] = U[u_idx, r] * W[w_idx, r]
                            J[idx, U.size + V.size + w_idx*R + r] = U[u_idx, r] * V[v_idx, r]
                        idx += 1
            try:
                JtJ = J.T @ J
                JtJ += 1e-8 * np.eye(total_vars)
                delta = np.linalg.solve(JtJ, -J.T @ res)
            except np.linalg.LinAlgError:
                try:
                    delta = np.linalg.lstsq(JtJ, -J.T @ res, rcond=None)[0]
                except Exception:
                    break
            Uf = U.flatten() + delta[:U.size]
            Vf = V.flatten() + delta[U.size:U.size+V.size]
            Wf = W.flatten() + delta[U.size+V.size:]
            U = Uf.reshape(U.shape)
            V = Vf.reshape(V.shape)
            W = Wf.reshape(W.shape)
        return U, V, W
    
    def snap_to_rational(U, V, W, denom=2):
        """Snap values to multiples of 1/denom"""
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        return U_s, V_s, W_s
    
    def try_rank(R, seed=42, n_init=5):
        if time_left() < 15:
            return None
        best = None
        best_err = float('inf')
        for s in range(n_init):
            if time_left() < 10:
                break
            np.random.seed(seed + s)
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            for r in range(R):
                for M in (U, V, W):
                    mx = np.max(np.abs(M[:, r]))
                    if mx > 0:
                        M[:, r] /= mx
            U, V, W = alternating(U, V, W, cycles=40)
            U, V, W = gauss_newton(U, V, W, max_iter=150, tol=1e-10)
            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-8:
                for denom in [1, 2, 4, 8]:
                    Us, Vs, Ws = snap_to_rational(U, V, W, denom)
                    err_snapped = max_abs_residual(Us, Vs, Ws)
                    if err_snapped < 1e-6:
                        return Us, Vs, Ws
                return U, V, W
        if best is not None and best_err < 1e-6:
            for denom in [1, 2, 4, 8]:
                Us, Vs, Ws = snap_to_rational(best[0], best[1], best[2], denom)
                err_snapped = max_abs_residual(Us, Vs, Ws)
                if err_snapped < 1e-5:
                    return Us, Vs, Ws
            return best
        return None
    
    def try_winograd_structured():
        """
        Try to exploit the structure of 2x5x5 matrix multiplication.
        The key insight: since n=2, we can potentially use shared bilinear forms.
        
        For C = A*B where A is 2x5 and B is 5x5:
        C[0,k] = sum_j A[0,j]*B[j,k]
        C[1,k] = sum_j A[1,j]*B[j,k]
        
        We can try to find rank-R decompositions where the two output rows
        share some of the bilinear products.
        
        Strategy: Use the fact that the tensor has a specific structure.
        Try ranks from 25 up to 50.
        """
        if time_left() < 20:
            return None
        
        # Try structured initialization based on Winograd ideas
        # For 2x5x5, the minimal rank is conjectured to be around 25-35
        for R in [25, 26, 27, 28, 29, 30, 32, 35, 40, 45]:
            if time_left() < 10:
                break
            result = try_rank(R, seed=R*100, n_init=3)
            if result is not None:
                U, V, W = result
                err = max_abs_residual(U, V, W)
                if err < 1e-5:
                    return U, V, W
        return None
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = (U_school, V_school, W_school)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    if time_left() > 30:
        result = try_winograd_structured()
        if result is not None:
            U, V, W = result
            err = max_abs_residual(U, V, W)
            if err < 1e-4:
                best_result = (U, V, W)
                best_rank = U.shape[1]
    
    # Additional attempts with different strategies if time permits
    if time_left() > 20 and best_rank > 35:
        # Try more aggressive random searches at intermediate ranks
        for R in [30, 32, 35]:
            if time_left() < 10:
                break
            result = try_rank(R, seed=R*200, n_init=4)
            if result is not None:
                U, V, W = result
                err = max_abs_residual(U, V, W)
                if err < 1e-5 and R < best_rank:
                    best_result = (U, V, W)
                    best_rank = R
    
    U, V, W = best_result
    
    # Convert to integer/half-integer format
    result = {}
    for key, mat in [('U', U), ('V', V), ('W', W)]:
        new_mat = []
        for row in mat:
            new_row = []
            for val in row:
                # Try to represent as integer or half-integer
                val_float = float(val)
                # Check if it's close to an integer
                int_val = round(val_float)
                if abs(val_float - int_val) < 1e-9:
                    new_row.append(int(int_val))
                else:
                    # Check if it's close to a half-integer
                    half_val = round(val_float * 2) / 2.0
                    if abs(val_float - half_val) < 1e-9:
                        if half_val == int(half_val):
                            new_row.append(int(half_val))
                        else:
                            new_row.append(half_val)
                    else:
                        # Use fraction for exactness
                        frac = Fraction(val_float).limit_denominator(1000)
                        if frac.denominator == 1:
                            new_row.append(int(frac.numerator))
                        else:
                            new_row.append(float(frac))
            new_mat.append(new_row)
        result[key] = new_mat
    
    return result
