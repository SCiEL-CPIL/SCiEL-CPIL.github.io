import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 115.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen decompositions. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    r2 = 7
    r = r2 * r2  # 49
    
    U2_3d = U2.reshape(2, 2, r2)
    V2_3d = V2.reshape(2, 2, r2)
    W2_3d = W2.reshape(2, 2, r2)
    
    U4_5d = np.einsum('abk,cdl->acbdkl', U2_3d, U2_3d)
    U4 = U4_5d.reshape(16, 49).astype(np.int16)
    V4 = np.einsum('abk,cdl->acbdkl', V2_3d, V2_3d).reshape(16, 49).astype(np.int16)
    W4 = np.einsum('abk,cdl->acbdkl', W2_3d, W2_3d).reshape(16, 49).astype(np.int16)
    
    return U4, V4, W4

def _build_4x4x8():
    """Build 4x4x8 via two independent 4x4x4 Strassen-Kronecker blocks. Rank 98."""
    U4, V4, W4 = _build_4x4x4_strassen()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """Verify the tensor decomposition on random test matrices."""
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    rank = U.shape[1]
    C_comp = np.zeros((n, p), dtype=np.float64)
    for r in range(rank):
        Ar = A * U[:, r].reshape(n, m)
        Br = B * V[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W[:, r].reshape(p, n).T
        C_comp += Mr * Wr
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err

def _residual_frobenius(params, n, m, p, target_rank):
    """Compute Frobenius norm of residual tensor for given factor matrices."""
    U = params[:n*m*target_rank].reshape(n*m, target_rank)
    V = params[n*m*target_rank:(n*m + m*p)*target_rank].reshape(m*p, target_rank)
    W = params[(n*m + m*p)*target_rank:].reshape(p*n, target_rank)
    
    # Build tensor: T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if (i,j,k) corresponds to identity
    # The multiplication tensor has T[i,j,k] = delta[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is: T[i,j,k] = 1 if the (i,k) output uses input (i,j) and (j,k)
    # For matrix mult: T[i,j,k] = 1 always (it's the standard multiplication tensor)
    
    # Actually, the tensor for matrix multiplication A(4x4) * B(4x8) -> C(4x8) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the 3-tensor T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Compute residual efficiently
    residual = 0.0
    for r in range(target_rank):
        Ur = U[:, r]  # shape (n*m,)
        Vr = V[:, r]  # shape (m*p,)
        Wr = W[:, r]  # shape (p*n,)
        
        # Ur[i*m+j], Vr[j*p+k], Wr[k*n+i]
        # Residual contribution: sum over i,j,k of (Ur[i*m+j]*Vr[j*p+k]*Wr[k*n+i] - 1)^2
        # But we need the total sum over r of each factor product = 1
        # So residual = sum_{i,j,k} (sum_r Ur_r[i*m+j]*Vr_r[j*p+k]*Wr_r[k*n+i] - 1)^2
        
    # More efficient: compute the full tensor
    T_comp = np.zeros((n, m, p), dtype=np.float64)
    for r in range(target_rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_comp[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
    
    T_target = np.ones((n, m, p), dtype=np.float64)
    residual = np.sum((T_comp - T_target)**2)
    return residual

def _residual_and_grad(params, n, m, p, target_rank):
    """Compute residual and gradient for L-BFGS-B."""
    U = params[:n*m*target_rank].reshape(n*m, target_rank)
    V = params[n*m*target_rank:(n*m + m*p)*target_rank].reshape(m*p, target_rank)
    W = params[(n*m + m*p)*target_rank:].reshape(p*n, target_rank)
    
    T_comp = np.zeros((n, m, p), dtype=np.float64)
    for r in range(target_rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_comp[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
    
    T_target = np.ones((n, m, p), dtype=np.float64)
    residual = np.sum((T_comp - T_target)**2)
    
    # Gradient: d/dU[i*m+j, r] = 2 * sum_{i,j,k} (T_comp[i,j,k] - 1) * Vr[j*p+k, r] * Wr[k*n+i, r]
    grad = np.zeros_like(params)
    diff = T_comp - T_target  # (n, m, p)
    
    for r in range(target_rank):
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        
        # Gradient w.r.t. U[i*m+j, r]
        for i in range(n):
            for j in range(m):
                # sum_k diff[i,j,k] * Vr[j,k] * Wr[k,i]
                g = 0.0
                for k in range(p):
                    g += diff[i,j,k] * Vr[j,k] * Wr[k,i]
                grad[i*m+j, r] = 2.0 * g
        
        # Gradient w.r.t. V[j*p+k, r]
        Ur = U[:, r].reshape(n, m)
        for j in range(m):
            for k in range(p):
                g = 0.0
                for i in range(n):
                    g += diff[i,j,k] * Ur[i,j] * Wr[k,i]
                grad[n*m*target_rank + j*p+k, r] = 2.0 * g
        
        # Gradient w.r.t. W[k*n+i, r]
        for k in range(p):
            for i in range(n):
                g = 0.0
                for j in range(m):
                    g += diff[i,j,k] * Ur[i,j] * Vr[j,k]
                grad[(n*m+m*p)*target_rank + k*n+i, r] = 2.0 * g
    
    return residual, grad

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50, time_budget=80.0):
    """Try to reduce rank by zeroing out columns with small norm, then re-optimizing."""
    start_time = time.time()
    current_U, current_V, current_W = U.astype(np.float64).copy(), V.astype(np.float64).copy(), W.astype(np.float64).copy()
    current_rank = U.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        if current_rank <= best_rank:
            break
            
        # Find columns with smallest combined norm
        col_norms = np.zeros(current_rank)
        for r in range(current_rank):
            col_norms[r] = np.linalg.norm(current_U[:, r]) + np.linalg.norm(current_V[:, r]) + np.linalg.norm(current_W[:, r])
        
        # Try zeroing the column with smallest norm
        min_col = np.argmin(col_norms)
        
        # Check if zeroing this column still gives good residual
        test_U = current_U.copy()
        test_V = current_V.copy()
        test_W = current_W.copy()
        test_U[:, min_col] = 0
        test_V[:, min_col] = 0
        test_W[:, min_col] = 0
        
        # Compute residual
        T_comp = np.zeros((n, m, p), dtype=np.float64)
        for r in range(current_rank):
            if r == min_col:
                continue
            Ur = test_U[:, r].reshape(n, m)
            Vr = test_V[:, r].reshape(m, p)
            Wr = test_W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_comp[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        
        T_target = np.ones((n, m, p), dtype=np.float64)
        residual = np.sum((T_comp - T_target)**2)
        
        if residual < 1e-6:
            # Successfully removed a column
            mask = np.arange(current_rank) != min_col
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 1
            best_rank = current_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        else:
            # Try to optimize the remaining factors to compensate
            # Use L-BFGS-B on the non-zeroed columns
            params = np.concatenate([current_U.ravel(), current_V.ravel(), current_W.ravel()])
            
            def objective(params):
                U_p = params[:n*m*current_rank].reshape(n*m, current_rank)
                V_p = params[n*m*current_rank:(n*m+m*p)*current_rank].reshape(m*p, current_rank)
                W_p = params[(n*m+m*p)*current_rank:].reshape(p*n, current_rank)
                T_c = np.zeros((n, m, p), dtype=np.float64)
                for r in range(current_rank):
                    Ur = U_p[:, r].reshape(n, m)
                    Vr = V_p[:, r].reshape(m, p)
                    Wr = W_p[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_c[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                T_t = np.ones((n, m, p), dtype=np.float64)
                return np.sum((T_c - T_t)**2)
            
            # Try optimization with limited iterations
            try:
                result = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 10, 'gtol': 1e-4})
                if result.fun < 1e-6:
                    opt_params = result.x
                    opt_U = opt_params[:n*m*current_rank].reshape(n*m, current_rank)
                    opt_V = opt_params[n*m*current_rank:(n*m+m*p)*current_rank].reshape(m*p, current_rank)
                    opt_W = opt_params[(n*m+m*p)*current_rank:].reshape(p*n, current_rank)
                    # Check if we can now zero the min_col
                    test_U2 = opt_U.copy()
                    test_V2 = opt_V.copy()
                    test_W2 = opt_W.copy()
                    test_U2[:, min_col] = 0
                    test_V2[:, min_col] = 0
                    test_W2[:, min_col] = 0
                    T_c2 = np.zeros((n, m, p), dtype=np.float64)
                    for r in range(current_rank):
                        if r == min_col:
                            continue
                        Ur = test_U2[:, r].reshape(n, m)
                        Vr = test_V2[:, r].reshape(m, p)
                        Wr = test_W2[:, r].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_c2[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                    T_t2 = np.ones((n, m, p), dtype=np.float64)
                    res2 = np.sum((T_c2 - T_t2)**2)
                    if res2 < 1e-6:
                        mask2 = np.arange(current_rank) != min_col
                        current_U = opt_U[:, mask2]
                        current_V = opt_V[:, mask2]
                        current_W = opt_W[:, mask2]
                        current_rank -= 1
                        best_rank = current_rank
                        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            except Exception:
                pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    n, m, p = 4, 4, 8
    U, V, W = _build_4x4x8()
    
    ok, err = _verify(U, V, W)
    if not ok:
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=np.int16)
        V_f = np.zeros((m * p, rank), dtype=np.int16)
        W_f = np.zeros((p * n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Try to reduce rank using optimization
    time_budget = _time_left() - 10.0  # Leave 10s safety margin
    if time_budget > 5.0:
        U_opt, V_opt, W_opt, reduced_rank = _try_reduce_rank(U, V, W, n, m, p, time_budget=time_budget)
        if reduced_rank < U.shape[1]:
            # Verify the reduced decomposition
            U_int = np.round(U_opt).astype(np.int16)
            V_int = np.round(V_opt).astype(np.int16)
            W_int = np.round(W_opt).astype(np.int16)
            ok2, err2 = _verify(U_int, V_int, W_int)
            if ok2:
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
