import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
        
    def verify(U, V, W):
        # Check if U, V, W are half-integers
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        # Reconstruct tensor and compare
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i, j, k] = 1 for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def make_objective(r):
        u_size = n * m * r
        v_size = m * p * r
        
        def objective(x):
            U = x[:u_size].reshape(n * m, r)
            V = x[u_size:u_size+v_size].reshape(m * p, r)
            W = x[u_size+v_size:].reshape(p * n, r)
            
            # Compute T_approx[i, j, k]
            # Using einsum for efficiency
            # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
            # We can reshape U, V, W to (n, m, r), (m, p, r), (p, n, r)
            U_3d = U.reshape(n, m, r)
            V_3d = V.reshape(m, p, r)
            W_3d = W.reshape(p, n, r)
            
            # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            T_approx = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            
            # Minimize sum of squares of (T_approx - 1)
            return np.sum((T_approx - 1.0) ** 2)
            
        return objective

    def try_rank(r):
        if time.time() > deadline:
            return None
        
        objective = make_objective(r)
        u_size = n * m * r
        v_size = m * p * r
        total_size = u_size + v_size + p * n * r
        
        # Try a few random initializations
        for attempt in range(30):
            if time.time() > deadline:
                return None
            
            # Random initialization
            x0 = np.random.randn(total_size)
            
            # Use L-BFGS-B for speed
            try:
                result = opt.minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B', 
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if result.fun < 1e-6:
                U = result.x[:u_size].reshape(n * m, r)
                V = result.x[u_size:u_size+v_size].reshape(m * p, r)
                W = result.x[u_size+v_size:].reshape(p * n, r)
                
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
        
        return None

    # Try ranks from 28 down to 29
    for r in [28, 29]:
        if time.time() > deadline:
            break
        res = try_rank(r)
        if res:
            U, V, W = res
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Return schoolbook
    return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
