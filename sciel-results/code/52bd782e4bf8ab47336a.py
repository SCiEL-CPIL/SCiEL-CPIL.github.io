import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # Use alternating least squares to find a low-rank decomposition
    # Conditions: for all i,k,j1,j2:
    # sum_t W[k*n+i, t] * U[i*m+j1, t] * V[j2*p+k, t] = delta(j1,j2)
    
    for rank in [50, 55]:
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        for iteration in range(100):
            # Update U
            for i in range(n):
                for j1 in range(m):
                    A_sys = np.zeros((p * m, rank))
                    b_sys = np.zeros(p * m)
                    idx = 0
                    for k in range(p):
                        for j2 in range(m):
                            A_sys[idx, :] = W[k * n + i, :] * V[j2 * p + k, :]
                            b_sys[idx] = 1.0 if j1 == j2 else 0.0
                            idx += 1
                    U[i * m + j1, :], _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            
            # Update V
            for j2 in range(m):
                for k in range(p):
                    A_sys = np.zeros((n * m, rank))
                    b_sys = np.zeros(n * m)
                    idx = 0
                    for i in range(n):
                        for j1 in range(m):
                            A_sys[idx, :] = W[k * n + i, :] * U[i * m + j1, :]
                            b_sys[idx] = 1.0 if j1 == j2 else 0.0
                            idx += 1
                    V[j2 * p + k, :], _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A_sys = np.zeros((m * m, rank))
                    b_sys = np.zeros(m * m)
                    idx = 0
                    for j1 in range(m):
                        for j2 in range(m):
                            A_sys[idx, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                            b_sys[idx] = 1.0 if j1 == j2 else 0.0
                            idx += 1
                    W[k * n + i, :], _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
        
        # Check exactness
        max_error = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = np.sum(W[k * n + i, :] * U[i * m + j1, :] * V[j2 * p + k, :])
                        target = 1.0 if j1 == j2 else 0.0
                        max_error = max(max_error, abs(val - target))
        
        if max_error < 1e-8:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            max_error_r = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            val = np.sum(W_r[k * n + i, :] * U_r[i * m + j1, :] * V_r[j2 * p + k, :])
                            target = 1.0 if j1 == j2 else 0.0
                            max_error_r = max(max_error_r, abs(val - target))
            
            if max_error_r < 1e-8:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    rank = 63
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
