import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS

    def verify(U, V, W):
        U_mat = U.reshape(n, m, -1)
        V_mat = V.reshape(m, p, -1)
        W_mat = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        return float(np.max(np.abs(T - 1.0)))

    def alternating_minimization(U, V, W, rank, max_iters=200, tol=1e-10):
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            # For each (i,j), U[i*m+j, :] = argmin || V[j*p+k, :] * W[k*n+i, :] * U[i*m+j, :] - 1 ||
            # This is a linear system: B @ U_i = 1
            # B is p x rank
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    B = np.einsum('kc,ic->k', V[j*p:(j+1)*p, :], W[:, i]) # This is wrong, W is p*n
                    # W is (p*n, rank). W[k*n+i, :]
                    # B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    B = np.einsum('kc,ic->kc', V[j*p:(j+1)*p, :], W[np.arange(p)*n+i, :])
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
            U = U_new

            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    A = np.einsum('ic,kc->ic', U[np.arange(n)*m+j, :], W[k*n:(k+1)*n, :])
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            V = V_new

            # Update W
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    C = np.einsum('jc,kc->jc', U[np.arange(m)*m+j, :], V[j*p+k, :]) # Wrong indexing
                    # C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    C = np.einsum('jc,kc->jc', U[np.arange(m)*m+j, :], V[np.arange(m)*p+k, :])
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    except:
                        pass
            W = W_new

            err = verify(U, V, W)
            if err < tol:
                break
        
        return verify(U, V, W)

    # Try rank 49
    rank = 49
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    err = alternating_minimization(U, V, W, rank, max_iters=100)
    
    if err < 1e-8:
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r) < 1e-8:
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }

    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
