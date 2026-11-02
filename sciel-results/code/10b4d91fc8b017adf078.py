import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    d1, d2, d3 = T.shape  # 15, 35, 21
    
    def run_als(T, rank, max_iter=3000, tol=1e-12, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(d1, rank)
        V = rng.randn(d2, rank)
        W = rng.randn(d3, rank)
        
        # Precompute matricizations of T
        T1 = T.reshape(d1, d2*d3)
        T2 = T.transpose(1,0,2).reshape(d2, d1*d3)
        T3 = T.transpose(2,0,1).reshape(d3, d1*d2)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Compute M_VW[(j,k), t] = V[j,t] * W[k,t]
            M_VW = np.einsum('jt,kt->jkt', V, W).reshape(d2*d3, rank)
            # Update U
            A = M_VW.T @ M_VW
            B = T1 @ M_VW
            try:
                U = np.linalg.solve(A, B.T).T
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(A, B.T, rcond=None)[0].T
            
            # Compute M_UW[(i,k), t] = U[i,t] * W[k,t]
            M_UW = np.einsum('it,kt->ikt', U, W).reshape(d1*d3, rank)
            # Update V
            A = M_UW.T @ M_UW
            B = T2 @ M_UW
            try:
                V = np.linalg.solve(A, B.T).T
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(A, B.T, rcond=None)[0].T
            
            # Compute M_UV[(i,j), t] = U[i,t] * V[j,t]
            M_UV = np.einsum('it,jt->ijt', U, V).reshape(d1*d2, rank)
            # Update W
            A = M_UV.T @ M_UV
            B = T3 @ M_UV
            try:
                W = np.linalg.solve(A, B.T).T
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A, B.T, rcond=None)[0].T
            
            # Compute error periodically
            if it % 100 == 0:
                recon = np.einsum('it,jt,kt->ijk', U, V, W)
                err = np.linalg.norm(recon - T)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < tol:
                    break
                if time.time() > deadline:
                    break
        
        recon = np.einsum('it,jt,kt->ijk', best_U, best_V, best_W)
        err = np.linalg.norm(recon - T)
        return best_U, best_V, best_W, err
    
    def snap_to_half_int(M):
        return np.round(2.0 * M) / 2.0
    
    def verify(U, V, W, T, tol=1e-9):
        recon = np.einsum('it,jt,kt->ijk', U, V, W)
        return np.linalg.norm(recon - T) < tol
    
    def try_rank(r, n_init=5):
        best = None
        for init in range(n_init):
            U, V, W, err = run_als(T, r, max_iter=2000, seed=init)
            if time.time() > deadline:
                break
            # Snap to half-integers
            U_s = snap_to_half_int(U)
            V_s = snap_to_half_int(V)
            W_s = snap_to_half_int(W)
            if verify(U_s, V_s, W_s, T, tol=1e-8):
                return U_s, V_s, W_s, 0.0
            if best is None or err < best[3]:
                best = (U, V, W, err)
        return best
    
    # Try ranks from 35 upward
    for r in [35, 40, 50, 60, 80, 105]:
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W, err = result
            if verify(U, V, W, T, tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
