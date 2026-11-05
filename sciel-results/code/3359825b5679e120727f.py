import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_base_decomposition(n, m, p):
    """Construct a structured decomposition using Strassen for 2x2x2 blocks."""
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        for dj in range(2):
            for dk in range(2):
                U[(rA+0)*m + (cA+dj), col] = 1
                V[(rB+dj)*p + (cB+dk), col] = 1
                W[(cB+dk)*n + (rA+0), col] = 1
                col += 1

    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=200, tol=1e-8):
    """Try to reduce rank by finding linear dependencies."""
    rank = U.shape[1]
    if rank <= target_rank:
        return U, V, W, rank
    
    # Try ALS with target rank
    np.random.seed(42)
    U_new = np.random.randn(n*m, target_rank) * 0.5
    V_new = np.random.randn(m*p, target_rank) * 0.5
    W_new = np.random.randn(p*n, target_rank) * 0.5
    
    T = _build_target_tensor(n, m, p)
    
    for it in range(max_iter):
        # Update U
        for r in range(target_rank):
            Vr = V_new[:, r]
            Wr = W_new[:, r]
            num = np.zeros(n*m)
            den = np.zeros(n*m)
            for i in range(n):
                for j in range(m):
                    s_num = 0.0
                    s_den = 0.0
                    for k in range(p):
                        s_num += T[i,j,k] * Vr[j*p+k] * Wr[k*n+i]
                        s_den += Vr[j*p+k]**2 * Wr[k*n+i]**2
                    num[i*m+j] = s_num
                    den[i*m+j] = s_den
            mask = den > 1e-12
            U_new[mask, r] = num[mask] / den[mask]
        
        # Update V
        for r in range(target_rank):
            Ur = U_new[:, r]
            Wr = W_new[:, r]
            num = np.zeros(m*p)
            den = np.zeros(m*p)
            for j in range(m):
                for k in range(p):
                    s_num = 0.0
                    s_den = 0.0
                    for i in range(n):
                        s_num += T[i,j,k] * Ur[i*m+j] * Wr[k*n+i]
                        s_den += Ur[i*m+j]**2 * Wr[k*n+i]**2
                    num[j*p+k] = s_num
                    den[j*p+k] = s_den
            mask = den > 1e-12
            V_new[mask, r] = num[mask] / den[mask]
        
        # Update W
        for r in range(target_rank):
            Ur = U_new[:, r]
            Vr = V_new[:, r]
            num = np.zeros(p*n)
            den = np.zeros(p*n)
            for k in range(p):
                for i in range(n):
                    s_num = 0.0
                    s_den = 0.0
                    for j in range(m):
                        s_num += T[i,j,k] * Ur[i*m+j] * Vr[j*p+k]
                        s_den += Ur[i*m+j]**2 * Vr[j*p+k]**2
                    num[k*n+i] = s_num
                    den[k*n+i] = s_den
            mask = den > 1e-12
            W_new[mask, r] = num[mask] / den[mask]
        
        # Check convergence
        T_recon = np.zeros((n, m, p))
        for r in range(target_rank):
            u = U_new[:, r].reshape(n, m)
            v = V_new[:, r].reshape(m, p)
            w = W_new[:, r].reshape(p, n)
            T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
    
    if np.max(np.abs(T_recon - T)) < 1e-4:
        return U_new, V_new, W_new, target_rank
    
    return U, V, W, rank

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    U, V, W = _construct_base_decomposition(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to reduce rank
    for target_rank in [43, 42, 41, 40, 39, 38, 37, 36, 35]:
        if time.time() > deadline - 20:
            break
        
        U_new, V_new, W_new, new_rank = _try_reduce_rank(
            best_U, best_V, best_W, n, m, p, target_rank, max_iter=100
        )
        
        if new_rank < best_rank and _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-4):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = new_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
