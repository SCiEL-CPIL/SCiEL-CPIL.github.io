import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)
    
    def build_strassen_rank50():
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        
        def emit_strassen(i0, j0, k0):
            nonlocal col
            U[i0*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            U[(i0+1)*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = -1.0
            col += 1
            U[i0*m + j0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[j0*p + k0, col] = -1.0
            W[k0*n + (i0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            U[i0*m + j0, col] = 1.0
            U[i0*m + (j0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            col += 1
            U[(i0+1)*m + j0, col] = 1.0
            U[i0*m + j0, col] = -1.0
            V[j0*p + k0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            U[i0*m + (j0+1), col] = 1.0
            U[(i0+1)*m + (j0+1), col] = -1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1

        def emit_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col] = 1.0
                    V[(j0+dj)*p + k0, col] = 1.0
                    W[k0*n + (i0+di), col] = 1.0
                    col += 1

        emit_strassen(0, 0, 0)
        emit_strassen(0, 0, 2)
        emit_strassen(0, 0, 4)
        emit_schoolbook_2x2x1(0, 0, 6)
        emit_strassen(0, 2, 0)
        emit_strassen(0, 2, 2)
        emit_strassen(0, 2, 4)
        emit_schoolbook_2x2x1(0, 2, 6)
        
        return U, V, W, rank
    
    def try_als_decomposition(R, max_iters=500, n_restarts=5):
        """Try to find a rank-R decomposition using ALS."""
        for restart in range(n_restarts):
            if time.time() - start_time > DEADLINE - 15:
                break
            rng = np.random.RandomState(42 + restart)
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
            
            for it in range(max_iters):
                if time.time() - start_time > DEADLINE - 15:
                    break
                
                # Update U: for each a, U[a,:] = argmin ||T[a,:,:].reshape(-1) - (V⊗W) @ U[a,:]||
                VW = np.einsum('br,cr->bc', V, W).reshape(-1, R)
                U = np.linalg.lstsq(VW, T.reshape(n*m, -1).T, rcond=None)[0].T
                
                # Update V
                UW = np.einsum('ar,cr->ac', U, W).reshape(-1, R)
                V = np.linalg.lstsq(UW, T.transpose(1,0,2).reshape(m*p, -1).T, rcond=None)[0].T
                
                # Update W
                UV = np.einsum('ar,br->ab', U, V).reshape(-1, R)
                W = np.linalg.lstsq(UV, T.transpose(2,0,1).reshape(p*n, -1).T, rcond=None)[0].T
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_recon - T)
                if err < 1e-10:
                    if verify(U, V, W):
                        return U, V, W, R
                if err < 1e-12:
                    break
        
        return None
    
    # Try ranks from low to high
    best_result = None
    for R in range(28, 51):
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            result = try_als_decomposition(R, max_iters=200, n_restarts=3)
            if result is not None:
                U, V, W, rank = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            # If ALS fails, continue to next rank or fallback
            continue
    
    # Fall back to rank-50
    U, V, W, rank = build_strassen_rank50()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Last resort: schoolbook rank 56
    rank = 56
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
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If schoolbook also fails verification (unlikely), return zeros? 
    # But schoolbook is mathematically exact.
    # So this return is a safety net.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
