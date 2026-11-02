"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 135  # seconds, leave 15s margin for 150s limit


def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                result[i, k, j] = s  # note: careful with indexing
    # Actually let's do it properly
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                result[i, j, k] = s
    return np.allclose(result, 1.0, atol=1e-9)


def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-6


def _try_round(U, V, W):
    """Try to round entries to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    n, m, p = DIMENSIONS
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None


def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _kronecker_structure(n, m, p):
    """
    Try to exploit Kronecker/block structure.
    For 2x4 * 4x7: think of the middle dimension 4 as 2*2.
    We can write A (2x4) as [A1 | A2] where A1, A2 are 2x2.
    B (4x7) as [B1; B2] where B1, B2 are 2x7.
    Then C = A1*B1 + A2*B2.
    
    This doesn't directly reduce rank below schoolbook for the sum,
    but we can try to find a decomposition where the rank is
    rank(A1*B1) + rank(A2*B2) if they share structure.
    
    More practically: try to find a rank-r decomposition where r < 56
    by using the fact that the tensor has some structure.
    
    Let's try a different approach: use the known bounds.
    For C(n,m,p), we have C(n,m,p) <= n*m*p (schoolbook).
    Also C(n,m,p) <= min(n*m*p, ...).
    
    Actually, let's just try to find ANY rank below 56 using ALS
    with smart initialization based on the Kronecker structure.
    """
    pass


def _als_decompose(n, m, p, rank, max_iters=50, tol=1e-6, seed=42):
    """
    Alternating least squares for tensor decomposition.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Normalize columns to prevent blowup
    for c in range(rank):
        nU = np.linalg.norm(U[:, c])
        nV = np.linalg.norm(V[:, c])
        nW = np.linalg.norm(W[:, c])
        if nU > 0 and nV > 0 and nW > 0:
            scale = (nU * nV * nW) ** (1.0 / 3.0)
            U[:, c] /= scale
            V[:, c] /= scale
            W[:, c] /= scale
    
    T = np.ones((n, m, p), dtype=np.float64)
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Fix (i,j): T[i,j,:] = U[i*m+j,:] @ M where M[c,k] = V[j*p+k,c]*W[k*n+i,c]
                M = np.zeros((rank, p))
                for k in range(p):
                    for c in range(rank):
                        M[c, k] = V[j * p + k, c] * W[k * n + i, c]
                # Solve: M.T @ U[i*m+j,:].T = T[i,j,:].T
                # i.e., U[i*m+j,:] = lstsq(M.T, T[i,j,:].T)
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :].T, rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Fix (j,k): T[:,j,k] = V[j*p+k,:] @ N where N[c,i] = U[i*m+j,c]*W[k*n+i,c]
                N = np.zeros((rank, n))
                for i in range(n):
                    for c in range(rank):
                        N[c, i] = U[i * m + j, c] * W[k * n + i, c]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(N.T, T[:, j, k].T, rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Fix (k,i): T[i,:,k] = W[k*n+i,:] @ P where P[c,j] = U[i*m+j,c]*V[j*p+k,c]
                P = np.zeros((rank, m))
                for j in range(m):
                    for c in range(rank):
                        P[c, j] = U[i * m + j, c] * V[j * p + k, c]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(P.T, T[i, :, k].T, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
        res = np.linalg.norm(recon - T)
        if res < tol:
            break
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = n * m * p
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # But tensor rank is typically higher. Let's try from 30 to 55.
    for rank in range(30, best_rank):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Try a few seeds
        for trial in range(2):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            U, V, W = _als_decompose(n, m, p, rank, max_iters=30, seed=rank * 100 + trial)
            
            # Try rounding to half-integers
            rounded = _try_round(U.tolist(), V.tolist(), W.tolist())
            if rounded is not None:
                U_r, V_r, W_r = rounded
                if _verify(U_r, V_r, W_r, n, m, p):
                    return {"U": U_r, "V": V_r, "W": W_r}
            
            # Also check if the raw solution is close enough that rounding works
            # Even if not half-integer, check if it's valid as-is (but entries must be half-integers)
            # So we must round.
    
    # If we haven't found anything better, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
