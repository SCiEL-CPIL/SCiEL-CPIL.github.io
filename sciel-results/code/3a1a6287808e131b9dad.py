"""Matrix multiplication 2x5 by 5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
TIME_LIMIT = 130  # safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the 3-way tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] = delta_{j,j'}*delta_{k,k'}*delta_{i,i'}"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    T_recon = np.zeros((n * m, m * p, p * n))
    for r in range(R):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    T_true = _build_tensor(n, m, p)
    max_err = np.max(np.abs(T_recon - T_true))
    return max_err < tol

def _make_schoolbook(n, m, p):
    """Create the schoolbook rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _try_strassen_based(n, m, p):
    """
    Try to construct a decomposition using Strassen-like algorithms.
    
    For (2x5)*(5x5), we can view this as two (1x5)*(5x5) multiplications.
    The (1x5)*(5x5) multiplication computes 5 dot products of length 5.
    
    One approach: use the fact that we can combine computations.
    However, for (1x5)*(5x5), the schoolbook rank is 25, and it's known
    that this is actually minimal (each of the 25 products is independent).
    
    Let's try a different approach: use random initialization and alternating
    least squares to try to find a lower rank decomposition.
    """
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    T = _build_tensor(n, m, p)
    
    # Try ranks below schoolbook
    schoolbook_rank = n * m * p  # 50
    
    for R in range(20, schoolbook_rank + 1):
        # Initialize with random values
        np.random.seed(R)
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        # Alternating minimization
        converged = False
        for iteration in range(100):
            # Fix V, W and solve for U
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
            # Let M_r = V[:,r] * W[:,r] (element-wise product, shape mp*pn)
            # Then T[a,:,:] = sum_r U[a,r] * M_r reshaped to (mp, pn)
            
            # Reshape T[a] to (mp*pn,)
            # Build system: for each a, T[a].reshape(mp*pn) = sum_r U[a,r] * (V[:,r] * W[:,r]).reshape(mp*pn)
            
            # This is a linear system in U for fixed V, W
            # T_flat[a] = U @ (V*W)_flat where (V*W)_flat[r] = (V[:,r]*W[:,r]).reshape(-1)
            
            # Let A = matrix of shape (nm, R) where A[a,r] = 1 if we're solving for U
            # Actually, T_flat[a] = sum_r U[a,r] * B[r] where B[r] = (V[:,r]*W[:,r]).reshape(-1)
            # So T_flat = U @ B^T where B is (R, mp*pn)
            
            B = np.array([(V[:, r] * W[:, r]).flatten() for r in range(R)])  # (R, mp*pn)
            T_flat = T.reshape(nm, mp * pn)  # (nm, mp*pn)
            
            # Solve T_flat = U @ B^T for U
            # U = T_flat @ (B^T)^+ = T_flat @ pinv(B^T)
            try:
                U_new = T_flat @ np.linalg.pinv(B.T)
            except:
                break
            
            # Fix U, W and solve for V
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each b: T[:,:,b] = sum_r V[b,r] * (U[:,r] * W[:,r])
            # Let C_r = U[:,r] * W[:,r] (element-wise, shape nm*pn)
            # T[:,:,b].reshape(nm*pn) = sum_r V[b,r] * C_r.reshape(nm*pn)
            
            C = np.array([(U[:, r] * W[:, r]).flatten() for r in range(R)])  # (R, nm*pn)
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # (mp, nm*pn)
            
            try:
                V_new = T_flat_v @ np.linalg.pinv(C.T)
            except:
                break
            
            # Fix U, V and solve for W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each c: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
            # Let D_r = U[:,r] * V[:,r] (element-wise, shape nm*mp)
            # T[:,:,c].reshape(nm*mp) = sum_r W[c,r] * D_r.reshape(nm*mp)
            
            D = np.array([(U[:, r] * V[:, r]).flatten() for r in range(R)])  # (R, nm*mp)
            T_flat_w = T.transpose(0, 2, 1).reshape(pn, nm * mp)  # (pn, nm*mp)
            
            try:
                W_new = T_flat_w @ np.linalg.pinv(D.T)
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_recon = np.zeros((nm, mp, pn))
            for r in range(R):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.max(np.abs(T_recon - T))
            
            if err < 1e-6:
                converged = True
                break
        
        if converged:
            # Verify with exact check
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                print(f"Found rank {R} decomposition!")
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    
    # First, try to find a better decomposition using alternating minimization
    result = _try_strassen_based(n, m, p)
    
    if result is None:
        # Fall back to schoolbook
        result = _make_schoolbook(n, m, p)
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            print(f"Schoolbook decomposition verified, rank = {n*m*p}")
        else:
            print("WARNING: Schoolbook decomposition failed verification!")
    
    elapsed = time.time() - start_time
    print(f"construct() completed in {elapsed:.2f}s")
    
    return result
