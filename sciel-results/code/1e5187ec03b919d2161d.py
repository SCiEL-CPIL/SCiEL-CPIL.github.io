"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T.reshape(-1)
    
    def _als_step(U, V, W, T, rank):
        """Perform one ALS step: update U given V, W, then V given U, W, then W given U, V."""
        # Update U
        # For each a, T[a, :, :] is approx U[a, :] @ (V[:, r] * W[:, r])^T? No.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Fix V, W. For each a, we want to find U[a, :] such that 
        # U[a, :] @ M_a^T approx T[a, :, :] flattened?
        # Let X be a matrix of size (m*p * p*n) x rank, where X[b*p+c, r] = V[b, r] * W[c, r].
        # Then T[a, b, c] = sum_r U[a, r] X[b*p+c, r].
        # So T[a, :] (flattened b,c) = U[a, :] X^T.
        # U[a, :] X^T = T[a, :].
        # U X^T = T_flat (shape n*m x m*p*p*n).
        # U = T_flat (X^T)^+
        
        X = np.zeros((m*p, p*n, rank))
        # X[b, c, r] = V[b, r] * W[c, r]
        for r in range(rank):
            X[:, :, r] = np.outer(V[:, r], W[:, r])
        
        # Flatten X to (m*p*p*n, rank)
        X_flat = X.reshape(-1, rank)
        
        # Solve U X_flat^T = T_flat
        # T_flat is (n*m, m*p*p*n)
        # X_flat^T is (rank, m*p*p*n)
        # We want U (n*m, rank)
        # U X_flat^T = T_flat
        # This is a least squares problem for each row of U.
        # Or U = T_flat pinv(X_flat^T)
        # Since X_flat^T is (rank, m*p*p*n) and rank is small, we can use lstsq.
        # Actually, for each a, U[a, :] is the solution to U[a, :] X_flat^T = T_flat[a, :]
        # This is equivalent to solving (X_flat X_flat^T) U[a, :]^T = X_flat T_flat[a, :]^T
        # Let's just use lstsq on the transposed system or solve directly.
        
        # Efficient way:
        # For all a: U[a, :] = argmin || T_flat[a, :] - U[a, :] X_flat^T ||
        # This is equivalent to solving:
        # X_flat X_flat^T U[a, :]^T = X_flat T_flat[a, :]^T
        # Let A = X_flat X_flat^T (rank x rank)
        # Let B = X_flat T_flat^T (rank x n*m)
        # Then U^T = A^+ B
        # U = B^T (A^T)^+^T
        
        try:
            A = X_flat @ X_flat.T
            B = X_flat @ T_flat.T
            # Solve A U^T = B
            U_new = np.linalg.lstsq(A, B, rcond=None)[0].T
        except:
            U_new = U
            
        # Update V
        # Fix U, W.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For each b, T[:, b, :] flattened (a,c) = V[b, :] Y^T where Y[a, c, r] = U[a, r] W[c, r]
        # Y_flat (n*m * p*n, rank) with Y_flat[a*p*n+c, r] = U[a, r] * W[c, r]
        # T_flat_b is T[:, b, :] reshaped to (n*m * p*n, 1)? No, T_flat is (n*m * m*p * p*n, 1)
        # We need to extract T[:, b, :] for each b.
        
        Y = np.zeros((n*m, p*n, rank))
        for r in range(rank):
            Y[:, :, r] = np.outer(U[:, r], W[:, r])
        Y_flat = Y.reshape(-1, rank)
        
        # T is (n*m, m*p, p*n)
        # We want to solve V[b, :] Y_flat^T = T[:, b, :].flatten()
        # Let T_b be T[:, b, :].reshape(-1) (size n*m*p*n)
        # V[b, :] Y_flat^T = T_b
        # Sum over b: V (m*p, rank)
        # This can be solved as a block system or individually.
        # A = Y_flat Y_flat^T (rank x rank)
        # C = Y_flat T_reshaped^T where T_reshaped is (m*p, n*m*p*n)?
        # T is (n*m, m*p, p*n). We want T[:, b, :].
        # Let's transpose T to (m*p, n*m, p*n) and flatten the last two dims.
        T_b_flat = T.transpose(1, 0, 2).reshape(m*p, -1) # (m*p, n*m*p*n)
        
        try:
            A = Y_flat @ Y_flat.T
            C = Y_flat @ T_b_flat.T # (rank, m*p)
            V_new = np.linalg.lstsq(A, C, rcond=None)[0].T
        except:
            V_new = V
            
        # Update W
        # Fix U, V.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For each c, T[:, :, c] flattened (a,b) = W[c, :] Z^T where Z[a, b, r] = U[a, r] V[b, r]
        # Z_flat (n*m * m*p, rank) with Z_flat[a*m*p+b, r] = U[a, r] V[b, r]
        # T_c_flat is T[:, :, c].reshape(-1) (size n*m*m*p)
        
        Z = np.zeros((n*m, m*p, rank))
        for r in range(rank):
            Z[:, :, r] = np.outer(U[:, r], V[:, r])
        Z_flat = Z.reshape(-1, rank)
        
        # T is (n*m, m*p, p*n). We want T[:, :, c].
        T_c_flat = T.transpose(2, 0, 1).reshape(p*n, -1) # (p*n, n*m*m*p)
        
        try:
            A = Z_flat @ Z_flat.T
            D = Z_flat @ T_c_flat.T # (rank, p*n)
            W_new = np.linalg.lstsq(A, D, rcond=None)[0].T
        except:
            W_new = W
            
        return U_new, V_new, W_new

    def _verify(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < 1e-9

    def _verify_rational(U, V, W):
        """Check if a rational approximation with small denominators exactly matches T."""
        # Try denominators 1 and 2
        for denom in [1, 2]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            
            # Check if exactly representable
            if not np.allclose(U_r * denom, np.round(U_r * denom)): continue
            if not np.allclose(V_r * denom, np.round(V_r * denom)): continue
            if not np.allclose(W_r * denom, np.round(W_r * denom)): continue
            
            T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.max(np.abs(T_approx - T)) < 1e-10:
                return U_r, V_r, W_r
        return None

    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p # 18
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 13 up to 17
    for rank in range(13, 18):
        if _remaining() < 10:
            break
            
        found = False
        max_restarts = 200
        if rank > 13:
            max_restarts = 500
            
        for restart in range(max_restarts):
            if _remaining() < 5:
                break
                
            # Random initialization
            scale = 0.5
            U = rng.standard_normal((n*m, rank)) * scale
            V = rng.standard_normal((m*p, rank)) * scale
            W = rng.standard_normal((p*n, rank)) * scale
            
            # ALS iterations
            for it in range(100):
                U, V, W = _als_step(U, V, W, T, rank)
                
                # Check error
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_approx - T))
                
                if err < 1e-9:
                    # Try rational reconstruction
                    result = _verify_rational(U, V, W)
                    if result is not None:
                        best_U, best_V, best_W = result
                        best_rank = rank
                        found = True
                        break
            
            if found:
                break
                
        if found:
            break
    
    # If we didn't find a better rank, return schoolbook
    if best_rank == n * m * p:
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
