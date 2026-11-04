import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target: 3x3x4
    # Rank 36 is schoolbook.
    # We try to find a lower rank using ALS + Snapping to half-integers.
    
    n, m, p = 3, 3, 4
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    start_time = time.time()
    timeout = 100
    
    best_rank = 36
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return U, V, W, rank

    # Try ranks from 25 to 35
    for R in range(25, 36):
        if time.time() - start_time > timeout:
            break
            
        for attempt in range(5):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + R + attempt)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS Iterations
            converged = False
            for it in range(10):
                # Solve for W
                M = np.zeros((n*m*m*p, p*n*R))
                for c in range(p*n):
                    for r in range(R):
                        M[:, c*R + r] = np.outer(U[:, r], V[:, r]).flatten()
                T_flat = T.reshape(n*m*m*p, p*n)
                
                W_new = np.zeros((p*n, R))
                for c in range(p*n):
                    try:
                        W_new[c, :] = np.linalg.lstsq(M[:, c*R : (c+1)*R], T_flat[:, c], rcond=None)[0]
                    except:
                        pass
                        
                error = np.sum((T_flat - M @ W_new.T).reshape(n*m*m*p, p*n)**2)
                # Note: M @ W_new.T is not quite right for the flattening.
                # T_flat[:, c] = sum_r M[:, c*R+r] W_new[c, r]
                # This is a block diagonal system.
                # Let's just calculate the error directly.
                error = 0
                for c in range(p*n):
                    pred = np.sum(M[:, c*R : (c+1)*R] * W_new[c, :], axis=1)
                    error += np.sum((pred - T_flat[:, c])**2)
                    
                if error < 1e-6:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W_new * 2) / 2
                    
                    # Verify exactness
                    T_recon = np.zeros((n*m, m*p, p*n))
                    for r in range(R):
                        T_recon += np.einsum('ar,br,cr->abc', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                    
                    if np.allclose(T_recon, T, atol=1e-9):
                        best_rank = R
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        converged = True
                        break
                        
                # Solve for U
                M_V = np.zeros((m*p*p*n, n*m*R))
                for a in range(n*m):
                    for r in range(R):
                        M_V[:, a*R + r] = np.outer(V[:, r], W[:, r]).flatten()
                T_flat_V = T.reshape(m*p*p*n, n*m)
                U_new = np.zeros((n*m, R))
                for a in range(n*m):
                    try:
                        U_new[a, :] = np.linalg.lstsq(M_V[:, a*R : (a+1)*R], T_flat_V[:, a], rcond=None)[0]
                    except:
                        pass
                U = U_new
                
                # Solve for V
                M_W = np.zeros((p*n*n*m, m*p*R))
                for b in range(m*p):
                    for r in range(R):
                        M_W[:, b*R + r] = np.outer(U[:, r], W[:, r]).flatten()
                T_flat_W = T.reshape(p*n*n*m, m*p)
                V_new = np.zeros((m*p, R))
                for b in range(m*p):
                    try:
                        V_new[b, :] = np.linalg.lstsq(M_W[:, b*R : (b+1)*R], T_flat_W[:, b], rcond=None)[0]
                    except:
                        pass
                V = V_new
                
            if converged:
                break
                
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W, rank = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

