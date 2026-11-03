import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_222():
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            
            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])
            
    return U, V, W

def _direct_sum_decomposition(U1, V1, W1):
    n, r = U1.shape
    m, _ = V1.shape
    p, _ = W1.shape
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)
    W = np.zeros((2 * p, R), dtype=float)
    
    U[:, :r] = U1
    V[:m, :r] = V1
    W[:p, :r] = W1
    
    U[:, r:] = U1
    V[m:, r:] = V1
    W[p:, r:] = W1
    
    return U, V, W

def _verify_444(U, V, W):
    # U: (4, R), V: (4, R), W: (4, R)
    # Check sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
    n = 4
    R = U.shape[1]
    # Reshape to (n, n, R)
    U3 = U.reshape(n, n, R)
    V3 = V.reshape(n, n, R)
    W3 = W.reshape(n, n, R)
    
    # Sum over r
    # result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # We can do this via einsum
    # 'ijr,jkr,kir->ijk'
    prod = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Check if prod is all 1s
    if not np.allclose(prod, 1.0, atol=1e-6):
        return False
    return True

def _snap_to_half_integer(M):
    # Round to nearest 0.5
    return np.round(M * 2) / 2

def _try_find_444(R, time_limit=10):
    # Try to find decomposition of rank R for 4x4x4
    start_time = time.time()
    n = 4
    # Initialize
    U = np.random.randn(n*n, R) * 0.1
    V = np.random.randn(n*n, R) * 0.1
    W = np.random.randn(n*n, R) * 0.1
    
    for _ in range(200): # Max iterations
        if time.time() - start_time > time_limit:
            break
        
        # Solve for W
        # System: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # For fixed U, V, linear in W.
        # Variable: W_flat (16*R)
        # Eq (i,j,k): sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        A_W = np.zeros((n*n*n, n*n*R))
        b = np.ones(n*n*n)
        
        idx = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for r in range(R):
                        # Coeff of W[k,i,r] is U[i,j,r] * V[j,k,r]
                        col = r * n*n + k*n + i
                        A_W[idx, col] = U[i*n+j, r] * V[j*n+k, r]
                    idx += 1
        
        W_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
        W = W_flat.reshape(n*n, R)
        
        # Solve for V
        A_V = np.zeros((n*n*n, n*n*R))
        idx = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for r in range(R):
                        # Coeff of V[j,k,r] is U[i,j,r] * W[k,i,r]
                        col = r * n*n + j*n + k
                        A_V[idx, col] = U[i*n+j, r] * W[k*n+i, r]
                    idx += 1
        V_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
        V = V_flat.reshape(n*n, R)
        
        # Solve for U
        A_U = np.zeros((n*n*n, n*n*R))
        idx = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for r in range(R):
                        # Coeff of U[i,j,r] is V[j,k,r] * W[k,i,r]
                        col = r * n*n + i*n + j
                        A_U[idx, col] = V[j*n+k, r] * W[k*n+i, r]
                    idx += 1
        U_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
        U = U_flat.reshape(n*n, R)
        
        # Check convergence
        if _verify_444(U, V, W):
            # Try to snap
            U_s = _snap_to_half_integer(U)
            V_s = _snap_to_half_integer(V)
            W_s = _snap_to_half_integer(W)
            if _verify_444(U_s, V_s, W_s):
                return U_s, V_s, W_s
            else:
                # Maybe the float solution is good enough? 
                # But task requires integer/half-integer.
                # If float works but snapped doesn't, it's not a valid solution for the task.
                pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better rank for 4x4x4
    # If found, use it.
    best_444 = None
    for R in range(48, 39, -1): # Try 48 down to 40
        res = _try_find_444(R, time_limit=5)
        if res is not None:
            best_444 = res
            break
            
    if best_444 is not None:
        U4, V4, W4 = best_444
        U, V, W = _direct_sum_decomposition(U4, V4, W4)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to seed
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    U, V, W = _direct_sum_decomposition(U444, V444, W444)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
