import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower rank
    # We will try ranks from 35 down to 30
    # Using a simple random restart local search (Newton's method on the residual)
    
    def get_tensor(U, V, W):
        # U: nm x R, V: mp x R, W: pn x R
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        R = U.shape[1]
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Efficient computation
        # Loop over k
        for k in range(p):
            # W[k*n + i, :] for all i
            # V[j*p + k, :] for all j
            # U[i*m + j, :] for all i,j
            
            # Reshape for easier indexing
            # This is slow in pure python loops, so we use numpy broadcasting if possible
            # But R is small, so looping is okay
            
            for i in range(n):
                for j in range(m):
                    u_val = U_arr[i * m + j, :]
                    v_val = V_arr[j * p + k, :]
                    w_val = W_arr[k * n + i, :]
                    T[i, j, k] = np.dot(u_val, v_val * w_val)
        return T

    def residual_norm(U, V, W):
        T = get_tensor(U, V, W)
        # Target tensor T_target[i, j, k] = 1 if i,j,k are valid indices for schoolbook?
        # No, the target is the matrix multiplication tensor.
        # T_target[i, j, k] is 1 if the term C[i,k] += A[i,j]B[j,k] is generated.
        # Actually, the tensor T for matrix multiplication has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
        # So T[i,j,k] = 1 for all i,j,k? No.
        # The multiplication is C = AB.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T is defined such that T(A,B) = C.
        # T is a multilinear map.
        # T[i, j, k] = 1 if the term A[i,j] * B[j,k] appears in C[i,k].
        # Yes, for all i in 0..n-1, j in 0..m-1, k in 0..p-1, T[i,j,k] = 1.
        # Wait, is that right?
        # If T[i,j,k] = 1, then C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]?
        # The decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
        # Then C[i,k] = sum_j (sum_r U[i,j,r] V[j,k,r] W[k,i,r]) A[i,j] B[j,k].
        # We want the coefficient of A[i,j] B[j,k] in C[i,k] to be 1.
        # So yes, the target tensor T_target[i, j, k] is 1 for all valid i, j, k.
        
        T_target = np.ones((n, m, p))
        return np.linalg.norm(T - T_target)

    def try_rank(R):
        if R >= best_rank:
            return None
        
        # Initialize with random values
        # Or initialize with schoolbook terms and perturb
        # For R < 36, we can't just take a subset.
        # We use random initialization with small values to avoid divergence
        # Or use the schoolbook U,V,W but truncate? No.
        
        # Random init
        for _ in range(20): # 20 restarts
            if time.time() > deadline - 5:
                return None
            
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Newton's method / Gradient Descent
            # We minimize f(U,V,W) = || T(U,V,W) - T_target ||^2
            # Gradient with respect to U[i,j,r] is 2 * sum_{j',k} (T[i,j',k] - 1) V[j',k,r] W[k,i,r]
            # This is complex to implement efficiently without autograd.
            # Let's use a simple finite difference gradient descent for a few steps?
            # Too slow.
            
            # Alternative: Use scipy.optimize.least_squares
            # Variables: x = [U.flatten(), V.flatten(), W.flatten()]
            # Residual: T(x) - T_target
            
            def residuals(x):
                U_r = x[:n*m*R].reshape(n*m, R)
                V_r = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_r = x[n*m*R + m*p*R :].reshape(p*n, R)
                T_r = get_tensor(U_r, V_r, W_r)
                return (T_r - np.ones((n,m,p))).flatten()
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = scipy.optimize.least_squares(residuals, x0, method='lm', max_nfev=100)
                if result.success or result.cost < 1e-6:
                    U_sol = result.x[:n*m*R].reshape(n*m, R)
                    V_sol = result.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = result.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    # Check for integer/half-integer
                    # Snap to nearest 0.5
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    if residual_norm(U_snap, V_snap, W_snap) < 1e-9:
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), R
            except Exception:
                pass
        return None

    import scipy.optimize
    
    for R in range(35, 29, -1):
        if time.time() > deadline - 10:
            break
        res = try_rank(R)
        if res:
            best_U, best_V, best_W, best_rank = res
            break

    return {"U": best_U, "V": best_V, "W": best_W}
