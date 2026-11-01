import scipy
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def check_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Target T[i, j, k] = 1
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Efficient check using einsum or loops
    # T_calc[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute T_calc
    # T_calc[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    # This is a tensor contraction.
    
    T_calc = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                T_calc[i, j, k] = s
                # Early exit if large error?
    
    # Check if T_calc is all 1s
    if np.allclose(T_calc, 1.0, atol=1e-9):
        return True
    return False

def snap_to_half_integer(matrix):
    # Snap to nearest 0.5
    return np.round(matrix * 2) / 2.0

def als_decompose(rank, max_iters=100, tol=1e-8):
    n, m, p = DIMENSIONS
    # Init
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    prev_err = float('inf')
    
    for it in range(max_iters):
        # Update U
        # U[i, j, r] needs to satisfy sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # This is a linear system for each (i,j) if we fix V, W?
        # Actually, for fixed V, W, the term for U[i,j,:] is:
        # sum_k (V[j,k,:] * W[k,i,:]) * U[i,j,:] = 1? No.
        # The sum is over r.
        # Let's define G[i, j, k] = sum_r V[j, k, r] W[k, i, r]
        # Then sum_r U[i, j, r] * (V[j, k, r] W[k, i, r]) is not a simple dot product because the index k varies.
        # Wait, the term is sum_r U[i, j, r] * V[j, k, r] * W[k, i, r].
        # For a fixed i, j, and target value 1 for ALL k?
        # No, the equation is: For each i, j, k, sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1.
        # This couples all k for a given i, j.
        
        # This is not a simple ALS because the indices are coupled in the target.
        # Standard ALS works if the tensor is separable or if we minimize Frobenius norm.
        # Minimize || T - sum U_r V_r W_r ||^2.
        # This is a standard tensor decomposition problem.
        
        # We can use the gradient of the Frobenius norm.
        # Or use the fact that for fixed V, W, the optimal U minimizes:
        # sum_{i,j,k} (1 - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
        
        # Let's use a simple gradient descent on the flattened parameters?
        # Or use scipy.optimize?
        
        # Given the small size (9x23), we can use a numerical solver.
        # But ALS is faster.
        # Let's try a simpler approach: Random restarts with snap.
        
        pass

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try ranks from 23 upwards
    for rank in [23, 24, 25, 26, 27]:
        if time.time() - start_time > 100:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > 105:
                break
                
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Optimize
            # Use a simple iterative solver
            # Since the target is all-ones, and we want integer/half-integer solutions,
            # we can just snap and check? No, random snap won't work.
            
            # We need to solve for the coefficients.
            # This is a non-linear least squares problem.
            # Let's use scipy.optimize.least_squares on the flattened vector.
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_curr = x[:n*m*rank].reshape(n*m, rank)
                V_curr = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_curr = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                U_3d = U_curr.reshape(n, m, rank)
                V_3d = V_curr.reshape(m, p, rank)
                W_3d = W_curr.reshape(p, n, rank)
                
                res = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res[i, j, k] = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :]) - 1.0
                return res.flatten()
            
            try:
                result = scipy.optimize.least_squares(residuals, x0, method='lm')
                x_sol = result.x
                U_sol = x_sol[:n*m*rank].reshape(n*m, rank)
                V_sol = x_sol[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = x_sol[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                # Snap
                U_snap = snap_to_half_integer(U_sol)
                V_snap = snap_to_half_integer(V_sol)
                W_snap = snap_to_half_integer(W_sol)
                
                if check_decomposition(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            except Exception:
                continue
                
    # Fallback to schoolbook
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
