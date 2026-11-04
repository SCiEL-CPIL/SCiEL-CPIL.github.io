"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check if the decomposition is exact
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _gradient_descent(U, V, W, n, m, p, max_iter=200, lr=0.01):
    """
    Perform gradient descent to minimize the error of the tensor decomposition.
    U, V, W are numpy arrays.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Precompute indices for efficiency? 
    # Loop over i, j, k is 42 iterations. Rank is ~30-40.
    # Total operations per step: 42 * 40 * 3 ~ 5000. Very fast.
    
    for _ in range(max_iter):
        # Compute current values and gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # We need to compute the residual for each (i,j,k)
        # res[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
        
        # To avoid storing res for all i,j,k explicitly in a loop, we can just loop.
        # Since n,m,p are small, explicit loop is fine.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    # Compute dot product for this (i,j,k)
                    # val = sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    # This is not a simple dot product because of the 3-way sum.
                    # We can compute it as:
                    # temp = U[idx_u, :] * V[idx_v, :]
                    # val = sum(temp * W[idx_w, :])
                    
                    temp = U[idx_u, :] * V[idx_v, :]
                    val = np.sum(temp * W[idx_w, :])
                    res = val - 1.0
                    
                    # Gradients
                    # dL/dU[idx_u, c] = 2 * res * V[idx_v, c] * W[idx_w, c]
                    # dL/dV[idx_v, c] = 2 * res * U[idx_u, c] * W[idx_w, c]
                    # dL/dW[idx_w, c] = 2 * res * U[idx_u, c] * V[idx_v, c]
                    
                    grad_U[idx_u, :] += 2.0 * res * (V[idx_v, :] * W[idx_w, :])
                    grad_V[idx_v, :] += 2.0 * res * (U[idx_u, :] * W[idx_w, :])
                    grad_W[idx_w, :] += 2.0 * res * (U[idx_u, :] * V[idx_v, :])
                    
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Optional: Check convergence
        # if np.max(np.abs(grad_U)) < 1e-8: break
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Prepare schoolbook fallback
    U_fb = np.zeros((n*m, schoolbook_rank))
    V_fb = np.zeros((m*p, schoolbook_rank))
    W_fb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # We try ranks from 31 to 41
    # We use random initialization and gradient descent
    # Then we snap to nearest 0.5 and verify
    
    for target_rank in range(31, schoolbook_rank):
        if time.time() > deadline - 15:
            break
            
        # Try multiple random initializations
        # Higher ranks are easier to find, so we might want to start lower 
        # but if we fail, we go higher.
        # However, finding rank 31 is hard. Rank 41 is easy.
        # Let's try a few attempts for each rank.
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(target_rank * 100 + attempt)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Optimize
            U_opt, V_opt, W_opt = _gradient_descent(U, V, W, n, m, p, max_iter=100, lr=0.05)
            
            # Snap to half-integers
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            # Verify
            if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
