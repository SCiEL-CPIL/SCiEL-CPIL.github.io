"""Rewritten program for 2x5 x 5x5 matrix multiplication tensor rank minimization."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    The tensor T_{i,j,k} = delta_{i,k} (where i in [0,n), j in [0,m), k in [0,p))
    represents C[i,k] = sum_j A[i,j] * B[j,k].
    
    We seek rank r such that:
      T_{i,j,k} = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Returns dict with 'U', 'V', 'W' as lists of lists.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook rank
    rank_school = n * m * p  # 50
    
    def make_schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_school
    
    def tensor_residual(U, V, W):
        """Compute the Frobenius norm of the residual tensor."""
        r = U.shape[1]
        # Residual_{i,j,k} = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - T_{i,j,k}
        # T_{i,j,k} = 1 if i==k else 0
        residual = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        # Subtract the target tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i == k:
                        residual[i, j, k] -= 1
        return np.linalg.norm(residual)
    
    def tensor_residual_fast(U, V, W):
        """Faster residual computation using vectorization."""
        r = U.shape[1]
        residual = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # For each c, contribution is outer product structure
            # residual[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                ui = u[i, :]  # length m
                wi = w[:, i]  # length p
                # For fixed i: residual[i,j,k] += ui[j] * v[j,k] * wi[k]
                # = (ui * w_i^T) contracted with V
                # v[j,k] * ui[j] * wi[k] = (ui @ V) elementwise * wi^T? No.
                # Let's just do it directly but vectorized over j,k
                # tmp[j,k] = ui[j] * v[j,k] * wi[k]
                tmp = np.outer(ui, wi) * v  # (m, p)
                residual[i, :, :] += tmp
        # Subtract target
        for i in range(n):
            for j in range(m):
                if i < p:  # k=i is valid
                    residual[i, j, i] -= 1
        return np.linalg.norm(residual)
    
    def gradient_descent_step(U, V, W, lr=0.01, steps=500):
        """Perform gradient descent to reduce rank by zeroing small columns."""
        r = U.shape[1]
        U = U.astype(float).copy()
        V = V.astype(float).copy()
        W = W.astype(float).copy()
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            # Compute gradients
            # d/dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c] * residual[i,j,k] summed over k
            # This is complex; use numerical gradient for small cases or analytic
            
            # Use analytic gradient
            r_curr = U.shape[1]
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # Compute residual tensor
            R = np.zeros((n, m, p))
            for c in range(r_curr):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                for i in range(n):
                    tmp = np.outer(u[i, :], w[:, i]) * v
                    R[i, :, :] += tmp
            for i in range(n):
                for j in range(m):
                    if i < p:
                        R[i, j, i] -= 1
            
            # Gradients
            for c in range(r_curr):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            Rijk = R[i, j, k]
                            dU[i*m+j, c] += Rijk * v[j*p+k, c] * w[k*n+i, c]
                            dV[j*p+k, c] += Rijk * u[i*m+j, c] * w[k*n+i, c]
                            dW[k*n+i, c] += Rijk * u[i*m+j, c] * v[j*p+k, c]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return U, V, W
    
    # Try to reduce rank by starting from schoolbook and using optimization
    # Strategy: try various lower ranks with random initialization + gradient descent
    
    target_ranks = [30, 35, 40, 45, 48, 49]  # Try to beat 50
    
    for target_r in target_ranks:
        if time.time() > deadline - 20:
            break
        
        best_res = float('inf')
        best_U_r, best_V_r, best_W_r = None, None, None
        
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U_r = np.random.randn(n*m, target_r) * 0.5
            V_r = np.random.randn(m*p, target_r) * 0.5
            W_r = np.random.randn(p*n, target_r) * 0.5
            
            # Run gradient descent
            U_r, V_r, W_r = gradient_descent_step(U_r, V_r, W_r, lr=0.005, steps=300)
            
            res = tensor_residual_fast(U_r, V_r, W_r)
            if res < best_res:
                best_res = res
                best_U_r, best_V_r, best_W_r = U_r.copy(), V_r.copy(), W_r.copy()
            
            if best_res < 1e-8:
                break
        
        if best_res < 1e-8 and best_U_r is not None:
            # Verify exactness with higher precision
            U_r, V_r, W_r = best_U_r, best_V_r, best_W_r
            # Check if we can round to integers/half-integers
            # First verify the decomposition is exact
            verify_res = tensor_residual_fast(U_r, V_r, W_r)
            if verify_res < 1e-6:
                # Try to find integer/half-integer solution
                # Round and check
                U_rounded = np.round(U_r * 2) / 2
                V_rounded = np.round(V_r * 2) / 2
                W_rounded = np.round(W_r * 2) / 2
                res_rounded = tensor_residual_fast(U_rounded, V_rounded, W_rounded)
                if res_rounded < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_r
                    break
                else:
                    # Keep the floating point solution if it's very close
                    # But we need integer/half-integer entries
                    # Try local search for exact half-integer solution
                    # For now, skip if can't round
                    pass
    
    # If no better solution found, return schoolbook
    if best_rank == rank_school:
        U, V, W = U0, V0, W0
    else:
        U, V, W = best_U, best_V, best_W
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Ensure it's integer or half-integer
                val_rounded = round(val * 2) / 2
                if abs(val_rounded) < 1e-10:
                    val_rounded = 0
                # Represent as int if integer, else float
                if abs(val_rounded - round(val_rounded)) < 1e-10:
                    row_list.append(int(round(val_rounded)))
                else:
                    row_list.append(val_rounded)
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
