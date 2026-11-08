import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    r_school = n * m * p  # 56
    
    # Build the schoolbook decomposition (rank 56)
    U = np.zeros((n * m, r_school))
    V = np.zeros((m * p, r_school))
    W = np.zeros((p * n, r_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Use gradient-based optimization to find a rank-r decomposition
    # that exactly (or nearly exactly) matches the tensor.
    # Since we need exact integer/half-integer entries, we'll try to find
    # a solution and then round to nearest half-integer and verify.
    
    start_time = time.time()
    max_time = 100  # Leave margin before 120s limit
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = r_school
    
    # Try ranks from 30 to r_school-1
    for r in range(30, r_school):
        if time.time() - start_time > max_time:
            break
        
        if r >= best_rank:
            continue
        
        # Try a few restarts
        for restart in range(3):
            if time.time() - start_time > max_time:
                break
            
            # Random initialization
            np.random.seed(42 + r * 100 + restart)
            U_r = np.random.randn(n * m, r) * 0.5
            V_r = np.random.randn(m * p, r) * 0.5
            W_r = np.random.randn(p * n, r) * 0.5
            
            # Gradient descent to minimize sum_{i,j,k} (1 - sum_r U[i*m+j,r]V[j*p+k,r]W[k*n+i,r])^2
            lr = 0.05
            for step in range(200):
                if time.time() - start_time > max_time:
                    break
                
                # Compute reconstruction
                Rec = np.zeros((n, m, p))
                for rr in range(r):
                    U_reshaped = U_r[:, rr].reshape(n, m)
                    V_reshaped = V_r[:, rr].reshape(m, p)
                    W_reshaped = W_r[:, rr].reshape(p, n)
                    Rec_r = np.einsum('ij,jk,ki->ijk', U_reshaped, V_reshaped, W_reshaped)
                    Rec += Rec_r
                
                # Error
                Error = 1.0 - Rec  # shape (n, m, p)
                
                # Check if we're close enough
                max_err = np.max(np.abs(Error))
                if max_err < 1e-10:
                    break
                
                # Compute gradients
                # dL/dU[i*m+j, rr] = -2 * sum_{j,k} Error[i,j,k] * V[j*p+k, rr] * W[k*n+i, rr]
                # dL/dV[j*p+k, rr] = -2 * sum_{i,k} Error[i,j,k] * U[i*m+j, rr] * W[k*n+i, rr]
                # dL/dW[k*n+i, rr] = -2 * sum_{i,j} Error[i,j,k] * U[i*m+j, rr] * V[j*p+k, rr]
                
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                
                for rr in range(r):
                    U_reshaped = U_r[:, rr].reshape(n, m)
                    V_reshaped = V_r[:, rr].reshape(m, p)
                    W_reshaped = W_r[:, rr].reshape(p, n)
                    
                    # Gradient for U
                    # dU[i*m+j, rr] = -2 * sum_{k} Error[i,j,k] * V[j*p+k, rr] * W[k*n+i, rr]
                    for i in range(n):
                        for j in range(m):
                            s = 0.0
                            for k in range(p):
                                s += Error[i, j, k] * V[j * p + k, rr] * W[k * n + i, rr]
                            dU[i * m + j, rr] = -2.0 * s
                    
                    # Gradient for V
                    # dV[j*p+k, rr] = -2 * sum_{i} Error[i,j,k] * U[i*m+j, rr] * W[k*n+i, rr]
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for i in range(n):
                                s += Error[i, j, k] * U[i * m + j, rr] * W[k * n + i, rr]
                            dV[j * p + k, rr] = -2.0 * s
                    
                    # Gradient for W
                    # dW[k*n+i, rr] = -2 * sum_{j} Error[i,j,k] * U[i*m+j, rr] * V[j*p+k, rr]
                    for k in range(p):
                        for i in range(n):
                            s = 0.0
                            for j in range(m):
                                s += Error[i, j, k] * U[i * m + j, rr] * V[j * p + k, rr]
                            dW[k * n + i, rr] = -2.0 * s
                
                # Update
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
            
            # After gradient descent, try rounding to nearest half-integer
            U_rounded = np.round(U_r * 2) / 2
            V_rounded = np.round(V_r * 2) / 2
            W_rounded = np.round(W_r * 2) / 2
            
            # Verify the rounded solution
            Rec_check = np.zeros((n, m, p))
            for rr in range(r):
                U_reshaped = U_rounded[:, rr].reshape(n, m)
                V_reshaped = V_rounded[:, rr].reshape(m, p)
                W_reshaped = W_rounded[:, rr].reshape(p, n)
                Rec_r = np.einsum('ij,jk,ki->ijk', U_reshaped, V_reshaped, W_reshaped)
                Rec_check += Rec_r
            
            max_err = np.max(np.abs(Rec_check - 1.0))
            if max_err < 1e-10:
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = r
                break  # Found a valid solution for this rank, move to next lower rank
    
    return best_result
