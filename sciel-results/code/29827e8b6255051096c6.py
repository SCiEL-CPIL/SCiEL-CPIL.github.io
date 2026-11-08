import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    r_school = n * m * p  # 56
    
    # Build the schoolbook decomposition (rank 56) as fallback
    U_school = np.zeros((n * m, r_school))
    V_school = np.zeros((m * p, r_school))
    W_school = np.zeros((p * n, r_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = r_school
    
    start_time = time.time()
    max_time = 110  # Leave margin before 120s limit
    
    # Try to find a lower-rank decomposition
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def compute_reconstruction(U_r, V_r, W_r, r):
        """Compute the reconstructed tensor."""
        Rec = np.zeros((n, m, p))
        for rr in range(r):
            U_reshaped = U_r[:, rr].reshape(n, m)
            V_reshaped = V_r[:, rr].reshape(m, p)
            W_reshaped = W_r[:, rr].reshape(p, n)
            Rec_r = np.einsum('ij,jk,ki->ijk', U_reshaped, V_reshaped, W_reshaped)
            Rec += Rec_r
        return Rec
    
    def compute_gradients(Error, U_r, V_r, W_r, r):
        """Compute gradients for the loss function."""
        dU = np.zeros_like(U_r)
        dV = np.zeros_like(V_r)
        dW = np.zeros_like(W_r)
        
        for rr in range(r):
            U_reshaped = U_r[:, rr].reshape(n, m)
            V_reshaped = V_r[:, rr].reshape(m, p)
            W_reshaped = W_r[:, rr].reshape(p, n)
            
            # Gradient for U: dU[i*m+j, rr] = -2 * sum_k Error[i,j,k] * V[j*p+k, rr] * W[k*n+i, rr]
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for k in range(p):
                        s += Error[i, j, k] * V[j * p + k, rr] * W[k * n + i, rr]
                    dU[i * m + j, rr] = -2.0 * s
            
            # Gradient for V: dV[j*p+k, rr] = -2 * sum_i Error[i,j,k] * U[i*m+j, rr] * W[k*n+i, rr]
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for i in range(n):
                        s += Error[i, j, k] * U[i * m + j, rr] * W[k * n + i, rr]
                    dV[j * p + k, rr] = -2.0 * s
            
            # Gradient for W: dW[k*n+i, rr] = -2 * sum_j Error[i,j,k] * U[i*m+j, rr] * V[j*p+k, rr]
            for k in range(p):
                for i in range(n):
                    s = 0.0
                    for j in range(m):
                        s += Error[i, j, k] * U[i * m + j, rr] * V[j * p + k, rr]
                    dW[k * n + i, rr] = -2.0 * s
        
        return dU, dV, dW
    
    # Try ranks from 35 to r_school-1
    for r in range(35, r_school):
        if time.time() - start_time > max_time:
            break
        
        if r >= best_rank:
            continue
        
        # Try a few restarts
        for restart in range(5):
            if time.time() - start_time > max_time:
                break
            
            # Random initialization with different scales
            np.random.seed(42 + r * 100 + restart * 17)
            scale = 0.5 + 0.1 * restart
            U_r = np.random.randn(n * m, r) * scale
            V_r = np.random.randn(m * p, r) * scale
            W_r = np.random.randn(p * n, r) * scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.02
            best_loss = np.inf
            stuck_counter = 0
            
            for step in range(300):
                if time.time() - start_time > max_time:
                    break
                
                # Compute reconstruction
                Rec = compute_reconstruction(U_r, V_r, W_r, r)
                
                # Error
                Error = 1.0 - Rec  # shape (n, m, p)
                
                # Check if we're close enough
                max_err = np.max(np.abs(Error))
                if max_err < 1e-10:
                    break
                
                # Compute gradients
                dU, dV, dW = compute_gradients(Error, U_r, V_r, W_r, r)
                
                # Update
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
                
                # Adaptive learning rate: reduce if loss increases
                new_err = np.max(np.abs(1.0 - compute_reconstruction(U_r, V_r, W_r, r)))
                if new_err > max_err * 1.1:
                    lr *= 0.5
                    # Revert
                    U_r += lr * dU
                    V_r += lr * dV
                    W_r += lr * dW
                else:
                    if new_err < best_loss:
                        best_loss = new_err
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                        if stuck_counter > 50:
                            lr *= 0.95
            
            # After gradient descent, try rounding to nearest half-integer
            U_rounded = np.round(U_r * 2) / 2
            V_rounded = np.round(V_r * 2) / 2
            W_rounded = np.round(W_r * 2) / 2
            
            # Verify the rounded solution
            Rec_check = compute_reconstruction(U_rounded, V_rounded, W_rounded, r)
            max_err = np.max(np.abs(Rec_check - 1.0))
            if max_err < 1e-10:
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = r
                break  # Found a valid solution for this rank, move to next lower rank
    
    return best_result
