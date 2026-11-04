"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is exact for all tensor entries."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def compute_loss_and_grads(U, V, W, n, m, p):
        """Compute loss and gradients for all parameters."""
        rank = U.shape[1]
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Precompute residuals
        # residual[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m+j, :]
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    residuals[i,j,k] = np.dot(u_row, v_row * w_row) - 1.0
                    loss += residuals[i,j,k] ** 2
        
        # Gradients
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    r = residuals[i,j,k]
                    # d/dU[i,j,r'] = 2*r*V[j,k,r']*W[k,i,r']
                    grad_U[i*m+j, :] += 2.0 * r * v_row * w_row
                    # d/dV[j,k,r'] = 2*r*U[i,j,r']*W[k,i,r']
                    grad_V[j*p+k, :] += 2.0 * r * u_row * w_row
                    # d/dW[k,i,r'] = 2*r*U[i,j,r']*V[j,k,r']
                    grad_W[k*n+i, :] += 2.0 * r * u_row * v_row
        
        return loss, grad_U, grad_V, grad_W
    
    # Try to find lower-rank decompositions using gradient descent
    target_ranks = [60, 70, 80, 90, 100, 110]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 1000 + target_rank)
            
            U_try = np.random.randn(n*m, target_rank) * 0.5
            V_try = np.random.randn(m*p, target_rank) * 0.5
            W_try = np.random.randn(p*n, target_rank) * 0.5
            
            # Initialize with some structure
            for r in range(target_rank):
                idx = r % (n*m)
                U_try[idx, r] += 1.0
                V_try[idx % (m*p), r] += 1.0
                W_try[idx % (p*n), r] += 1.0
            
            loss, gU, gV, gW = compute_loss_and_grads(U_try, V_try, W_try, n, m, p)
            best_loss = loss
            lr = 0.01
            max_steps = 5000
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                loss, gU, gV, gW = compute_loss_and_grads(U_try, V_try, W_try, n, m, p)
                
                if loss < 1e-14:
                    if verify_decomposition(U_try, V_try, W_try, n, m, p):
                        best_U = U_try.copy()
                        best_V = V_try.copy()
                        best_W = W_try.copy()
                        best_rank = target_rank
                        break
                
                # Adaptive learning rate
                if loss < best_loss * 0.99:
                    best_loss = loss
                    lr = min(lr * 1.05, 0.1)
                else:
                    lr *= 0.95
                
                # Gradient step
                U_try -= lr * gU
                V_try -= lr * gV
                W_try -= lr * gW
            
            if best_rank == target_rank:
                break
    
    # Convert to list of lists with rounding
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    # Clean up near-zero and near-integer/half-integer values
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
                elif abs(v - round(v*2)/2) < 1e-10:
                    row[idx] = round(v*2)/2
        return lst
    
    U_out = clean_list(to_list(best_U))
    V_out = clean_list(to_list(best_V))
    W_out = clean_list(to_list(best_W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
