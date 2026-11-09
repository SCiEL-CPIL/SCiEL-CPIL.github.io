"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor R[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1"""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = np.zeros((n, m, p))
    for c in range(rank):
        # u_c has shape (n*m), v_c has shape (m*p), w_c has shape (p*n)
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    R -= 1.0
    return R

def _try_rank(r, n, m, p, seed=0, max_iter=20000, lr_init=0.01):
    """Try to find a rank-r decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    
    # Compute residual
    R = _compute_residual(U, V, W, n, m, p)
    loss = np.sum(R * R)
    
    if loss < 1e-10:
        return U, V, W, True
    
    lr = lr_init
    for it in range(max_iter):
        # Compute gradients
        # d/dU[i,j,c] = sum_k R[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
        # d/dV[j,k,c] = sum_i R[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
        # d/dW[k,i,c] = sum_j R[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_ijk = R[i, j, k]
                        dU[i * m + j, c] += r_ijk * V[j * p + k, c] * W[k * n + i, c]
                        dV[j * p + k, c] += r_ijk * U[i * m + j, c] * W[k * n + i, c]
                        dW[k * n + i, c] += r_ijk * U[i * m + j, c] * V[j * p + k, c]
        
        # Gradient step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Recompute residual
        R = _compute_residual(U, V, W, n, m, p)
        new_loss = np.sum(R * R)
        
        if new_loss < loss:
            loss = new_loss
            if loss < 1e-10:
                return U, V, W, True
        else:
            lr *= 0.9
        
        # Early stopping if stuck
        if it % 1000 == 0 and loss > 1e-5:
            break
    
    return U, V, W, loss < 1e-10

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition."""
    R = _compute_residual(U, V, W, n, m, p)
    return np.max(np.abs(R)) < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find low-rank decompositions
    # Known bounds suggest R(3,3,5) is around 14-16
    best_result = None
    
    for r in [14, 15, 16, 17, 18, 20, 25, 30]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            U, V, W, success = _try_rank(r, n, m, p, seed=seed, max_iter=5000, lr_init=0.005)
            
            if success and _verify_decomposition(U, V, W, n, m, p):
                # Convert to lists of integers/half-integers if possible
                U_list = [[round(val, 10) for val in row] for row in U]
                V_list = [[round(val, 10) for val in row] for row in V]
                W_list = [[round(val, 10) for val in row] for row in W]
                
                # Clean up near-zero and near-integer values
                for matrix in [U_list, V_list, W_list]:
                    for row in matrix:
                        for idx, val in enumerate(row):
                            if abs(val) < 1e-8:
                                row[idx] = 0
                            elif abs(val - round(val)) < 1e-8:
                                row[idx] = round(val)
                            elif abs(val * 2 - round(val * 2)) < 1e-8:
                                row[idx] = val  # half-integer
                            
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                found = True
                break
        
        if found:
            break
    
    # Fallback to schoolbook
    if best_result is None:
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
