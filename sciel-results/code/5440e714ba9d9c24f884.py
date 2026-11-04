"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Try to find low-rank decomposition using optimization
    # Start with various ranks and try to minimize error
    
    best_rank = n * m * p  # fallback to schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    def make_schoolbook(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = make_schoolbook(best_rank)
    best_U, best_V, best_W = U_s, V_s, W_s
    
    def compute_error(U, V, W):
        """Compute max absolute error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = max(err, abs(val - 1.0))
        return err
    
    # Try random initialization with gradient descent for various ranks
    np.random.seed(42)
    
    for target_rank in [30, 35, 40, 45, 50, 55, 60, 70, 80]:
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err_val = val - 1.0
                            # d/dU[i*m+j, c] = err_val * V[j*p+k, c] * W[k*n+i, c]
                            dU[i*m+j, :] += err_val * V[j*p+k, :] * W[k*n+i, :]
                            dV[j*p+k, :] += err_val * U[i*m+j, :] * W[k*n+i, :]
                            dW[k*n+i, :] += err_val * U[i*m+j, :] * V[j*p+k, :]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if step % 50 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-8:
                        break
            
            err = compute_error(U, V, W)
            if err < 1e-6 and target_rank < best_rank:
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
    
    # Convert to lists with integer/half-integer rounding if close
    def round_entries(mat):
        """Round entries to nearest half-integer if close."""
        mat_rounded = np.round(mat * 2) / 2
        if np.max(np.abs(mat - mat_rounded)) < 1e-6:
            return mat_rounded.tolist()
        return mat.tolist()
    
    U_list = round_entries(best_U)
    V_list = round_entries(best_V)
    W_list = round_entries(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
