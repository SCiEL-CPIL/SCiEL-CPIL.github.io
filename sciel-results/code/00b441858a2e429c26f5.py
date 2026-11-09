import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, atol=1e-7):
        n, m, p = DIMENSIONS
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        if U_np.ndim == 1:
            U_np = U_np.reshape(-1, 1)
        if V_np.ndim == 1:
            V_np = V_np.reshape(-1, 1)
        if W_np.ndim == 1:
            W_np = W_np.reshape(-1, 1)
        rank = U_np.shape[1]
        if V_np.shape[1] != rank or W_np.shape[1] != rank:
            return False
        if U_np.shape[0] != n*m or V_np.shape[0] != m*p or W_np.shape[0] != p*n:
            return False
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_mat = U_np[:, r].reshape(n, m)
            v_mat = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            tensor += u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_mat.T[:, np.newaxis, :]
        expected = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(tensor, expected, atol=atol)
    
    def tensor_residual(x, rank):
        """Compute sum of (sum_r u[i,j,r]*v[j,k,r]*w[k,i,r] - 1)^2 over all i,j,k"""
        n, m, p = DIMENSIONS
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
        W = x[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            tensor += u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_mat.T[:, np.newaxis, :]
        residual = tensor - 1.0
        return np.sum(residual**2)
    
    def try_rank(rank, max_iters=30, seed=None):
        """Try to find a rank-rank decomposition using random restarts + gradient descent"""
        if time.time() > deadline - 5:
            return None
        
        n, m, p = DIMENSIONS
        dim = 3 * rank * n * m  # U is n*m*rank, V is m*p*rank, W is p*n*rank
        # Actually: U has n*m*rank entries, V has m*p*rank, W has p*n*rank
        total_dim = n*m*rank + m*p*rank + p*n*rank
        
        best_result = None
        best_score = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            s = seed if seed is not None else (it * 7919 + 42)
            rng = np.random.RandomState(s)
            
            # Random initialization
            scale = 1.0 / np.sqrt(rank)
            x = rng.randn(total_dim) * scale
            
            # Try to optimize
            for opt_iter in range(100):
                if time.time() > deadline - 5:
                    break
                
                res = tensor_residual(x, rank)
                if res < 1e-10:
                    # Found solution
                    U = x[:n*m*rank].reshape(n*m, rank)
                    V = x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                    W = x[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
                    return (U.tolist(), V.tolist(), W.tolist())
                
                if res < best_score:
                    best_score = res
                    best_result = x.copy()
                
                # Compute gradient numerically or analytically
                # Use analytical gradient for speed
                # dL/dU[i,j,r] = 2 * sum_{k} (T[i,j,k]-1) * V[j,k,r] * W[k,i,r]
                # dL/dV[j,k,r] = 2 * sum_{i} (T[i,j,k]-1) * U[i,j,r] * W[k,i,r]
                # dL/dW[k,i,r] = 2 * sum_{j} (T[i,j,k]-1) * U[i,j,r] * V[j,k,r]
                
                U = x[:n*m*rank].reshape(n*m, rank)
                V = x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W = x[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
                
                tensor = np.zeros((n, m, p), dtype=np.float64)
                for r in range(rank):
                    u_mat = U[:, r].reshape(n, m)
                    v_mat = V[:, r].reshape(m, p)
                    w_mat = W[:, r].reshape(p, n)
                    tensor += u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_mat.T[:, np.newaxis, :]
                
                err = tensor - 1.0
                
                # Gradient for U
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(rank):
                    u_mat = U[:, r].reshape(n, m)
                    v_mat = V[:, r].reshape(m, p)
                    w_mat = W[:, r].reshape(p, n)
                    
                    # dL/dU[i,j,r] = 2 * sum_k err[i,j,k] * v_mat[j,k] * w_mat[k,i]
                    for i in range(n):
                        for j in range(m):
                            s = 0.0
                            for k in range(p):
                                s += err[i,j,k] * v_mat[j,k] * w_mat[k,i]
                            grad_U[i*m+j, r] = 2.0 * s
                    
                    # dL/dV[j,k,r] = 2 * sum_i err[i,j,k] * u_mat[i,j] * w_mat[k,i]
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for i in range(n):
                                s += err[i,j,k] * u_mat[i,j] * w_mat[k,i]
                            grad_V[j*p+k, r] = 2.0 * s
                    
                    # dL/dW[k,i,r] = 2 * sum_j err[i,j,k] * u_mat[i,j] * v_mat[j,k]
                    for k in range(p):
                        for i in range(n):
                            s = 0.0
                            for j in range(m):
                                s += err[i,j,k] * u_mat[i,j] * v_mat[j,k]
                            grad_W[k*n+i, r] = 2.0 * s
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                # Adaptive step size
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-15:
                    break
                lr = 0.1 * (1.0 / np.sqrt(rank)) / max(grad_norm, 1e-10)
                lr = min(lr, 0.5)
                
                x = x - lr * grad
                
                # Check convergence
                res_new = tensor_residual(x, rank)
                if abs(res_new - res) < 1e-12:
                    break
                if res_new < 1e-10:
                    U = x[:n*m*rank].reshape(n*m, rank)
                    V = x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                    W = x[2*n*m*rank:3*n*m*rank].reshape(p*n, rank)
                    return (U.tolist(), V.tolist(), W.tolist())
        
        return None
    
    # Try ranks from low to high
    # Lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Also rank >= m*p = 24 for this type of multiplication
    # Known: rank(2x4x6) could be as low as ~18-20
    
    for rank in range(18, 25):
        if time.time() > deadline - 10:
            break
        result = try_rank(rank, max_iters=20)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook()
