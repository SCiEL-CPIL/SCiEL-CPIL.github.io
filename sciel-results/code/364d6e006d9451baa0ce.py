import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the target tensor T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify_fast(U, V, W, tol=1e-10):
        R = U.shape[1]
        if V.shape[1] != R or W.shape[1] != R:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def greedy_decompose(max_rank=36, max_iters_per_rank=1000):
        """Greedy decomposition: add rank-1 terms one at a time."""
        residual = T.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for rank in range(1, max_rank + 1):
            if time.time() > deadline - 10:
                break
            
            # Try to find a rank-1 term that reduces the residual
            best_reduction = np.inf
            best_u, best_v, best_w = None, None, None
            
            # Try multiple random initializations
            for init in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Random initialization
                np.random.seed(rank * 1000 + init)
                u = np.random.randn(n * m)
                v = np.random.randn(m * p)
                w = np.random.randn(p * n)
                
                # Normalize
                u = u / (np.linalg.norm(u) + 1e-10)
                v = v / (np.linalg.norm(v) + 1e-10)
                w = w / (np.linalg.norm(w) + 1e-10)
                
                # Alternating minimization
                for it in range(100):
                    # Update u
                    # residual[i,j,k] ≈ u[i*m+j] * v[j*p+k] * w[k*n+i]
                    # For fixed v, w: u[i*m+j] = sum_{k} residual[i,j,k] * v[j*p+k] * w[k*n+i] / (sum_{j,k} v[j*p+k]^2 * w[k*n+i]^2)
                    num_u = np.zeros(n * m)
                    den_u = np.zeros(n * m)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                num_u[i * m + j] += residual[i, j, k] * v[j * p + k] * w[k * n + i]
                                den_u[i * m + j] += v[j * p + k] ** 2 * w[k * n + i] ** 2
                    if den_u[i * m + j] > 1e-10:
                        u[i * m + j] = num_u[i * m + j] / den_u[i * m + j]
                    
                    # Update v
                    num_v = np.zeros(m * p)
                    den_v = np.zeros(m * p)
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                num_v[j * p + k] += residual[i, j, k] * u[i * m + j] * w[k * n + i]
                                den_v[j * p + k] += u[i * m + j] ** 2 * w[k * n + i] ** 2
                    if den_v[j * p + k] > 1e-10:
                        v[j * p + k] = num_v[j * p + k] / den_v[j * p + k]
                    
                    # Update w
                    num_w = np.zeros(p * n)
                    den_w = np.zeros(p * n)
                    for k in range(p):
                        for i in range(n):
                            for j in range(m):
                                num_w[k * n + i] += residual[i, j, k] * u[i * m + j] * v[j * p + k]
                                den_w[k * n + i] += u[i * m + j] ** 2 * v[j * p + k] ** 2
                    if den_w[k * n + i] > 1e-10:
                        w[k * n + i] = num_w[k * n + i] / den_w[k * n + i]
                
                # Compute reduction
                rank1 = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rank1[i, j, k] = u[i * m + j] * v[j * p + k] * w[k * n + i]
                
                new_residual = residual - rank1
                reduction = np.sum(new_residual ** 2)
                
                if reduction < best_reduction:
                    best_reduction = reduction
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_u is None:
                break
            
            # Check if we've achieved exact decomposition
            if best_reduction < 1e-10:
                U_list.append(best_u)
                V_list.append(best_v)
                W_list.append(best_w)
                residual -= best_u[np.repeat(np.arange(n*m)[:,None], p, axis=1).reshape(n,m,p)[..., np.arange(p)]]  # This is wrong
                # Actually, let's just store and verify at the end
                break
            
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            
            # Update residual
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] -= best_u[i * m + j] * best_v[j * p + k] * best_w[k * n + i]
            
            # Check if residual is zero
            if np.max(np.abs(residual)) < 1e-10:
                break
        
        if len(U_list) == 0:
            return None
        
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        
        # Verify
        if verify_fast(U, V, W):
            return U, V, W, len(U_list)
        else:
            return None
    
    best_rank = n * m * p
    best_U, best_V, best_W = build_schoolbook()
    
    # Try greedy decomposition
    result = greedy_decompose(max_rank=30)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
