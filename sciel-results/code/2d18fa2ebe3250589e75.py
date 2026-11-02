"""Matrix multiplication tensor decomposition for 3x5 x 5x7."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    deadline = time.time() + 140.0
    margin = 5.0

    # Schoolbook rank: n*m*p = 105
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matmul tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W):
        """Fast verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i * m + j]
                    v_val = V_np[j * p + k]
                    w_val = W_np[k * n + i]
                    s = np.dot(u_val * v_val, w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(rank, max_iters=50):
        """Try to find a rank-r decomposition using randomized search."""
        if rank >= n * m * p:
            return None
        
        # Initialize with random matrices
        for attempt in range(max_iters):
            if time.time() > deadline - margin:
                return None
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to help convergence
            for c in range(rank):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    U[:, c] /= norm_u
                    V[:, c] /= norm_v
                    W[:, c] /= norm_w
            
            # Try a few gradient steps to minimize error
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - margin:
                    break
                
                # Compute error and gradients
                err = 0.0
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target = 1.0
                            u_row = U[i * m + j]
                            v_row = V[j * p + k]
                            w_row = W[k * n + i]
                            
                            # Compute current value and gradients
                            val = 0.0
                            for c in range(rank):
                                val += u_row[c] * v_row[c] * w_row[c]
                            err += (val - target) ** 2
                            
                            for c in range(rank):
                                dv = u_row[c] * w_row[c]
                                dw = u_row[c] * v_row[c]
                                du = v_row[c] * w_row[c]
                                U_grad[i * m + j, c] += 2 * (val - target) * dv
                                V_grad[j * p + k, c] += 2 * (val - target) * dw
                                W_grad[k * n + i, c] += 2 * (val - target) * du
                
                if err < 1e-12:
                    # Found a valid decomposition
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Renormalize
                for c in range(rank):
                    norm_u = np.linalg.norm(U[:, c])
                    norm_v = np.linalg.norm(V[:, c])
                    norm_w = np.linalg.norm(W[:, c])
                    if norm_u > 1e-10:
                        U[:, c] /= norm_u
                    if norm_v > 1e-10:
                        V[:, c] /= norm_v
                    if norm_w > 1e-10:
                        W[:, c] /= norm_w
        
        return None

    # Try to find lower-rank decompositions
    best = None
    
    # Try ranks from 104 down to 100 (limited search)
    for rank in range(104, 99, -1):
        if time.time() > deadline - margin:
            break
        result = try_rank(rank, max_iters=20)
        if result is not None:
            best = result
            break
    
    if best is not None:
        return best
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END

