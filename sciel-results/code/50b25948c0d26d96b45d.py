import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    def verify_exact(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def try_rank(rank):
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        lr = 0.005
        best_err = float('inf')
        best_U, best_V, best_W = U, V, W
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = s - 1.0
                        for c in range(rank):
                            grad_U[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                            grad_V[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                            grad_W[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Let's just compute error periodically
            if step % 100 == 0:
                err_val = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err_val += (s - 1.0)**2
                if err_val < best_err:
                    best_err = err_val
                    best_U, best_V, best_W = U, V, W
                if best_err < 1e-4:
                    break
        
        if best_err < 1e-4:
            # Try to round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify exactness of rounded
            if verify_exact(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), rank):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
            # If rounding failed, maybe try integers?
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if verify_exact(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
        return None

    result = try_rank(17)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    # Fallback
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
