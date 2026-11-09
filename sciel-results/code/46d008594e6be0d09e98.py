"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    sb_rank = n * m * p  # 42
    
    def make_schoolbook():
        rank = sb_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = sb_rank
    
    # Try to find a lower rank decomposition using random initialization + gradient descent
    # The tensor T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor for 2x3 * 3x7)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For small ranks, we can try to solve this exactly using random search + local optimization
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=2000, seed=0):
        """Try to find a decomposition of rank r using random init + gradient descent."""
        if time.time() > deadline - 5.0:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Gradient descent to minimize sum of (T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Compute loss and gradients
            loss = 0.0
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # val = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = val - 1.0
                        loss += err * err
                        # Gradients:
                        # d/dU[i*m+j,c] = 2*err * V[j*p+k,c] * W[k*n+i,c]
                        # d/dV[j*p+k,c] = 2*err * U[i*m+j,c] * W[k*n+i,c]
                        # d/dW[k*n+i,c] = 2*err * U[i*m+j,c] * V[j*p+k,c]
                        gU[i*m+j] += 2.0 * err * V[j*p+k] * W[k*n+i]
                        gV[j*p+k] += 2.0 * err * U[i*m+j] * W[k*n+i]
                        gW[k*n+i] += 2.0 * err * U[i*m+j] * V[j*p+k]
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-10:
                break
            
            # Update
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
            # Adaptive learning rate
            if it % 100 == 99:
                lr = max(0.001, lr * 0.9)
        
        if best_loss < 1e-8:
            # Convert to list of lists with half-integer entries if possible
            U_list = [[round(x * 2) / 2 for x in row] for row in best_U]
            V_list = [[round(x * 2) / 2 for x in row] for row in best_V]
            W_list = [[round(x * 2) / 2 for x in row] for row in best_W]
            
            # Verify
            if verify(U_list, V_list, W_list, tol=1e-4):
                return {"U": U_list, "V": V_list, "W": W_list}
            else:
                # Try without rounding
                U_list = [[float(x) for x in row] for row in best_U]
                V_list = [[float(x) for x in row] for row in best_V]
                W_list = [[float(x) for x in row] for row in best_W]
                if verify(U_list, V_list, W_list, tol=1e-6):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Also rank >= n + m + p - 2 = 2 + 3 + 7 - 2 = 10 (weak bound)
    # For matrix multiplication, known results suggest rank might be around 21-25
    
    min_possible_rank = max(n*m, m*p, p*n)  # 21
    
    for r in range(min_possible_rank, sb_rank):
        if time.time() > deadline - 15.0:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(10):
            if time.time() > deadline - 15.0:
                break
            
            result = try_rank(r, max_iters=1500, seed=seed)
            if result is not None:
                best_result = result
                best_rank = r
                found = True
                break
        
        if found:
            # Found a valid decomposition at rank r, try to go lower
            continue
        else:
            # If we can't find at rank r, likely can't find at higher ranks either
            # (but this isn't guaranteed, so we continue)
            pass
    
    return best_result
# EVOLVE-BLOCK-END
