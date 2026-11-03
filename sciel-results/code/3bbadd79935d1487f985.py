import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0
    
    # Precompute indices for non-zero entries and some zero entries for constraint
    # Non-zero: i, j, k -> a=i*m+j, b=j*p+k, c=k*n+i, val=1
    # Zero: i, j, j', k with j!=j' -> a=i*m+j, b=j'*p+k, c=k*n+i, val=0
    # Also need to ensure other c are 0? 
    # The decomposition defines C_{i,k} for all i,k.
    # The condition is sum_r U[i,j,r]V[j',k,r]W[k,i,r] = delta(j,j').
    # This covers all i,j,j',k.
    
    indices = []
    targets = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    a = i * m + j
                    b = jp * p + k
                    c = k * n + i
                    indices.append((a, b, c))
                    targets.append(1.0 if j == jp else 0.0)
    
    indices = np.array(indices)
    targets = np.array(targets)
    
    def get_schoolbook():
        rank = n * m * p
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
    
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Vectorized check
        # Expand indices
        # This is slow in python loops, so we do it in numpy
        # But R is small, so we can loop r
        # Sum_r U[a,r] V[b,r] W[c,r]
        
        # For speed, we can check a subset or all.
        # All is 6*30*20 = 3600 entries.
        # For each entry, sum over R.
        
        for idx in range(len(indices)):
            a, b, c = indices[idx]
            target = targets[idx]
            s = 0.0
            for r in range(rank):
                s += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
            if abs(s - target) > 1e-5:
                return False
        return True

    def round_to_half_int(arr):
        return np.round(arr * 2) / 2

    # Try rank 55
    for rank in [55, 56, 57]:
        if time.time() > deadline:
            break
            
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Simple Gradient Descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                    
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # This is O(N * R) where N is number of entries
                # N=3600, R=55. 3600*55 = 200k. Fast.
                for idx in range(len(indices)):
                    a, b, c = indices[idx]
                    target = targets[idx]
                    
                    # s = sum_r U[a,r]V[b,r]W[c,r]
                    # ds/dr = V[b,r]W[c,r] for U, etc.
                    
                    u_row = U[a, :]
                    v_row = V[b, :]
                    w_row = W[c, :]
                    
                    s = np.sum(u_row * v_row * w_row)
                    err = s - target
                    
                    # grad_U[a, r] += err * V[b,r] * W[c,r]
                    grad_U[a, :] += err * v_row * w_row
                    grad_V[b, :] += err * u_row * w_row
                    grad_W[c, :] += err * u_row * v_row
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 20 == 0:
                    # Quick check
                    # Just check one random entry
                    pass
            
            # After optimization, round and verify
            U_r = round_to_half_int(U)
            V_r = round_to_half_int(V)
            W_r = round_to_half_int(W)
            
            if verify(U_r, V_r, W_r, rank):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                
    return get_schoolbook()
