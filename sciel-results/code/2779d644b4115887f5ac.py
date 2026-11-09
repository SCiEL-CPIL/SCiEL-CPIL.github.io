import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit
    
    # Target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def calculate_error(U, V, W):
        # S = sum_r U_r V_r W_r
        S = np.einsum('ur,vr,wr->uvw', U, V, W)
        # Error = || T - S ||^2
        # = ||T||^2 + ||S||^2 - 2 <T, S>
        # ||T||^2 is 27
        # <T, S> is sum(T*S)
        # ||S||^2 is sum(S*S)
        t_norm_sq = np.sum(T * T)
        s_norm_sq = np.sum(S * S)
        ts_dot = np.sum(T * S)
        return t_norm_sq + s_norm_sq - 2 * ts_dot

    def try_find_decomposition(rank, max_iters=50000):
        # Initialize with random integers -1, 0, 1
        U = np.random.randint(-1, 2, size=(n*m, rank))
        V = np.random.randint(-1, 2, size=(m*p, rank))
        W = np.random.randint(-1, 2, size=(p*n, rank))
        
        current_error = calculate_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Pick a random matrix to update
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                dim = n*m
            elif mat_idx == 1:
                mat = V
                dim = m*p
            else:
                mat = W
                dim = p*n
                
            row = random.randint(0, dim - 1)
            col = random.randint(0, rank - 1)
            
            # Try changing to -1, 0, 1
            old_val = mat[row, col]
            # Determine possible new values
            candidates = []
            if old_val != -1: candidates.append(-1)
            if old_val != 0: candidates.append(0)
            if old_val != 1: candidates.append(1)
            
            # Pick a random candidate
            new_val = random.choice(candidates)
            
            # Apply change
            mat[row, col] = new_val
            
            # Calculate new error
            new_error = calculate_error(U, V, W)
            
            # Accept if better, or accept with some probability if worse (Simulated Annealing)
            # For simplicity, just accept if better (Hill Climbing)
            # To escape local minima, we can restart or use SA.
            # Let's use simple Hill Climbing with restarts.
            if new_error < current_error:
                current_error = new_error
                if current_error < 1e-9:
                    return U, V, W
            else:
                # Revert
                mat[row, col] = old_val
                
        # Check final error
        if current_error < 1e-9:
            return U, V, W
        return None

    # Try rank 23
    # Use multiple restarts to improve chances
    best_result = None
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        result = try_find_decomposition(23, max_iters=2000)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 27
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, column] = 1
                V[v, column] = 1
                W[w, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
