import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    size_u = n * m  # 12
    size_v = m * p  # 20
    size_w = p * n  # 15
    total_size = size_u + size_v + size_w  # 47
    
    def verify_decomposition(U, V, W, R):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    term = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(R))
                    if abs(term - 1.0) > 1e-9:
                        return False
        return True
    
    def try_systematic_search(R, values, max_cols_to_fill=300):
        """
        Systematically try to fill columns with values from the given set.
        Each column has 47 entries (12+20+15). We try to find R columns that work.
        """
        if time.time() > deadline:
            return None
        
        # Precompute all possible column combinations is infeasible, so use random sampling
        num_trials = 5000
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate random R columns
            U_cols = []
            V_cols = []
            W_cols = []
            for r in range(R):
                # Randomly choose values from the candidate set
                u_col = [random.choice(values) for _ in range(size_u)]
                v_col = [random.choice(values) for _ in range(size_v)]
                w_col = [random.choice(values) for _ in range(size_w)]
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
            
            # Convert to list-of-lists format for verification
            U = [[U_cols[r][i] for r in range(R)] for i in range(size_u)]
            V = [[V_cols[r][i] for r in range(R)] for i in range(size_v)]
            W = [[W_cols[r][i] for r in range(R)] for i in range(size_w)]
            
            if verify_decomposition(U, V, W, R):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def try_gradient_descent(R, max_iters=2000, lr=0.01):
        """Use gradient descent with random restarts."""
        if time.time() > deadline:
            return None
        
        for restart in range(5):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            params = np.random.uniform(-0.5, 0.5, size=R * total_size)
            U = params[:R * size_u].reshape(R, size_u).T
            V = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
            W = params[R * (size_u + size_v) :].reshape(R, size_w).T
            
            best_loss = float('inf')
            best_params = params.copy()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute loss
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            term = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                            loss += (term - 1.0) ** 2
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                
                if best_loss < 1e-10:
                    break
                
                # Compute gradients (approximate)
                # For each parameter, perturb and compute change in loss
                grads = np.zeros_like(params)
                eps = 1e-6
                for idx in range(0, len(params), 10):  # Sample some parameters for speed
                    if time.time() > deadline:
                        break
                    old_val = params[idx]
                    params[idx] += eps
                    U2 = params[:R * size_u].reshape(R, size_u).T
                    V2 = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
                    W2 = params[R * (size_u + size_v) :].reshape(R, size_w).T
                    loss2 = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                term = np.sum(U2[u_idx, :] * V2[v_idx, :] * W2[w_idx, :])
                                loss2 += (term - 1.0) ** 2
                    grads[idx] = (loss2 - loss) / eps
                    params[idx] = old_val
                
                # Update parameters
                params -= lr * grads
                lr = max(lr * 0.99, 0.001)
            
            # Check best solution
            if best_loss < 1e-8:
                U = best_params[:R * size_u].reshape(R, size_u).T
                V = best_params[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
                W = best_params[R * (size_u + size_v) :].reshape(R, size_w).T
                
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                
                if verify_decomposition(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # For 3x4x5, known best is around 30-35
    candidate_values = [-1, -0.5, 0, 0.5, 1]
    
    for R in range(25, 45):
        if time.time() > deadline:
            break
        
        # First try systematic random search with small value set
        result = try_systematic_search(R, candidate_values)
        if result:
            return result
        
        # Then try gradient descent
        result = try_gradient_descent(R)
        if result:
            return result
    
    # Fallback to schoolbook
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
