import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Target tensor is all 1s
    # We search for U, V, W with half-integer entries such that
    # sum_s U[s,i,j] * V[s,j,k] * W[s,i,k] = 1
    
    # We will try ranks from 15 up to 30
    # For each rank, we use a local search to minimize the squared error
    
    def calc_error(U, V, W, rank, n, m, p):
        # S[i,j,k] = sum_s U[s,i,j] V[s,j,k] W[s,i,k]
        # Error = sum (S - 1)^2
        # U shape: (rank, n, m)
        # V shape: (rank, m, p)
        # W shape: (rank, n, p)
        S = np.zeros((n, m, p))
        for s in range(rank):
            S += U[s] * np.einsum('jk,ik->ijk', V[s], W[s]) # Wait, V[s] is (m,p), W[s] is (n,p)
            # U[s] is (n,m)
            # Term s is U[s,i,j] * V[s,j,k] * W[s,i,k]
            # This is an outer product of U[s] and (V[s] * W[s] broadcasted?)
            # Let's do it explicitly for clarity and speed in small dims
            # Actually, np.einsum is good.
            # U[s] is (n, m) -> indices i, j
            # V[s] is (m, p) -> indices j, k
            # W[s] is (n, p) -> indices i, k
            # Product: U[s,i,j] * V[s,j,k] * W[s,i,k]
            # Sum over j for fixed i, k? No, sum over s is the outer sum.
            # S[i,k] matrix is sum_j U[i,j] V[j,k] W[i,k]? No.
            # S[i,j,k] is the term.
            # We can compute S using einsum:
            # S = np.einsum('sij,sjk,sik->ijk', U, V, W)
        return np.sum((S - 1.0)**2)

    def calc_S(U, V, W, rank):
        return np.einsum('sij,sjk,sik->ijk', U, V, W)

    best_result = None
    
    # Fallback rank 30 solution generator
    def get_rank_30_solution():
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * p * n for _ in range(1)] # Wait, W shape is (p*n, rank)
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    for rank in range(15, 31):
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts of local search
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 13 + rank)
            
            # Initialize with random half-integers in [-2, 2]
            # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
            vals = np.arange(-2, 2.5, 0.5)
            
            U = np.random.choice(vals, size=(rank, n, m))
            V = np.random.choice(vals, size=(rank, m, p))
            W = np.random.choice(vals, size=(rank, n, p))
            
            S = calc_S(U, V, W, rank)
            error = np.sum((S - 1.0)**2)
            
            # Local search
            # We will iterate until error is 0 or time limit
            # To avoid infinite loops, we limit iterations per restart
            max_iters = 5000 
            iter_count = 0
            
            while error > 1e-9 and iter_count < max_iters:
                if time.time() > deadline - 5:
                    break
                
                iter_count += 1
                
                # Randomly choose a tensor to update: U, V, or W
                tensor_id = np.random.choice(3)
                
                if tensor_id == 0:
                    # Update U
                    s, i, j = np.random.randint(rank), np.random.randint(n), np.random.randint(m)
                    old_val = U[s, i, j]
                    # Try neighbor
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    # Check if new_val is in reasonable range? 
                    # We can allow it to grow, but let's cap it to [-5, 5] to keep search local
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    # Calculate change in error
                    # S[i, j, k] changes by delta * V[s, j, k] * W[s, i, k]
                    # delta = new_val - old_val
                    delta = new_val - old_val
                    # Term to add to S[i, j, k] for all k
                    # contrib[k] = delta * V[s, j, k] * W[s, i, k]
                    
                    # Current S[i, j, :]
                    current_S_row = S[i, j, :]
                    new_S_row = current_S_row + delta * (V[s, j, :] * W[s, i, :])
                    
                    old_err_part = np.sum((current_S_row - 1.0)**2)
                    new_err_part = np.sum((new_S_row - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        # Accept
                        U[s, i, j] = new_val
                        S[i, j, :] = new_S_row
                        error += (new_err_part - old_err_part)
                    else:
                        # Reject (or accept with small prob for simulated annealing? 
                        # Let's stick to greedy for now)
                        pass
                        
                elif tensor_id == 1:
                    # Update V
                    s, j, k = np.random.randint(rank), np.random.randint(m), np.random.randint(p)
                    old_val = V[s, j, k]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    delta = new_val - old_val
                    # S[i, j, k] changes by delta * U[s, i, j] * W[s, i, k]
                    # For fixed j, k, S[:, j, k] changes
                    current_S_col = S[:, j, k]
                    # contrib[i] = delta * U[s, i, j] * W[s, i, k]
                    new_S_col = current_S_col + delta * (U[s, :, j] * W[s, :, k])
                    
                    old_err_part = np.sum((current_S_col - 1.0)**2)
                    new_err_part = np.sum((new_S_col - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        V[s, j, k] = new_val
                        S[:, j, k] = new_S_col
                        error += (new_err_part - old_err_part)
                        
                else:
                    # Update W
                    s, i, k = np.random.randint(rank), np.random.randint(n), np.random.randint(p)
                    old_val = W[s, i, k]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    delta = new_val - old_val
                    # S[i, j, k] changes by delta * U[s, i, j] * V[s, j, k]
                    # For fixed i, k, S[i, :, k] changes
                    current_S_vec = S[i, :, k]
                    # contrib[j] = delta * U[s, i, j] * V[s, j, k]
                    new_S_vec = current_S_vec + delta * (U[s, i, :] * V[s, :, k])
                    
                    old_err_part = np.sum((current_S_vec - 1.0)**2)
                    new_err_part = np.sum((new_S_vec - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        W[s, i, k] = new_val
                        S[i, :, k] = new_S_vec
                        error += (new_err_part - old_err_part)

            if error < 1e-9:
                # Found a solution!
                # Convert to lists
                U_list = [[U[s, i, j] for s in range(rank)] for i in range(n) for j in range(m)]
                V_list = [[V[s, j, k] for s in range(rank)] for j in range(m) for k in range(p)]
                W_list = [[W[s, i, k] for s in range(rank)] for k in range(p) for i in range(n)]
                
                return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback
    return get_rank_30_solution()
