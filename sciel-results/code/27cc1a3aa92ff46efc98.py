import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    def calc_S(U, V, W, rank):
        """Compute S[i,j,k] = sum_s U[s,i,j] * V[s,j,k] * W[s,i,k]"""
        return np.einsum('sij,sjk,sik->ijk', U, V, W)
    
    def verify_solution(U_list, V_list, W_list, rank, n, m, p):
        """Verify that the solution exactly decomposes the tensor"""
        U = np.array(U_list, dtype=float).reshape(n, m, rank)
        V = np.array(V_list, dtype=float).reshape(m, p, rank)
        W = np.array(W_list, dtype=float).reshape(p, n, rank)
        # Transpose to get (rank, n, m), (rank, m, p), (rank, n, p)
        U_t = U.transpose(2, 0, 1)
        V_t = V.transpose(2, 0, 1)
        W_t = W.transpose(2, 0, 1)
        S = calc_S(U_t, V_t, W_t, rank)
        return np.allclose(S, 1.0, atol=1e-10)
    
    def get_rank_30_solution():
        """Trivial rank-30 solution"""
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
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
    
    def local_search(U, V, W, target_rank, max_time=5):
        """
        Perform local search to minimize error.
        U, V, W are (rank, n, m), (rank, m, p), (rank, p, n) arrays.
        Returns (U, V, W, error).
        """
        start_ls = time.time()
        deadline_ls = start_ls + max_time
        
        S = calc_S(U, V, W, target_rank)
        error = np.sum((S - 1.0)**2)
        
        if error < 1e-9:
            return U, V, W, error
            
        best_error = error
        stuck_count = 0
        max_stuck = 100
        
        while error > 1e-9 and time.time() < deadline_ls:
            # Choose a random tensor and coordinate
            tensor_id = np.random.randint(3)
            
            if tensor_id == 0:
                s, i, j = np.random.randint(target_rank), np.random.randint(n), np.random.randint(m)
                old_val = U[s, i, j]
                # Try a few random steps
                improved = False
                for _ in range(3):
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    if new_val < -10 or new_val > 10:
                        continue
                    
                    delta = new_val - old_val
                    change_vec = delta * (V[s, j, :] * W[s, i, :])
                    new_S_row = S[i, j, :] + change_vec
                    
                    old_err_part = np.sum((S[i, j, :] - 1.0)**2)
                    new_err_part = np.sum((new_S_row - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        U[s, i, j] = new_val
                        S[i, j, :] = new_S_row
                        error += (new_err_part - old_err_part)
                        stuck_count = 0
                        if error < best_error:
                            best_error = error
                        improved = True
                        break
                if not improved:
                    stuck_count += 1
                    
            elif tensor_id == 1:
                s, j, k = np.random.randint(target_rank), np.random.randint(m), np.random.randint(p)
                old_val = V[s, j, k]
                improved = False
                for _ in range(3):
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    if new_val < -10 or new_val > 10:
                        continue
                    
                    delta = new_val - old_val
                    change_vec = delta * (U[s, :, j] * W[s, :, k])
                    new_S_col = S[:, j, k] + change_vec
                    
                    old_err_part = np.sum((S[:, j, k] - 1.0)**2)
                    new_err_part = np.sum((new_S_col - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        V[s, j, k] = new_val
                        S[:, j, k] = new_S_col
                        error += (new_err_part - old_err_part)
                        stuck_count = 0
                        if error < best_error:
                            best_error = error
                        improved = True
                        break
                if not improved:
                    stuck_count += 1
                        
            else:
                s, i, k = np.random.randint(target_rank), np.random.randint(n), np.random.randint(p)
                old_val = W[s, i, k]
                improved = False
                for _ in range(3):
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    if new_val < -10 or new_val > 10:
                        continue
                    
                    delta = new_val - old_val
                    change_vec = delta * (U[s, i, :] * V[s, :, k])
                    new_S_vec = S[i, :, k] + change_vec
                    
                    old_err_part = np.sum((S[i, :, k] - 1.0)**2)
                    new_err_part = np.sum((new_S_vec - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        W[s, i, k] = new_val
                        S[i, :, k] = new_S_vec
                        error += (new_err_part - old_err_part)
                        stuck_count = 0
                        if error < best_error:
                            best_error = error
                        improved = True
                        break
                if not improved:
                    stuck_count += 1
            
            if stuck_count > max_stuck:
                # Kick: randomize a few variables to escape local minimum
                for _ in range(5):
                    tensor_id = np.random.randint(3)
                    if tensor_id == 0:
                        s, i, j = np.random.randint(target_rank), np.random.randint(n), np.random.randint(m)
                        U[s, i, j] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    elif tensor_id == 1:
                        s, j, k = np.random.randint(target_rank), np.random.randint(m), np.random.randint(p)
                        V[s, j, k] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    else:
                        s, i, k = np.random.randint(target_rank), np.random.randint(n), np.random.randint(p)
                        W[s, i, k] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                
                S = calc_S(U, V, W, target_rank)
                error = np.sum((S - 1.0)**2)
                stuck_count = 0

        return U, V, W, error

    def try_find_rank(target_rank, num_restarts=5, time_per_restart=5):
        """
        Try to find a solution of target_rank using pruning from rank 30.
        """
        # Get rank 30 solution
        sol30 = get_rank_30_solution()
        U30 = np.array(sol30['U'], dtype=float).reshape(n, m, 30).transpose(2, 0, 1)
        V30 = np.array(sol30['V'], dtype=float).reshape(m, p, 30).transpose(2, 0, 1)
        W30 = np.array(sol30['W'], dtype=float).reshape(p, n, 30).transpose(2, 0, 1)
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(r * 100 + target_rank)
            
            # Initialize with random subset of rank 30 terms
            U = U30.copy()
            V = V30.copy()
            W = W30.copy()
            
            # Select indices to keep
            keep_idx = np.random.choice(30, size=target_rank, replace=False)
            zero_idx = np.setdiff1d(np.arange(30), keep_idx)
            
            # Zero out the removed terms
            for idx in zero_idx:
                U[idx, :, :] = 0
                V[idx, :, :] = 0
                W[idx, :, :] = 0
            
            # Add some random noise to the kept terms to allow movement
            for idx in keep_idx:
                U[idx, :, :] += np.random.normal(0, 0.1, size=(n, m))
                V[idx, :, :] += np.random.normal(0, 0.1, size=(m, p))
                W[idx, :, :] += np.random.normal(0, 0.1, size=(p, n))
            
            # Run local search
            U, V, W, error = local_search(U, V, W, target_rank, max_time=time_per_restart)
            
            if error < 1e-9:
                # Convert to list format
                U_list = [[U[s, i, j] for s in range(target_rank)] for i in range(n) for j in range(m)]
                V_list = [[V[s, j, k] for s in range(target_rank)] for j in range(m) for k in range(p)]
                W_list = [[W[s, i, k] for s in range(target_rank)] for k in range(p) for i in range(n)]
                
                # Verify
                if verify_solution(U_list, V_list, W_list, target_rank, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 28 down to 25
    # Allocate time: 28 gets 10 restarts, 27 gets 5, 26 gets 5, 25 gets 5
    # Total 25 restarts * 5s = 125s. Might be tight.
    # Let's do 28: 10 restarts, 27: 5 restarts, 26: 5 restarts.
    
    if time.time() < deadline - 60:
        result = try_find_rank(28, num_restarts=10, time_per_restart=5)
        if result:
            return result

    if time.time() < deadline - 30:
        result = try_find_rank(27, num_restarts=5, time_per_restart=5)
        if result:
            return result

    if time.time() < deadline - 15:
        result = try_find_rank(26, num_restarts=5, time_per_restart=5)
        if result:
            return result

    # Fallback
    return get_rank_30_solution()
