import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
        n, m, p = DIMENSIONS
        start_time = time.time()
        deadline = start_time + 110  # Safety margin

        # Schoolbook decomposition as baseline
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        
        best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
        best_rank = schoolbook_rank

        def verify_decomposition(U, V, W, tol=1e-9):
            if len(U) != n * m or len(V) != m * p or len(W) != p * n:
                return False
            rank_ = len(U[0])
            if len(V[0]) != rank_ or len(W[0]) != rank_:
                return False
            if rank_ <= 0:
                return False
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1
                        actual = 0.0
                        for r in range(rank_):
                            actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if abs(actual - expected) > tol:
                            return False
            return True

        def make_result(U, V, W):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            for lst in [U_list, V_list, W_list]:
                for row in lst:
                    for idx in range(len(row)):
                        rounded = round(row[idx], 10)
                        if abs(row[idx] - rounded) < 1e-8:
                            row[idx] = rounded
            return {'U': U_list, 'V': V_list, 'W': W_list}

        # Discrete Local Search
        # Allowed values
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        # Try different ranks
        # Lower bound is max(n*m, m*p, p*n) = 21
        # Try ranks from 21 up to best_rank
        target_ranks = [21, 22, 23, 24, 25, 26, 27, 28, 30, 35, 40]
        
        for target_rank in target_ranks:
            if target_rank >= best_rank:
                continue
            if time.time() > deadline:
                break
                
            # Multiple restarts
            max_restarts = 10
            for restart in range(max_restarts):
                if time.time() > deadline:
                    break
                
                # Initialize
                U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p * n)]
                
                # Convert to numpy for faster error calculation?
                # Actually, the update logic is local, so lists are fine if we are careful.
                # But let's use numpy for the initial error calculation and maybe updates.
                
                # Precompute S_ijk = sum_r U_ij_r V_jk_r W_ki_r
                # S is a 3D array of shape (n, m, p)
                S = np.zeros((n, m, p), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0.0
                            for r in range(target_rank):
                                s_val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            S[i,j,k] = s_val
                
                error = np.sum((S - 1.0)**2)
                
                if error < 1e-9:
                    if verify_decomposition(U, V, W):
                        best_result = make_result(U, V, W)
                        best_rank = target_rank
                        break # Found a solution for this rank
                
                # Local search
                # We will do a fixed number of steps or until timeout
                steps = 0
                max_steps = 5000
                while steps < max_steps and time.time() < deadline:
                    steps += 1
                    
                    # Pick a random variable to change
                    # 1/3 chance for U, 1/3 for V, 1/3 for W
                    mat_idx = random.randint(0, 2)
                    
                    if mat_idx == 0:
                        # Change U[i, j, r]
                        i = random.randint(0, n-1)
                        j = random.randint(0, m-1)
                        r = random.randint(0, target_rank-1)
                        idx = i*m + j
                        old_val = U[idx][r]
                        new_val = random.choice(values)
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # Update S[i, j, k] for all k
                        # S[i,j,k] += delta * V[j*p+k][r] * W[k*n+i][r]
                        old_error_part = 0.0
                        new_error_part = 0.0
                        for k in range(p):
                            old_s = S[i,j,k]
                            new_s = old_s + delta * V[j*p+k][r] * W[k*n+i][r]
                            old_error_part += (old_s - 1.0)**2
                            new_error_part += (new_s - 1.0)**2
                            S[i,j,k] = new_s
                        
                        if new_error_part < old_error_part:
                            U[idx][r] = new_val
                            error += (new_error_part - old_error_part)
                            if error < 1e-9:
                                if verify_decomposition(U, V, W):
                                    best_result = make_result(U, V, W)
                                    best_rank = target_rank
                                    break # Found solution
                        else:
                            # Revert S
                            for k in range(p):
                                S[i,j,k] -= delta * V[j*p+k][r] * W[k*n+i][r]
                    
                    elif mat_idx == 1:
                        # Change V[j, k, r]
                        j = random.randint(0, m-1)
                        k = random.randint(0, p-1)
                        r = random.randint(0, target_rank-1)
                        idx = j*p + k
                        old_val = V[idx][r]
                        new_val = random.choice(values)
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # Update S[i, j, k] for all i
                        # S[i,j,k] += delta * U[i*m+j][r] * W[k*n+i][r]
                        old_error_part = 0.0
                        new_error_part = 0.0
                        for i in range(n):
                            old_s = S[i,j,k]
                            new_s = old_s + delta * U[i*m+j][r] * W[k*n+i][r]
                            old_error_part += (old_s - 1.0)**2
                            new_error_part += (new_s - 1.0)**2
                            S[i,j,k] = new_s
                        
                        if new_error_part < old_error_part:
                            V[idx][r] = new_val
                            error += (new_error_part - old_error_part)
                            if error < 1e-9:
                                if verify_decomposition(U, V, W):
                                    best_result = make_result(U, V, W)
                                    best_rank = target_rank
                                    break
                        else:
                            for i in range(n):
                                S[i,j,k] -= delta * U[i*m+j][r] * W[k*n+i][r]
                    
                    else:
                        # Change W[k, i, r]
                        k = random.randint(0, p-1)
                        i = random.randint(0, n-1)
                        r = random.randint(0, target_rank-1)
                        idx = k*n + i
                        old_val = W[idx][r]
                        new_val = random.choice(values)
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # Update S[i, j, k] for all j
                        # S[i,j,k] += delta * U[i*m+j][r] * V[j*p+k][r]
                        old_error_part = 0.0
                        new_error_part = 0.0
                        for j in range(m):
                            old_s = S[i,j,k]
                            new_s = old_s + delta * U[i*m+j][r] * V[j*p+k][r]
                            old_error_part += (old_s - 1.0)**2
                            new_error_part += (new_s - 1.0)**2
                            S[i,j,k] = new_s
                        
                        if new_error_part < old_error_part:
                            W[idx][r] = new_val
                            error += (new_error_part - old_error_part)
                            if error < 1e-9:
                                if verify_decomposition(U, V, W):
                                    best_result = make_result(U, V, W)
                                    best_rank = target_rank
                                    break
                        else:
                            for j in range(m):
                                S[i,j,k] -= delta * U[i*m+j][r] * V[j*p+k][r]
                    
                    if best_rank == target_rank:
                        break
                
                if best_rank == target_rank:
                    break
        
        return best_result
    # EVOLVE-BLOCK-END
