"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline with safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def check_decomposition(U, V, W, rank, n, m, p):
        """Check if U, V, W of given rank satisfy the decomposition."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True

    def local_search_decomposition(n, m, p, target_rank, max_iterations=20000):
        """
        Try to find a rank-target_rank decomposition for M_{n,m,p}
        using local search on sparse integer/half-integer entries.
        """
        # Initialize with random sparse integers
        # Use -1, 0, 1
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Random initialization
        for i in range(n * m):
            for r in range(target_rank):
                if random.random() < 0.3:
                    U[i][r] = random.choice([-1, 1])
        for i in range(m * p):
            for r in range(target_rank):
                if random.random() < 0.3:
                    V[i][r] = random.choice([-1, 1])
        for i in range(p * n):
            for r in range(target_rank):
                if random.random() < 0.3:
                    W[i][r] = random.choice([-1, 1])
        
        # Compute initial error
        def compute_error(U, V, W):
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                        err += (total - 1.0)**2
            return err
        
        current_error = compute_error(U, V, W)
        
        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                idx = random.randint(0, n*m - 1)
                r = random.randint(0, target_rank - 1)
                old_val = U[idx][r]
                # Try changing by +/- 1 or +/- 0.5
                delta = random.choice([-1, 1, -0.5, 0.5])
                new_val = old_val + delta
                U[idx][r] = new_val
            elif mat_choice == 'V':
                idx = random.randint(0, m*p - 1)
                r = random.randint(0, target_rank - 1)
                old_val = V[idx][r]
                delta = random.choice([-1, 1, -0.5, 0.5])
                new_val = old_val + delta
                V[idx][r] = new_val
            else:
                idx = random.randint(0, p*n - 1)
                r = random.randint(0, target_rank - 1)
                old_val = W[idx][r]
                delta = random.choice([-1, 1, -0.5, 0.5])
                new_val = old_val + delta
                W[idx][r] = new_val
            
            new_error = compute_error(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
                if current_error < 1e-6:
                    return U, V, W
            else:
                # Revert
                if mat_choice == 'U':
                    U[idx][r] = old_val
                elif mat_choice == 'V':
                    V[idx][r] = old_val
                else:
                    W[idx][r] = old_val
            
            # Occasionally restart if stuck
            if it % 5000 == 0 and it > 0:
                # Re-initialize
                for i in range(n * m):
                    for r in range(target_rank):
                        U[i][r] = 0
                        if random.random() < 0.3:
                            U[i][r] = random.choice([-1, 1])
                for i in range(m * p):
                    for r in range(target_rank):
                        V[i][r] = 0
                        if random.random() < 0.3:
                            V[i][r] = random.choice([-1, 1])
                for i in range(p * n):
                    for r in range(target_rank):
                        W[i][r] = 0
                        if random.random() < 0.3:
                            W[i][r] = random.choice([-1, 1])
                current_error = compute_error(U, V, W)
                
        return None

    # Strategy: Find a decomposition for M_{3,3,3} with rank < 27
    # Then use schoolbook for the remaining 5 columns (Rank 45)
    # Total Rank = R_{3,3,3} + 45
    
    found_333 = None
    for rank_333 in range(24, 27):
        if time.time() > deadline - 20:
            break
        res = local_search_decomposition(3, 3, 3, rank_333, max_iterations=10000)
        if res:
            found_333 = (rank_333, res)
            break
            
    if found_333:
        rank_333, (U_333, V_333, W_333) = found_333
        # Construct full decomposition
        # We need to map the 3x3x3 solution to the 3x3x8 problem
        # The first 3 columns of B (k=0,1,2) use this decomposition.
        # The remaining 5 columns (k=3..7) use schoolbook.
        
        total_rank = rank_333 + 15 # 3*3*5 = 45? No, 3*3*5 is 45.
        # Wait, schoolbook for 3x3x5 is 3*3*5 = 45.
        total_rank = rank_333 + 45
        
        U_full = [[0] * total_rank for _ in range(3 * 3)]
        V_full = [[0] * total_rank for _ in range(3 * 8)]
        W_full = [[0] * total_rank for _ in range(8 * 3)]
        
        # Copy 3x3x3 part
        for i in range(9):
            for r in range(rank_333):
                U_full[i][r] = U_333[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(rank_333):
                    V_full[j*8 + k][r] = V_333[j*3 + k][r]
        for k in range(3):
            for i in range(3):
                for r in range(rank_333):
                    W_full[k*3 + i][r] = W_333[k*3 + i][r]
                    
        # Add schoolbook for k=3..7
        col_offset = rank_333
        for i in range(3):
            for j in range(3):
                for k in range(3, 8):
                    U_full[i*3 + j][col_offset] = 1
                    V_full[j*8 + k][col_offset] = 1
                    W_full[k*3 + i][col_offset] = 1
                    col_offset += 1
                    
        if check_decomposition(U_full, V_full, W_full, total_rank, 3, 3, 8):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = {"U": U_full, "V": V_full, "W": W_full}
                return best_result

    return best_result
# EVOLVE-BLOCK-END
