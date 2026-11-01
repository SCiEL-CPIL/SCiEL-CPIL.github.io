import random
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    # We search for lower ranks
    # Theoretical lower bound is max(n*m, m*p, p*n) = 8? No, rank >= max dimension product?
    # Actually rank >= max(nm, mp, pn) is not true. Rank >= max(n, m, p) is true?
    # For 2x2x2, rank is 7.
    # Let's try ranks from 15 down to 1.
    
    for target_rank in range(rank_trivial - 1, 0, -1):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try to find a solution for this rank
        found = False
        restarts = 0
        max_restarts = 50
        
        while not found and restarts < max_restarts:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Initialize with random small integers (scaled by 2, so target is 8)
            # Using integers -2, -1, 0, 1, 2
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Local Search
            # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 8
            # Current sums S[i,j,k]
            
            def calc_sums(U, V, W, R):
                sums = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(R):
                                s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                            sums[i][j][k] = s
                return sums

            sums = calc_sums(U, V, W, target_rank)
            error = sum((s - 8)**2 for i in range(n) for j in range(m) for k in range(p) for s in [sums[i][j][k]])
            
            if error == 0:
                # Convert back to half-integers (divide by 2)
                U_res = [[x / 2.0 for x in row] for row in U]
                V_res = [[x / 2.0 for x in row] for row in V]
                W_res = [[x / 2.0 for x in row] for row in W]
                
                # Verify
                # Check if entries are integer or half-integer
                # They are by construction (x/2 where x is int)
                
                # Update best
                best_rank = target_rank
                best_U = U_res
                best_V = V_res
                best_W = W_res
                found = True
                break
                
            steps = 0
            max_steps = 1000
            while error > 0 and steps < max_steps:
                if time.time() - start_time > time_limit - 5:
                    break
                    
                # Pick random variable
                var_type = random.choice(['U', 'V', 'W'])
                if var_type == 'U':
                    idx1 = random.randint(0, n*m - 1)
                    idx2 = random.randint(0, target_rank - 1)
                    delta = random.choice([-1, 1])
                    old_val = U[idx1][idx2]
                    U[idx1][idx2] += delta
                    
                    # Calculate change in error
                    # Only affected sums are for i,j such that idx1 = i*m+j
                    i = idx1 // m
                    j = idx1 % m
                    old_err_part = 0
                    new_err_part = 0
                    for k in range(p):
                        old_s = sums[i][j][k]
                        # Update sum
                        # sum_new = sum_old + delta * V[j*p+k][idx2] * W[k*n+i][idx2]
                        delta_s = delta * V[j*p + k][idx2] * W[k*n + i][idx2]
                        new_s = old_s + delta_s
                        sums[i][j][k] = new_s
                        old_err_part += (old_s - 8)**2
                        new_err_part += (new_s - 8)**2
                    
                    new_error = error - old_err_part + new_err_part
                    if new_error < error:
                        error = new_error
                    else:
                        # Revert
                        U[idx1][idx2] = old_val
                        # Revert sums
                        for k in range(p):
                            i_ = idx1 // m
                            j_ = idx1 % m
                            delta_s = delta * V[j_*p + k][idx2] * W[k*n + i_][idx2]
                            sums[i_][j_][k] -= delta_s
                    steps += 1
                    
                elif var_type == 'V':
                    idx1 = random.randint(0, m*p - 1)
                    idx2 = random.randint(0, target_rank - 1)
                    delta = random.choice([-1, 1])
                    old_val = V[idx1][idx2]
                    V[idx1][idx2] += delta
                    
                    j = idx1 // p
                    k = idx1 % p
                    old_err_part = 0
                    new_err_part = 0
                    for i in range(n):
                        old_s = sums[i][j][k]
                        delta_s = delta * U[i*m + j][idx2] * W[k*n + i][idx2]
                        new_s = old_s + delta_s
                        sums[i][j][k] = new_s
                        old_err_part += (old_s - 8)**2
                        new_err_part += (new_s - 8)**2
                        
                    new_error = error - old_err_part + new_err_part
                    if new_error < error:
                        error = new_error
                    else:
                        V[idx1][idx2] = old_val
                        for i in range(n):
                            delta_s = delta * U[i*m + j][idx2] * W[k*n + i][idx2]
                            sums[i][j][k] -= delta_s
                    steps += 1
                    
                else: # W
                    idx1 = random.randint(0, p*n - 1)
                    idx2 = random.randint(0, target_rank - 1)
                    delta = random.choice([-1, 1])
                    old_val = W[idx1][idx2]
                    W[idx1][idx2] += delta
                    
                    k = idx1 // n
                    i = idx1 % n
                    old_err_part = 0
                    new_err_part = 0
                    for j in range(m):
                        old_s = sums[i][j][k]
                        delta_s = delta * U[i*m + j][idx2] * V[j*p + k][idx2]
                        new_s = old_s + delta_s
                        sums[i][j][k] = new_s
                        old_err_part += (old_s - 8)**2
                        new_err_part += (new_s - 8)**2
                        
                    new_error = error - old_err_part + new_err_part
                    if new_error < error:
                        error = new_error
                    else:
                        W[idx1][idx2] = old_val
                        for j in range(m):
                            delta_s = delta * U[i*m + j][idx2] * V[j*p + k][idx2]
                            sums[i][j][k] -= delta_s
                    steps += 1

            restarts += 1

    return {"U": best_U, "V": best_V, "W": best_W}
