import time
import random
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    # Target value for the multiplication tensor entries
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # Using the scaling trick: if we find integer solutions for target 1, they are valid.
    # However, the current program uses target 8 with integers, implying a scaling factor.
    # Let's stick to the standard definition: sum U V W = 1.
    # If we use integers, we might miss solutions that require halves.
    # The prompt says "integer/half-integer entries".
    # So we can work with integers by scaling everything by 2.
    # Let U' = 2U, V' = 2V, W' = 2W.
    # Then sum (U'/2) (V'/2) (W'/2) = 1 => sum U' V' W' = 8.
    # So we search for integer U', V', W' such that sum U' V' W' = 8.
    # This is exactly what the current program does.
    # The inspirations mention rank 14.
    # Let's try to find a rank 14 solution.
    # If we find rank 14, score is 16/14 = 1.1428...
    # If we find rank 13, score is 16/13 = 1.23...
    # If we find rank 12, score is 16/12 = 1.33...
    
    # Trivial solution rank 16
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
    
    # Scale by 2 for the integer search target 8
    U_trivial = [[x * 2 for x in row] for row in U_trivial]
    V_trivial = [[x * 2 for x in row] for row in V_trivial]
    W_trivial = [[x * 2 for x in row] for row in W_trivial]
    
    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    # We will search for lower ranks.
    # Known results: 2x2x2 is 7. 2x2x4 might be 14?
    # Let's try to find a rank 14 solution first, then 13, etc.
    
    def verify(U, V, W, R, target=8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if s != target:
                        return False
        return True

    def calc_error(U, V, W, R, target=8):
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    err += (s - target)**2
        return err

    # Strategy:
    # 1. Try to find a rank 14 solution using a more structured random initialization or local search.
    # 2. The current local search is slow. Let's try a faster local search or a different one.
    # 3. Or, use the known fact that 2x2x2 has rank 7.
    #    2x2x4 can be seen as two 2x2x2 multiplications?
    #    If we split the 4 columns into two groups of 2, we have two independent 2x2x2 multiplications.
    #    Rank would be 7 + 7 = 14.
    #    So rank 14 is definitely possible.
    #    Can we do better?
    #    Strassen for 2x2x2 is 7.
    #    For 2x2x4, maybe we can share some computations?
    #    Let's try to find rank 13 or 12.
    
    # Let's first construct the rank 14 solution explicitly to have a good baseline.
    # 2x2x2 Strassen decomposition:
    # A (2x2), B (2x2)
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We have A (2x2) and B (2x4).
    # Let B = [B1 B2] where B1, B2 are 2x2.
    # C = A * B = [A*B1, A*B2]
    # We can compute A*B1 and A*B2 using Strassen.
    # Total rank 14.
    
    # Let's construct this rank 14 solution.
    # U indices: i*m + j (i=0,1; j=0,1) -> 0,1,2,3
    # V indices: j*p + k (j=0,1; k=0,1,2,3) -> 0,1,2,3,4,5,6,7
    # W indices: k*n + i (k=0,1,2,3; i=0,1) -> 0,1,2,3,4,5,6,7
    
    # Strassen M1..M7 for first 2x2 (k=0,1)
    # Strassen M1..M7 for second 2x2 (k=2,3)
    
    # Let's define the Strassen matrices for a 2x2 multiplication.
    # U_strassen (4x7):
    # a00 is at U[0], a01 at U[1], a10 at U[2], a11 at U[3]
    # M1: a00, a11, b00, b11 -> U: [1, 0, 0, 1], V: [1, 0, 1, 0], W: [1, 1, 0, 0] (for C00)
    # This is getting complicated to map directly.
    # Let's just use the local search to find a rank 14 solution, starting from the rank 16 one and zeroing out 2 columns.
    # Or, initialize random integers and run local search.
    
    # Let's try a faster local search.
    # Instead of checking all k for each update, we can precompute some things.
    # But the current local search is not that slow, it's just that it might get stuck.
    # Let's try more restarts for rank 14, then 13, then 12.
    
    for target_rank in [14, 13, 12, 11, 10]:
        if time.time() - start_time > time_limit - 20:
            break
            
        found = False
        restarts = 0
        max_restarts = 200
        
        while not found and restarts < max_restarts:
            if time.time() - start_time > time_limit - 10:
                break
                
            # Initialize with random integers in [-2, 2]
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Local Search
            # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 8
            # Current sums S[i,j,k]
            
            sums = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                        sums[i][j][k] = s
                        
            error = sum((s - 8)**2 for i in range(n) for j in range(m) for k in range(p) for s in [sums[i][j][k]])
            
            if error == 0:
                # Convert back to half-integers (divide by 2)
                U_res = [[x / 2.0 for x in row] for row in U]
                V_res = [[x / 2.0 for x in row] for row in V]
                W_res = [[x / 2.0 for x in row] for row in W]
                
                # Verify
                if verify(U_res, V_res, W_res, target_rank, target=1.0):
                    best_rank = target_rank
                    best_U = U_res
                    best_V = V_res
                    best_W = W_res
                    found = True
                    break
                
            steps = 0
            max_steps = 2000
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
                    i = idx1 // m
                    j = idx1 % m
                    old_err_part = 0
                    new_err_part = 0
                    for k in range(p):
                        old_s = sums[i][j][k]
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
                        for k in range(p):
                            delta_s = delta * V[j*p + k][idx2] * W[k*n + i][idx2]
                            sums[i][j][k] -= delta_s
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
