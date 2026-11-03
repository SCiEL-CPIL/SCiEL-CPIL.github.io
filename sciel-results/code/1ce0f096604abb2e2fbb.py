import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    best_rank = n * m * p  # 36
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    # Target ranks to try (lower is better)
    target_ranks = [28, 29, 30, 31, 32, 33, 34, 35]
    
    def verify(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication."""
        rank = U_mat.shape[1]
        
        # Check non-zero entries: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check zero entries: for all other (a,b,c), sum should be 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    if (i == i2 and j == j2 and k == k2):
                        continue  # Already checked
                    
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > 1e-9:
                        return False
        
        return True
    
    def compute_error(U_mat, V_mat, W_mat):
        """Compute squared error of the decomposition."""
        rank = U_mat.shape[1]
        err = 0.0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    err += (s - target) ** 2
        return err
    
    def local_search(U_init, V_init, W_init, target_rank, max_iters=5000):
        """Perform local search to minimize error."""
        U_cur = U_init.copy()
        V_cur = V_init.copy()
        W_cur = W_init.copy()
        
        current_error = compute_error(U_cur, V_cur, W_cur)
        
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Pick a random element to perturb
            mat_idx = random.choice([0, 1, 2])
            rows = [n*m, m*p, p*n]
            row = random.randint(0, rows[mat_idx] - 1)
            col = random.randint(0, target_rank - 1)
            
            # Store old value
            if mat_idx == 0:
                old_val = U_cur[row, col]
                U_cur[row, col] += random.choice([-0.5, 0.5])
            elif mat_idx == 1:
                old_val = V_cur[row, col]
                V_cur[row, col] += random.choice([-0.5, 0.5])
            else:
                old_val = W_cur[row, col]
                W_cur[row, col] += random.choice([-0.5, 0.5])
            
            new_error = compute_error(U_cur, V_cur, W_cur)
            
            if new_error < current_error:
                current_error = new_error
                if current_error < 1e-9:
                    if verify(U_cur, V_cur, W_cur):
                        return U_cur, V_cur, W_cur, current_error
            else:
                # Revert
                if mat_idx == 0:
                    U_cur[row, col] = old_val
                elif mat_idx == 1:
                    V_cur[row, col] = old_val
                else:
                    W_cur[row, col] = old_val
        
        return U_cur, V_cur, W_cur, current_error
    
    # Try to find a lower rank decomposition
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        found = False
        num_restarts = 0
        while time.time() < deadline - 10 and not found:
            num_restarts += 1
            
            # Initialize with random half-integers
            U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
            V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
            W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
            
            U_res, V_res, W_res, err = local_search(U_rand, V_rand, W_rand, target_rank)
            
            if err < 1e-9:
                if verify(U_res, V_res, W_res):
                    best_U = U_res.copy()
                    best_V = V_res.copy()
                    best_W = W_res.copy()
                    best_rank = target_rank
                    found = True
                    break
    
    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
