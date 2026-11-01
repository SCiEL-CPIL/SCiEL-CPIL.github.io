"""Deterministic block-decomposition search for 2x5x6 matrix multiplication tensor."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Helper to validate a decomposition
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Check all tensor components
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Consistent term should be 1
                    u_val = U_mat[i*m+j, :]
                    v_val = V_mat[j*p+k, :]
                    w_val = W_mat[k*n+i, :]
                    total = np.sum(u_val * v_val * w_val)
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
                    
                    # Inconsistent j terms should be 0
                    for j_prime in range(m):
                        if j_prime == j:
                            continue
                        u_val_p = U_mat[i*m+j_prime, :]
                        total_p = np.sum(u_val_p * v_val * w_val)
                        if not np.isclose(total_p, 0.0, atol=1e-9):
                            return False
                            
                    # Inconsistent k terms should be 0
                    for k_prime in range(p):
                        if k_prime == k:
                            continue
                        v_val_p = V_mat[j*p+k_prime, :]
                        w_val_p = W_mat[k_prime*n+i, :]
                        total_k = np.sum(u_val * v_val_p * w_val_p)
                        if not np.isclose(total_k, 0.0, atol=1e-9):
                            return False

                    # Inconsistent i terms should be 0
                    for i_prime in range(n):
                        if i_prime == i:
                            continue
                        u_val_i = U_mat[i_prime*m+j, :]
                        w_val_i = W_mat[k*n+i_prime, :]
                        total_i = np.sum(u_val_i * v_val * w_val_i)
                        if not np.isclose(total_i, 0.0, atol=1e-9):
                            return False

        return True

    # Schoolbook baseline: rank = n*m*p = 60
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    # Try to find better decompositions using structured approaches
    # Strategy: Try ranks from lower bounds upward
    
    # Lower bound: rank >= max(n*m, m*p, p*n) is not quite right
    # Actually for matrix multiplication, rank >= max(n, m, p) is trivial
    # But we can try from reasonable values
    
    # Try structured random search with various rank values
    for r in range(15, rank_sb):
        if time.time() - start_time > time_limit:
            break
        
        # Multiple attempts per rank
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate sparse random integer matrices
            U_rand = np.zeros((n * m, r), dtype=float)
            V_rand = np.zeros((m * p, r), dtype=float)
            
            # Use sparse structure with values from {-1, 0, 1}
            density = 0.3 + 0.4 * (r / rank_sb)  # Higher density for larger ranks
            
            for idx in range(n * m):
                for c in range(r):
                    if random.random() < density:
                        U_rand[idx, c] = random.choice([-1, 1])
            
            for idx in range(m * p):
                for c in range(r):
                    if random.random() < density:
                        V_rand[idx, c] = random.choice([-1, 1])
            
            # Now solve for W
            # For each (i, j, k), we need:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  (consistent)
            # sum_c U[i*m+j', c] * V[j*p+k, c] * W[k*n+i, c] = 0  (inconsistent j)
            # etc.
            
            # This is a linear system in W variables
            # Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r)
            # Total: p * n * r variables
            
            num_vars = p * n * r
            
            # Build the system: for each (i, j, k) triple, we have constraints
            # Let's collect all constraints
            
            A_rows = []
            b_vals = []
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Consistent: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
                        row = np.zeros(num_vars)
                        for c in range(r):
                            coeff = U_rand[i*m+j, c] * V_rand[j*p+k, c]
                            var_idx = (k*n + i) * r + c
                            row[var_idx] = coeff
                        A_rows.append(row)
                        b_vals.append(1.0)
                        
                        # Inconsistent j: for j' != j
                        for j_prime in range(m):
                            if j_prime == j:
                                continue
                            row = np.zeros(num_vars)
                            for c in range(r):
                                coeff = U_rand[i*m+j_prime, c] * V_rand[j*p+k, c]
                                var_idx = (k*n + i) * r + c
                                row[var_idx] = coeff
                            A_rows.append(row)
                            b_vals.append(0.0)
                        
                        # Inconsistent k: for k' != k
                        for k_prime in range(p):
                            if k_prime == k:
                                continue
                            row = np.zeros(num_vars)
                            for c in range(r):
                                coeff = U_rand[i*m+j, c] * V_rand[j*p+k_prime, c]
                                var_idx = (k_prime*n + i) * r + c
                                row[var_idx] = coeff
                            A_rows.append(row)
                            b_vals.append(0.0)
                        
                        # Inconsistent i: for i' != i
                        for i_prime in range(n):
                            if i_prime == i:
                                continue
                            row = np.zeros(num_vars)
                            for c in range(r):
                                coeff = U_rand[i_prime*m+j, c] * V_rand[j*p+k, c]
                                var_idx = (k*n + i_prime) * r + c
                                row[var_idx] = coeff
                            A_rows.append(row)
                            b_vals.append(0.0)
            
            if not A_rows:
                continue
                
            A = np.array(A_rows)
            b = np.array(b_vals)
            
            # Solve least squares
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check if solution is close to half-integers
            W_check = W_vec * 2
            W_rounded = np.round(W_check)
            
            diff = np.abs(W_check - W_rounded)
            if np.max(diff) < 1e-6:
                W_snap = W_rounded / 2.0
                W_mat = W_snap.reshape(p * n, r)
                
                # Verify
                if is_valid_decomposition(U_rand.tolist(), V_rand.tolist(), W_mat.tolist(), r):
                    best_U = U_rand.tolist()
                    best_V = V_rand.tolist()
                    best_W = W_mat.tolist()
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
