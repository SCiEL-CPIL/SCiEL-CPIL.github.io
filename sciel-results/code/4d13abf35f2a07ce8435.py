import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # 1. Schoolbook decomposition (Rank 12)
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s, V_s, W_s
    
    # 2. Try to find lower rank using improved search
    # Target ranks: 10, 11
    for target_rank in [10, 11]:
        if time.time() > deadline:
            break
            
        # Number of attempts based on remaining time
        # Estimate time per attempt: ~0.01s? Let's say 100 attempts/s.
        time_left = deadline - time.time()
        max_attempts = int(time_left * 20) # Be conservative
        if max_attempts < 10:
            max_attempts = 10
            
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Initialize randomly, but encourage sparsity
            # Use a mix of random uniform and sparse initialization
            if attempt % 2 == 0:
                U = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(p * n)]
            else:
                U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Alternating Least Squares
            for it in range(30): # Reduced iterations, more attempts
                if time.time() > deadline:
                    break
                U = solve_for_U(V, W, n, m, p, target_rank)
                V = solve_for_V(U, W, n, m, p, target_rank)
                W = solve_for_W(U, V, n, m, p, target_rank)
            
            # Round to half-integers
            def round_to_half(x):
                return round(x * 2) / 2
            
            U_r = [[round_to_half(x) for x in row] for row in U]
            V_r = [[round_to_half(x) for x in row] for row in V]
            W_r = [[round_to_half(x) for x in row] for row in W]
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U_r[i * m + j][c] * V_r[j * p + k][c] * W_r[k * n + i][c]
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}

def solve_for_U(V, W, n, m, p, r):
    num_eqs = n * m * p
    num_unk = n * m * r
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    unk_idx = (i * m + j) * r + c
                    A[eq_idx, unk_idx] = V[j * p + k][c] * W[k * n + i][c]
                eq_idx += 1
    U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    U = [[0] * r for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for c in range(r):
                U[i * m + j][c] = U_flat[(i * m + j) * r + c]
    return U

def solve_for_V(U, W, n, m, p, r):
    num_eqs = n * m * p
    num_unk = m * p * r
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    unk_idx = (j * p + k) * r + c
                    A[eq_idx, unk_idx] = U[i * m + j][c] * W[k * n + i][c]
                eq_idx += 1
    V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    V = [[0] * r for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            for c in range(r):
                V[j * p + k][c] = V_flat[(j * p + k) * r + c]
    return V

def solve_for_W(U, V, n, m, p, r):
    num_eqs = n * m * p
    num_unk = p * n * r
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    unk_idx = (k * n + i) * r + c
                    A[eq_idx, unk_idx] = U[i * m + j][c] * V[j * p + k][c]
                eq_idx += 1
    W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    W = [[0] * r for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for c in range(r):
                W[k * n + i][c] = W_flat[(k * n + i) * r + c]
    return W
# EVOLVE-BLOCK-END
