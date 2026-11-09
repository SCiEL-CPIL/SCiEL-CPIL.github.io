import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 56
    
    # Build the schoolbook solution first as fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # The multiplication tensor T has entries:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # All other entries are 0.
    # We want rank-r decomposition: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = T[...]
    
    start_time = time.time()
    time_limit = 110  # Leave buffer
    
    # Strategy: Use structured search with constraint propagation
    # Key insight: The tensor is sparse - only n*m*p = 56 entries are 1, rest are 0
    # For lower rank, we need the bilinear form to cancel out on non-matching indices
    
    # Try ranks from a lower bound upward
    # Lower bound: max(nm, mp, pn) = max(8, 28, 14) = 28
    # But known results suggest higher
    
    def verify_solution(U, V, W, r):
        """Verify that the decomposition is exact"""
        # Check all non-zero entries of T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for l in range(r):
                        s += U[u_idx][l] * V[v_idx][l] * W[w_idx][l]
                    if s != 1:
                        return False
        
        # Check a sample of zero entries to ensure they're actually zero
        # Total entries: (n*m)*(m*p)*(p*n) = 8*28*14 = 3136
        # Non-zero: 56, so zeros: 3080
        # Check all zeros for correctness
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    # Skip if this is a non-zero entry
                    i = u // m
                    j = u % m
                    j2 = v // p
                    k = v % p
                    k2 = w // n
                    i2 = w % n
                    if i == i2 and j == j2 and k == k2:
                        continue  # This is a non-zero entry, already checked
                    s = 0
                    for l in range(r):
                        s += U[u][l] * V[v][l] * W[w][l]
                    if s != 0:
                        return False
        return True
    
    def verify_partial(U, V, W, r):
        """Faster verification - only check non-zero entries and a random sample of zeros"""
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for l in range(r):
                        s += U[u_idx][l] * V[v_idx][l] * W[w_idx][l]
                    if s != 1:
                        return False
        return True
    
    # Approach: Greedy construction with backtracking
    # For each rank r, try to build U, V, W column by column
    # Each column (l) contributes a rank-1 tensor U_l ⊗ V_l ⊗ W_l
    
    def try_rank(r):
        """Try to find a valid decomposition of rank r"""
        if time.time() - start_time > time_limit:
            return None
        
        # Initialize U, V, W as zero matrices
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # We'll use a randomized local search approach
        # Start with random initialization and iteratively improve
        
        best_solution = None
        best_error = float('inf')
        
        for trial in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with small integers
            for i in range(n * m):
                for l in range(r):
                    U[i][l] = random.choice([-1, 0, 1])
            for i in range(m * p):
                for l in range(r):
                    V[i][l] = random.choice([-1, 0, 1])
            for i in range(p * n):
                for l in range(r):
                    W[i][l] = random.choice([-1, 0, 1])
            
            # Local search: iteratively modify entries to reduce error
            current_error = compute_error(U, V, W, r)
            best_error_local = current_error
            best_U_local = [row[:] for row in U]
            best_V_local = [row[:] for row in V]
            best_W_local = [row[:] for row in W]
            
            for iteration in range(2000):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick a random entry to modify
                choice = random.randint(0, 2)  # 0=U, 1=V, 2=W
                if choice == 0:
                    idx = random.randint(0, n * m - 1)
                    l = random.randint(0, r - 1)
                    old_val = U[idx][l]
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    U[idx][l] = new_val
                    new_error = compute_error(U, V, W, r)
                    if new_error < best_error_local:
                        best_error_local = new_error
                        best_U_local = [row[:] for row in U]
                        best_V_local = [row[:] for row in V]
                        best_W_local = [row[:] for row in W]
                    elif new_error > current_error:
                        U[idx][l] = old_val
                    else:
                        current_error = new_error
                elif choice == 1:
                    idx = random.randint(0, m * p - 1)
                    l = random.randint(0, r - 1)
                    old_val = V[idx][l]
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    V[idx][l] = new_val
                    new_error = compute_error(U, V, W, r)
                    if new_error < best_error_local:
                        best_error_local = new_error
                        best_U_local = [row[:] for row in U]
                        best_V_local = [row[:] for row in V]
                        best_W_local = [row[:] for row in W]
                    elif new_error > current_error:
                        V[idx][l] = old_val
                    else:
                        current_error = new_error
                else:
                    idx = random.randint(0, p * n - 1)
                    l = random.randint(0, r - 1)
                    old_val = W[idx][l]
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    W[idx][l] = new_val
                    new_error = compute_error(U, V, W, r)
                    if new_error < best_error_local:
                        best_error_local = new_error
                        best_U_local = [row[:] for row in U]
                        best_V_local = [row[:] for row in V]
                        best_W_local = [row[:] for row in W]
                    elif new_error > current_error:
                        W[idx][l] = old_val
                    else:
                        current_error = new_error
            
            # Check if we found a valid solution
            if best_error_local == 0:
                if verify_solution(best_U_local, best_V_local, best_W_local, r):
                    return (best_U_local, best_V_local, best_W_local)
            
            # Keep the best solution found
            if best_error_local < best_error:
                best_error = best_error_local
                best_solution = (best_U_local, best_V_local, best_W_local)
        
        return None
    
    def compute_error(U, V, W, r):
        """Compute the error of the decomposition (sum of squared differences)"""
        error = 0
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for l in range(r):
                        s += U[u_idx][l] * V[v_idx][l] * W[w_idx][l]
                    error += (s - 1) ** 2
        return error
    
    # Try ranks from lower to upper
    for r in range(35, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If no better solution found, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
