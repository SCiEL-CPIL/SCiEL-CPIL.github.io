"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

import time
import random
import copy
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # The tensor T[i, j, k] = 1 if i < n, j < m, k < p, else 0
    # Actually T[i, j, k] = 1 for all valid i, j, k
    # We want T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
    # Note: The interface says W has shape (p*n, rank). The index is W[k*n + i, r].
    # So W_row = k*n + i.
    # U_row = i*m + j
    # V_row = j*p + k
    
    # Let's verify the convention again.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # U shape (n*m, rank) -> rows i*m+j
    # V shape (m*p, rank) -> rows j*p+k
    # W shape (p*n, rank) -> rows k*n+i
    
    # Let's construct the schoolbook first to have a valid baseline.
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find a better decomposition using random search / local search
    # We will try ranks from lower to upper
    # Known lower bounds: n*m*p is trivial. 
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x3x3, rank is ?
    # For 2x3x4, let's try to search.
    
    # Random initialization
    def random_decomposition(rank, scale=1.0):
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        # Normalize? No, just integers.
        # Check validity
        # T[i,j,k] should be 1 for all i,j,k
        # sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1
        
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        error += abs(s - 1)
        return error, U, V, W
    
    # Local search
    def local_search(U, V, W, rank, max_iters=1000):
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        # Precompute errors for each (i,j,k)
        def get_error(i, j, k):
            s = 0
            for c in range(rank):
                s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
            return s - 1
        
        total_error = sum(abs(get_error(i, j, k)) for i in range(n) for j in range(m) for k in range(p))
        
        if total_error == 0:
            return U, V, W, True
        
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W, False
            
            # Randomly pick a matrix, row, col
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                r = random.randint(0, n * m - 1)
                c = random.randint(0, rank - 1)
                old_val = U[r][c]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                U[r][c] = new_val
                delta = 0
                # Recalculate errors affected by U[r][c]
                # U row r corresponds to i*m+j = r -> i = r//m, j = r%m
                i = r // m
                j = r % m
                for k in range(p):
                    old_e = abs(get_error(i, j, k))
                    new_e = abs(get_error(i, j, k))
                    delta += new_e - old_e
                if delta > 0:
                    U[r][c] = old_val
                else:
                    total_error += delta
                    if total_error == 0:
                        return U, V, W, True
            elif mat_idx == 1:
                r = random.randint(0, m * p - 1)
                c = random.randint(0, rank - 1)
                old_val = V[r][c]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                V[r][c] = new_val
                delta = 0
                j = r // p
                k = r % p
                for i in range(n):
                    old_e = abs(get_error(i, j, k))
                    new_e = abs(get_error(i, j, k))
                    delta += new_e - old_e
                if delta > 0:
                    V[r][c] = old_val
                else:
                    total_error += delta
                    if total_error == 0:
                        return U, V, W, True
            else:
                r = random.randint(0, p * n - 1)
                c = random.randint(0, rank - 1)
                old_val = W[r][c]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                W[r][c] = new_val
                delta = 0
                k = r // n
                i = r % n
                for j in range(m):
                    old_e = abs(get_error(i, j, k))
                    new_e = abs(get_error(i, j, k))
                    delta += new_e - old_e
                if delta > 0:
                    W[r][c] = old_val
                else:
                    total_error += delta
                    if total_error == 0:
                        return U, V, W, True
        
        return U, V, W, False
    
    # Try ranks from low to high
    # Start from a known lower bound or just below schoolbook
    # 2x2x2 is 7. 2x3x3 might be around 10-15?
    # Let's try 10 to 23
    for rank in range(10, n * m * p):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):  # 50 random restarts per rank
            if time.time() > deadline:
                break
            
            # Sparse random init
            density = 0.2
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for i in range(n * m):
                for c in range(rank):
                    if random.random() < density:
                        U[i][c] = random.choice([-1, 1])
            for j in range(m * p):
                for c in range(rank):
                    if random.random() < density:
                        V[j][c] = random.choice([-1, 1])
            for k in range(p * n):
                for c in range(rank):
                    if random.random() < density:
                        W[k][c] = random.choice([-1, 1])
            
            err, U, V, W = random_decomposition(rank)
            if err == 0:
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                found = True
                break
            
            # Local search
            U, V, W, success = local_search(U, V, W, rank, max_iters=200)
            if success:
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                found = True
                break
    
    # Final verification of best_result
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    rank = len(U[0])
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
