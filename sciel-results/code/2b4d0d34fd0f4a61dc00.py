import numpy as np
import random
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition
    U_sb = [[0] * rank_schoolbook for _ in range(nm)]
    V_sb = [[0] * rank_schoolbook for _ in range(mp)]
    W_sb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    def full_verify(U, V, W, r):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    s = sum(U[u_idx][l] * V[v_idx][l] * W[w_idx][l] for l in range(r))
                    if s != 1:
                        return False
        # Check a sample of zero entries (full check is too expensive)
        # Total entries: nm * mp * pn = 8 * 28 * 14 = 3136
        # Non-zero: n*m*p = 56
        # Zeros: 3136 - 56 = 3080
        # Check all zeros for correctness
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    # Check if this is a non-zero entry
                    is_nonzero = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if u == i*m+j and v == j*p+k and w == k*n+i:
                                    is_nonzero = True
                                    break
                            if is_nonzero:
                                break
                        if is_nonzero:
                            break
                    if is_nonzero:
                        continue
                    s = sum(U[u][l] * V[v][l] * W[w][l] for l in range(r))
                    if s != 0:
                        return False
        return True
    
    # Try to find lower rank using local search from schoolbook
    start_time = time.time()
    time_limit = 100
    
    # Use a greedy approach: try to find linear combinations
    # that allow us to merge terms
    
    best_rank = rank_schoolbook
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try ranks from 30 to 55
    for target_rank in range(30, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        # Try random restarts
        for trial in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Generate sparse random U, V, W
            density = 0.25
            vals = [-3, -2, -1, 1, 2, 3]
            
            U = [[0]*target_rank for _ in range(nm)]
            V = [[0]*target_rank for _ in range(mp)]
            W = [[0]*target_rank for _ in range(pn)]
            
            for u in range(nm):
                for l in range(target_rank):
                    if random.random() < density:
                        U[u][l] = random.choice(vals)
            
            for v in range(mp):
                for l in range(target_rank):
                    if random.random() < density:
                        V[v][l] = random.choice(vals)
            
            for w in range(pn):
                for l in range(target_rank):
                    if random.random() < density:
                        W[w][l] = random.choice(vals)
            
            if full_verify(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
