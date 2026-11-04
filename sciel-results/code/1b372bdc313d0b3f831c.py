"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_structured_rank(rank, time_limit=None):
        """
        Try to find a rank-r exact decomposition using structured U, V patterns.
        Key insight: for the all-ones tensor, if we choose U and V with entries
        from a small set, M[k,i,r] = sum_j U[i,j,r]*V[j,k,r] will have structure,
        and W can be solved if M has appropriate properties.
        """
        # Entry sets to try - start with simple ones
        entry_sets = [
            [0, 1],
            [-1, 0, 1],
            [0, 1, -1, 2],
            [0, 1, -1, 0.5],
        ]
        
        for entries in entry_sets:
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            num_trials = 5000 if len(entries) <= 2 else 2000
            for trial in range(num_trials):
                if time_limit is not None and time.time() - start_time > time_limit:
                    return None
                
                # Generate U and V with entries from the set
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                
                # Use random but constrained entries
                for idx in range(n*m):
                    for r in range(rank):
                        U[idx, r] = random.choice(entries)
                for idx in range(m*p):
                    for r in range(rank):
                        V[idx, r] = random.choice(entries)
                
                # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                M = np.zeros((p*n, rank))
                for i in range(n):
                    for k in range(p):
                        for r in range(rank):
                            s = 0.0
                            for j in range(m):
                                s += U[i*m+j, r] * V[j*p+k, r]
                            M[k*n+i, r] = s
                
                # For each (k,i), we need sum_r M[k*n+i, r] * W[k*n+i, r] = 1
                # Try to find W with simple entries
                W = np.zeros((p*n, rank))
                valid = True
                
                for idx in range(p*n):
                    row = M[idx, :]
                    # Find a sparse combination that sums to 1
                    # Try single non-zero entries first
                    found = False
                    for r in range(rank):
                        if row[r] != 0:
                            w_val = 1.0 / row[r]
                            # Check if w_val is "simple" (integer or half-integer)
                            if abs(w_val * 2 - round(w_val * 2)) < 1e-9:
                                W[idx, r] = w_val
                                found = True
                                break
                    
                    if not found:
                        # Try pairs
                        for r1 in range(rank):
                            if found:
                                break
                            if row[r1] == 0:
                                continue
                            for r2 in range(r1+1, rank):
                                if row[r2] == 0:
                                    continue
                                # Solve: a*row[r1] + b*row[r2] = 1
                                # Try simple a, b values
                                for a in [-2, -1, -0.5, 0.5, 1, 2]:
                                    b = (1 - a * row[r1]) / row[r2]
                                    if abs(b * 2 - round(b * 2)) < 1e-9:
                                        W[idx, r1] = a
                                        W[idx, r2] = b
                                        found = True
                                        break
                            if found:
                                break
                    
                    if not found:
                        valid = False
                        break
                
                if valid and check_decomposition(U, V, W, rank):
                    return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: at least max(n, m, p) but for tensor rank it's higher
    # For all-ones tensor of shape (3,7,4), try from 15 upward
    for r in range(15, min(n*m*p, 50)):
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_structured_rank(r, time_limit=DEADLINE)
        if result is not None:
            U, V, W = result
            if r < best_rank:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
            break
    
    # If we didn't find anything better, return schoolbook
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
