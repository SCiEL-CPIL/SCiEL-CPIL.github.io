"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

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
        """Check if U, V, W exactly decompose the matrix multiplication tensor using numpy."""
        # T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # Build the full tensor check efficiently
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_structured_random(rank, num_trials=50, time_limit=None):
        """
        Try to find a rank-r exact decomposition using random U, V with integer entries,
        then solve for W.
        """
        for trial in range(num_trials):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            # Generate random U, V with entries in {-1, 0, 1}
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            for _ in range(n*m):
                U[_] = np.random.choice([-1, 0, 1], size=rank)
            for _ in range(m*p):
                V[_] = np.random.choice([-1, 0, 1], size=rank)
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        M[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
            
            # For each (k,i), we need sum_r M[k*n+i, r] * W[k*n+i, r] = 1
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                found = False
                # Try single non-zero W entry
                for r in range(rank):
                    if M[idx, r] != 0:
                        w_val = 1.0 / M[idx, r]
                        if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                            W[idx, r] = w_val
                            found = True
                            break
                
                if not found:
                    # Try two non-zero W entries
                    for r1 in range(rank):
                        if M[idx, r1] == 0:
                            continue
                        for r2 in range(r1+1, rank):
                            if M[idx, r2] == 0:
                                continue
                            for a in range(-3, 4):
                                for b in range(-3, 4):
                                    if a == 0 and b == 0:
                                        continue
                                    if abs(a*M[idx, r1] + b*M[idx, r2] - 1.0) < 1e-9:
                                        W[idx, r1] = a
                                        W[idx, r2] = b
                                        found = True
                                        break
                                if found:
                                    break
                        if found:
                            break
                
                if not found:
                    valid = False
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_binary_structure(rank, time_limit=None):
        """
        Try a structured approach: U and V are binary (0 or 1), 
        and we look for W with simple entries.
        """
        for trial in range(20):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            # Each row of U has exactly one 1 at a random position
            for i in range(n*m):
                pos = random.randint(0, rank-1)
                U[i, pos] = 1.0
            for j in range(m*p):
                pos = random.randint(0, rank-1)
                V[j, pos] = 1.0
            
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        M[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
            
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                found = False
                for r in range(rank):
                    if M[idx, r] != 0:
                        w_val = 1.0 / M[idx, r]
                        if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                            W[idx, r] = w_val
                            found = True
                            break
                if not found:
                    valid = False
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    for rank in range(28, min(n*m*p, 84)):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        result = try_structured_random(rank, num_trials=30, time_limit=DEADLINE)
        if result is None:
            result = try_binary_structure(rank, time_limit=DEADLINE)
        
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
