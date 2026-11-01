"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # Schoolbook decomposition (Rank 40)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify(U, V, W, tol=1e-9):
        r = len(U[0]) if U else 0
        if r == 0:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :]), W_arr[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def to_half_integers(arr):
        """Convert array to half-integer values (multiples of 0.5)"""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                h = round(val * 2) / 2
                new_row.append(h if h != int(h) else int(h))
            result.append(new_row)
        return result

    # Try to find lower rank decompositions using structured search
    # Strategy: Use the fact that n=2 is small to create structured patterns
    
    # For each target rank, try random structured initializations
    for target_rank in range(20, rank_school):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try multiple random attempts for this rank
        attempts = 50
        for attempt in range(attempts):
            if time.time() - start_time > deadline - 5:
                break
            
            # Initialize with sparse random patterns
            # Use a structured approach: group by i values
            U = np.zeros((nm, target_rank), dtype=float)
            V = np.zeros((mp, target_rank), dtype=float)
            W = np.zeros((pn, target_rank), dtype=float)
            
            # Assign columns to different (i,j) or (j,k) or (k,i) groups
            # Each column r corresponds to a "term" in the decomposition
            # We need: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # Random assignment of which (i,j,k) triples each column "handles"
            # with possible overlap for cancellation
            for r in range(target_rank):
                # Randomly choose which indices this column affects
                # U column: affects some (i,j) pairs
                # V column: affects some (j,k) pairs  
                # W column: affects some (k,i) pairs
                
                # Simple random sparse initialization
                u_nonzero = random.randint(1, 3)
                v_nonzero = random.randint(1, 3)
                w_nonzero = random.randint(1, 3)
                
                u_idx = random.sample(range(nm), min(u_nonzero, nm))
                v_idx = random.sample(range(mp), min(v_nonzero, mp))
                w_idx = random.sample(range(pn), min(w_nonzero, pn))
                
                for idx in u_idx:
                    U[idx, r] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, r] = random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, r] = random.choice([-1, 1])
            
            # Check if this gives correct result
            if verify(U.tolist(), V.tolist(), W.tolist()):
                U_list = to_half_integers(U.tolist())
                V_list = to_half_integers(V.tolist())
                W_list = to_half_integers(W.tolist())
                
                # Re-verify with the converted values
                if verify(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break  # Found valid for this rank, try lower
        
        if time.time() - start_time > deadline - 5:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    return best_result
# EVOLVE-BLOCK-END
