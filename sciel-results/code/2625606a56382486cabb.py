import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    start_time = time.time()
    deadline = start_time + 100 # 20s margin

    # Strategy:
    # 1. The tensor is very sparse (only 54 non-zero entries out of 2916).
    # 2. We can try to find a decomposition with rank < 54.
    # 3. A simple randomized search over small integer/half-integer values is unlikely to find a solution
    #    for rank < 54 in the available time, but we can try a more structured approach.
    # 4. However, given the time limit and the complexity, we will first try a few quick random restarts
    #    for rank 53 and 52 to see if we get lucky. If not, we return the schoolbook result.
    # 5. We will also try a simple "greedy" or "structured" reduction if possible, but for now,
    #    we focus on the random search as it's the most general.
    
    # To make the random search more effective, we can use a small set of values.
    # Let's try values from {-1, 0, 1} first, then {-2, -1, -0.5, 0, 0.5, 1, 2}.
    
    values = [-1, 0, 1]
    
    # Try ranks from rank_sb - 1 down to max(rank_sb - 5, 40)
    # We only try a few ranks to stay within the time limit.
    
    for target_rank in range(rank_sb - 1, max(rank_sb - 5, 40), -1):
        if time.time() > deadline:
            break
            
        max_restarts = 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization with small integer values
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            for idx in range(nm):
                for r in range(target_rank):
                    U[idx, r] = random.choice(values)
            for idx in range(mp):
                for r in range(target_rank):
                    V[idx, r] = random.choice(values)
            for idx in range(pn):
                for r in range(target_rank):
                    W[idx, r] = random.choice(values)
            
            # Check if this random assignment works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = 0.0
                        for r in range(target_rank):
                            s += U[a, r] * V[b, r] * W[c, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists
                U_list = [[float(U[i, r]) for r in range(target_rank)] for i in range(nm)]
                V_list = [[float(V[i, r]) for r in range(target_rank)] for i in range(mp)]
                W_list = [[float(W[i, r]) for r in range(target_rank)] for i in range(pn)]
                
                # Verify once more
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            s = 0.0
                            for r in range(target_rank):
                                s += U_list[a][r] * V_list[b][r] * W_list[c][r]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                    if not valid:
                        break
                
                if valid:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    return best_result
