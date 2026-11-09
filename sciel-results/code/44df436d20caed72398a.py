import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin from 150s limit
    
    # Schoolbook baseline: rank = n*m*p = 54
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
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Verify helper
    def verify(U, V, W, R):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    # Try to find better decompositions using structured search
    # Key insight: use small integer/half-integer values and exploit symmetry
    
    # Strategy: Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    
    # For each rank, try structured random search
    values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    for R in range(28, min(rank_sb + 1, 45)):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        max_attempts = 200 if R < 35 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random values from our set
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Use a more structured initialization
            for idx in range(n * m):
                for r in range(R):
                    U[idx, r] = random.choice(values)
            for idx in range(m * p):
                for r in range(R):
                    V[idx, r] = random.choice(values)
            for idx in range(p * n):
                for r in range(R):
                    W[idx, r] = random.choice(values)
            
            # Check if this works
            if verify(U, V, W, R):
                best_rank = R
                best = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                return best
        
        # If we've tried many ranks without success, break early
        if R > 40:
            break
    
    # Try a more targeted approach: exploit the block structure
    # Split p=9 into 3 blocks of 3
    # For each block, we need a 2x3x3 decomposition
    # If we can find a rank-r decomposition for 2x3x3, total rank = 3*r
    
    # Known: schoolbook for 2x3x3 has rank 18
    # Let's try to find lower rank for 2x3x3
    
    # For 2x3x3: T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # Need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Try small ranks for 2x3x3
    for r in range(10, 19):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            U3 = np.zeros((2 * 3, r))
            V3 = np.zeros((3 * 3, r))
            W3 = np.zeros((3 * 2, r))
            
            for idx in range(2 * 3):
                for rr in range(r):
                    U3[idx, rr] = random.choice(values)
            for idx in range(3 * 3):
                for rr in range(r):
                    V3[idx, rr] = random.choice(values)
            for idx in range(3 * 2):
                for rr in range(r):
                    W3[idx, rr] = random.choice(values)
            
            # Check 2x3x3
            U3r = U3.reshape(2, 3, r)
            V3r = V3.reshape(3, 3, r)
            W3r = W3.reshape(3, 2, r)
            T3 = np.einsum('ijr,jkr,kir->ijk', U3r, V3r, W3r)
            
            if np.allclose(T3, np.ones((2, 3, 3)), atol=1e-9):
                # Build full 2x3x9 by repeating this pattern 3 times
                total_rank = 3 * r
                U_full = np.zeros((n * m, total_rank))
                V_full = np.zeros((m * p, total_rank))
                W_full = np.zeros((p * n, total_rank))
                
                for block in range(3):
                    for i in range(n):
                        for j in range(m):
                            U_full[i * m + j, block * r:(block + 1) * r] = U3[i * 3 + j, :]
                        for j in range(m):
                            for k in range(3):
                                V_full[j * p + (block * 3 + k), block * r:(block + 1) * r] = V3[j * 3 + k, :]
                        for k in range(3):
                            for i in range(n):
                                W_full[(block * 3 + k) * n + i, block * r:(block + 1) * r] = W3[k * 2 + i, :]
                
                if verify(U_full, V_full, W_full, total_rank):
                    best_rank = total_rank
                    best = {
                        "U": [[float(x) for x in row] for row in U_full],
                        "V": [[float(x) for x in row] for row in V_full],
                        "W": [[float(x) for x in row] for row in W_full]
                    }
                    return best
    
    return best
