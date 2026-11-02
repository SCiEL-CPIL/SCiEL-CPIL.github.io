"""
Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.

Uses block structure: split middle dimension (4) into two 2s, giving two
3x2x5 subproblems. Search for low-rank decompositions of 3x2x5 tensors.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

def verify_full(U, V, W, n, m, p, tol=1e-10):
    """Verify the full 3x4x5 decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try block decomposition: split m=4 into two 2s
    # Subproblem 1: 3x2x5 (j=0,1)
    # Subproblem 2: 3x2x5 (j=2,3)
    
    # For each subproblem, try to find a low-rank decomposition
    # Then combine them
    
    # If we find rank r1 for sub1 and r2 for sub2, total rank = r1 + r2
    
    # Search for rank-15 decomposition of 3x2x5 (schoolbook is 30)
    # Use random search with small rational values
    
    random.seed(123)
    best_total_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    for target_sub_rank in range(25, 10, -1):
        if time.time() > deadline - 30:
            break
        
        attempts = 20
        for attempt in range(attempts):
            if time.time() > deadline - 25:
                break
            
            # Generate random sub-decompositions for both 3x2x5 blocks
            U1 = [[0] * target_sub_rank for _ in range(3 * 2)]  # 6 rows
            V1 = [[0] * target_sub_rank for _ in range(2 * 5)]  # 10 rows
            W1 = [[0] * target_sub_rank for _ in range(5 * 3)]  # 15 rows
            
            U2 = [[0] * target_sub_rank for _ in range(3 * 2)]
            V2 = [[0] * target_sub_rank for _ in range(2 * 5)]
            W2 = [[0] * target_sub_rank for _ in range(5 * 3)]
            
            # Fill with random ±1 entries sparsely
            for r in range(target_sub_rank):
                for _ in range(random.randint(2, 5)):
                    row = random.randint(0, 5)
                    U1[row][r] = random.choice([-1, 1])
                for _ in range(random.randint(2, 5)):
                    row = random.randint(0, 9)
                    V1[row][r] = random.choice([-1, 1])
                for _ in range(random.randint(2, 5)):
                    row = random.randint(0, 14)
                    W1[row][r] = random.choice([-1, 1])
                
                for _ in range(random.randint(2, 5)):
                    row = random.randint(0, 5)
                    U2[row][r] = random.choice([-1, 1])
                for _ in range(random.randint(2, 5)):
                    row = random.randint(0, 9)
                    V2[row][r] = random.choice([-1, 1])
                for _ in range(random.randint(2, 5)):
                    row = random.randint(0, 14)
                    W2[row][r] = random.choice([-1, 1])
            
            # Combine into full 3x4x5 decomposition
            total_rank = 2 * target_sub_rank
            U_full = [[0] * total_rank for _ in range(n * m)]  # 12 rows
            V_full = [[0] * total_rank for _ in range(m * p)]  # 20 rows
            W_full = [[0] * total_rank for _ in range(p * n)]  # 15 rows
            
            # Sub1: j=0,1 -> U rows 0-5, V rows 0-9, W rows 0-14
            for i in range(3):
                for j in range(2):
                    for r in range(target_sub_rank):
                        U_full[i*4 + j][r] = U1[i*2 + j][r]
                for j in range(2):
                    for k in range(5):
                        for r in range(target_sub_rank):
                            V_full[j*5 + k][r] = V1[j*5 + k][r]
                for k in range(5):
                    for i in range(3):
                        for r in range(target_sub_rank):
                            W_full[k*3 + i][r] = W1[k*3 + i][r]
            
            # Sub2: j=2,3 -> U rows 6-11, V rows 10-19, W rows 0-14 (offset by target_sub_rank)
            for i in range(3):
                for j in range(2, 4):
                    for r in range(target_sub_rank):
                        U_full[i*4 + j][target_sub_rank + r] = U2[i*2 + (j-2)][r]
                for j in range(2, 4):
                    for k in range(5):
                        for r in range(target_sub_rank):
                            V_full[j*5 + k][target_sub_rank + r] = V2[(j-2)*5 + k][r]
                for k in range(5):
                    for i in range(3):
                        for r in range(target_sub_rank):
                            W_full[k*3 + i][target_sub_rank + r] = W2[k*3 + i][r]
            
            # Verify
            if verify_full(U_full, V_full, W_full, n, m, p):
                best_total_rank = total_rank
                best_U, best_V, best_W = U_full, V_full, W_full
                break
        
        if best_total_rank == 2 * target_sub_rank:
            break
    
    if best_total_rank < rank_sb:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
