"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=2, m=4, p=6
    # U: 8 x r
    # V: 24 x r
    # W: 12 x r
    
    # Schoolbook solution
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    if time.time() > deadline - 10:
        return best_result

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strategy: Random U, V -> Solve Linear System for W -> Check if W is half-integer
    # System: For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let W be a vector of length 12*r (flattened W matrix).
    # The coefficient for W[k*n+i, r] in equation (i,j,k) is U[i*m+j, r] * V[j*p+k, r].
    # Note: W[k*n+i, r] appears in ALL equations for that specific k and i, but with different j?
    # No, the equation for (i,j,k) involves W[k*n+i, r] for all r.
    # The coefficient of W[k*n+i, r] in the equation for (i,j,k) is U[i*m+j, r] * V[j*p+k, r].
    # So for a fixed (i,k), there are m equations (one for each j).
    # Each equation involves the same set of W variables (W[k*n+i, 0..r-1]).
    # This means the system is block-diagonal with respect to (i,k) pairs!
    # There are n*p = 2*6 = 12 such blocks.
    # Each block has m=4 equations and r unknowns.
    # So for each (i,k) pair, we have a 4 x r linear system:
    # A_{i,k} w_{i,k} = 1_vector (length 4)
    # where A_{i,k} is a 4 x r matrix, with A_{i,k}[j, r] = U[i*m+j, r] * V[j*p+k, r].
    # We can solve each block independently!
    # This is a HUGE simplification.
    
    # We need to find U, V (integer/half-integer) such that for each of the 12 (i,k) blocks,
    # the system A_{i,k} w = 1 has a solution w that is integer/half-integer.
    
    # To make it easier, we can fix U and V to be very sparse, e.g., only 1s and 0s.
    # If U and V are 0/1, then A_{i,k} is 0/1.
    # We want A_{i,k} w = 1.
    # If we choose A_{i,k} to be the identity matrix (or contain it), we can get w = 1 or 0.
    # For a 4 x r system, if r >= 4, we can easily find a 0/1 solution for w.
    # Specifically, if we can make A_{i,k} contain the 4x4 identity matrix as columns, then w can be 1 for those columns and 0 elsewhere.
    
    # So the problem reduces to:
    # Can we choose U (8 x r) and V (24 x r) with entries in {0, 1} such that
    # for every i in 0..1, k in 0..5, the matrix A_{i,k} (4 x r) with entries U[i*m+j, r]*V[j*p+k, r]
    # allows for a 0/1 solution w to A_{i,k} w = 1?
    # Actually, we don't need A to contain Identity. We just need the system to be consistent with a 0/1 w.
    # If we set w = 1 for all r, then we need sum_r U[i*m+j, r] V[j*p+k, r] = 1 for all j.
    # This means the dot product of row j of U_block and row j of V_block is 1.
    
    # Let's try a specific construction:
    # Let r = 12.
    # For each (i,k) block, we want to solve 4 equations.
    # If we assign one unique column r to each of the 4 equations in the block, we can set the corresponding U and V to 1, and 0 elsewhere for that column.
    # But columns are shared across blocks?
    # U[i*m+j, r] is shared across all k for fixed i,j.
    # V[j*p+k, r] is shared across all i for fixed j,k.
    # W[k*n+i, r] is specific to (k,i).
    
    # Let's try r = 24 (one column per j? no).
    # Let's try to construct a solution with r = 24.
    # We have 12 blocks (i,k). Each needs 4 equations solved.
    # Total 48 "slots" for equations.
    # If we use r=48, we can have 1 column per equation. This is schoolbook.
    # Can we share columns?
    # A column r contributes to equation (i,j,k) if U[i*m+j, r] != 0 and V[j*p+k, r] != 0.
    # If we set U[i*m+j, r] = 1 and V[j*p+k, r] = 1, it contributes 1 * W[k*n+i, r].
    # If we set W[k*n+i, r] = 1, it contributes 1 to equation (i,j,k).
    # If the same column r is used for multiple (i,j,k), it adds to multiple equations.
    # We want the sum to be 1.
    
    # Let's try r = 24.
    # We can try to use the fact that 2*4*6 = 48, and maybe halve it?
    # Let's use the "Linear Solve" strategy with random 0/1 U, V.
    # For each trial:
    # 1. Generate random 0/1 U (8 x r) and V (24 x r).
    # 2. For each (i,k) block (12 blocks):
    #    - Construct A (4 x r) where A[j, r] = U[i*m+j, r] * V[j*p+k, r].
    #    - Solve A w = 1 for w.
    #    - Check if w is 0/1 (or half-integer).
    # 3. If all blocks have valid w, construct W and verify.
    
    # To increase success rate, we can bias U and V to have more 1s.
    # Or, we can use a greedy approach:
    # Start with r=24.
    # For each block, we need 4 "units" of contribution.
    
    # Let's implement the random 0/1 search for r in [24, 47].
    
    values = [0, 1] # Try 0/1 first
    
    # We will try ranks from 47 down to 24.
    # Actually, let's start from 47 down to 40 to see if we can beat 48 easily.
    # Then go lower.
    
    for r in range(47, 23, -1):
        if time.time() > deadline - 10:
            break
        
        # Number of trials per rank
        max_trials = 500
        if r < 30:
            max_trials = 2000
            
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Generate random U, V with 0/1 entries
            # Bias: probability of 1 = 0.5
            U = [[random.choice(values) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice(values) for _ in range(r)] for _ in range(m * p)]
            
            # Solve for W
            W = [[0] * r for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct A (4 x r)
                    # A[j, r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                    # We need A w = 1 (vector of ones, length 4)
                    # This is a linear system.
                    # Since A is 0/1, and we want 0/1 w, we can check if the system has a 0/1 solution.
                    # For small r, we can brute force or use linear algebra.
                    # But r can be up to 47. 2^47 is too big.
                    # However, the system is 4 equations.
                    # We can iterate over possible w? No.
                    # We can solve the linear system over reals and check if it's close to 0/1?
                    # Or, we can check if the rows of A are linearly independent and the solution is unique?
                    
                    # Let's use a simpler check:
                    # Does there exist w in {0,1}^r such that sum_{r_idx: w[r_idx]=1} A[:, r_idx] = 1?
                    # This is the "subset sum" problem in 4D.
                    # We can use a meet-in-the-middle or just random sampling of w?
                    # Actually, if we pick U, V randomly, the columns of A are random 4-bit vectors.
                    # We want to sum a subset of these vectors to get (1,1,1,1).
                    # If r is large, this is likely possible.
                    
                    # Heuristic: Check if the all-ones vector is in the convex hull? No, 0/1.
                    # Let's try to solve A w = 1 using least squares and check residuals.
                    # If the residual is 0 and w is close to 0/1, it might work.
                    # But we need EXACT 0/1 (or half-int).
                    
                    # Better: For 4 equations, we can iterate over the null space?
                    # Or, we can just try to find a 0/1 solution by random sampling of w?
                    # No, let's use the fact that if A has a column of all 1s, we can set that w to 1 and others to 0.
                    # If not, maybe sum of two columns?
                    
                    # Let's check if the system is solvable with 0/1 w.
                    # We can use a simple backtracking for 4 equations?
                    # Or, we can check if the vector (1,1,1,1) is representable as a sum of columns of A.
                    
                    # To be fast, let's just solve A w = 1 using np.linalg.lstsq.
                    # If the solution is close to 0/1, we round and verify.
                    
                    A = np.array([[U[i * m + j][r_idx] * V[j * p + k][r_idx] for r_idx in range(r)] for j in range(m)])
                    b = np.ones(m)
                    
                    # Solve
                    w_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Check if w_sol is close to 0/1
                    # And if the reconstruction is exact
                    if np.allclose(A @ w_sol, b, atol=1e-9):
                        # Check if w_sol is close to 0/1
                        w_rounded = np.round(w_sol).astype(int)
                        if np.allclose(A @ w_rounded, b, atol=1e-9):
                            # Valid 0/1 solution found for this block
                            for r_idx in range(r):
                                W[k * n + i][r_idx] = w_rounded[r_idx]
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # If we got here, all blocks have a 0/1 W solution.
            # Verify the whole thing
            if verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                break
                
        if best_rank < 48:
            break
            
    return best_result

# EVOLVE-BLOCK-END
