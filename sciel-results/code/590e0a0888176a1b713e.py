import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Pre-compute index mappings
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Target tensor T is all 1s for the multiplication pattern?
    # Yes, T[i,j,k] = 1.
    
    # We search for rank r
    # Lower bound: 
    # R(3,3,3) >= 19? 
    # Let's start from 20 to be safe, or maybe 15.
    # If we find a solution, we return it.
    # We want the MINIMAL rank.
    # So we should iterate r from low to high.
    
    # However, finding a solution for small r might be hard.
    # If we don't find one in time, we should return the best found (lowest r) or fall back to schoolbook?
    # The score is 1.0 for schoolbook (rank 63).
    # Any rank < 63 is an improvement (score > 1.0).
    # So we want to find the smallest r < 63.
    
    best_solution = None
    best_rank = 63
    
    # Schoolbook solution as fallback
    U_school = [[0]*63 for _ in range(9)]
    V_school = [[0]*63 for _ in range(21)]
    W_school = [[0]*63 for _ in range(21)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 63

    # Search for smaller ranks
    # Try ranks from 20 to 62
    for r in range(20, 63):
        if time.time() > deadline - 10:
            break
            
        # Number of restarts for this rank
        # If r is small, we might need more restarts?
        # Let's try 100 restarts per rank.
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random U and V
            # Entries from -1, 0, 1
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            
            # Construct A (63 x 21*r)
            # Equation index e = i*p*(m) + j*p + k ? 
            # Let's use a linear index for (i,j,k)
            # i in 0..2, j in 0..2, k in 0..6
            # e = i * (m*p) + j * p + k
            
            # Variable index v = row * r + c
            # row = k*n + i
            # c in 0..r-1
            
            A = np.zeros((n*m*p, p*n*r))
            b = np.ones(n*m*p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i * (m*p) + j * p + k
                        row_w = k * n + i
                        for c in range(r):
                            v = row_w * r + c
                            coeff = U[i*m + j, c] * V[j*p + k, c]
                            A[e, v] = coeff
            
            # Solve A x = b
            # Use lstsq
            try:
                x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check if x is close to half-integers
            # 2x should be close to integers
            x_2 = 2 * x
            x_rounded = np.round(x_2)
            diff = np.abs(x_2 - x_rounded)
            
            if np.max(diff) < 1e-6:
                # Snap to half-integers
                W_flat = x_rounded / 2.0
                W = W_flat.reshape(p*n, r)
                
                # Verify exact decomposition
                # Check sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
                    
                if valid:
                    # Found a solution with rank r
                    best_rank = r
                    # Convert to lists of floats (or ints if possible)
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    # Since we are iterating r upwards, the first valid one is the minimal?
                    # Not necessarily, but we are going 20, 21, ...
                    # So this is the minimal rank found so far.
                    # We can break out of restart loop, but continue to lower r?
                    # No, we are going UP. So we want to check if we can go lower?
                    # We started at 20. If we found one at 25, maybe there is one at 22?
                    # But we checked 20, 21, 22, 23, 24 and failed.
                    # So 25 is the best so far.
                    # We should continue to see if we can find a lower one?
                    # No, we already tried lower ones.
                    # So we can return this?
                    # But maybe a random restart at r=22 would have worked?
                    # We did 200 restarts.
                    # It's a trade-off.
                    # Let's just return the first valid one found in the increasing r loop.
                    # This ensures we get a good score quickly.
                    # If we want to be sure, we could do a few more restarts at lower r?
                    # But time is limited.
                    # Returning the first valid found in increasing r order is a good heuristic.
                    
                    # However, we might have skipped r=20-24 due to bad luck.
                    # If we find r=25, it's better than 63.
                    # Score = 63/25 = 2.52.
                    # If we find r=20, score = 3.15.
                    # Let's return immediately to save time, as lower r is unlikely to be found with more restarts if we already tried.
                    # Actually, we should probably try a bit harder for lower ranks?
                    # No, the loop structure is: for r in 20..62.
                    # If we find r=25, we return.
                    # We don't go back to 20.
                    # This is fine.
                    
                    # One optimization: if we find a solution, we can return it.
                    return best_solution

    # If no solution found, return schoolbook
    return best_solution
