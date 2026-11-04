import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    def is_half_integer(x):
        # Check if x is close to a half-integer (k/2)
        # Multiply by 2, check if close to integer
        y = x * 2
        return abs(y - round(y)) < 1e-6

    def round_to_half_integer(x):
        return round(x * 2) / 2.0

    def solve_for_w(A):
        # A is m x R. Solve A w = 1
        # Use least squares
        try:
            w, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
            return w
        except np.linalg.LinAlgError:
            return None

    # Search for lower rank
    # Try ranks from 20 to best_rank - 1
    # Prioritize lower ranks
    
    # We can use a heuristic: 
    # 1. Random U, V with entries in {-1, 0, 1}
    # 2. Solve for W
    # 3. Check if W is close to half-integers
    # 4. If yes, round and verify
    
    # To improve success rate, we can try to make the system A w = 1 
    # have integer solutions.
    
    for rank in range(20, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Number of trials depends on time left
        # Let's do 1000 trials per rank
        for trial in range(1000):
            if time.time() > deadline - 10:
                break
                
            # Generate random U, V
            # Use {-1, 0, 1} for sparsity
            U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(m*p)]
            
            # Solve for W
            W = [[0.0] * rank for _ in range(p*n)]
            valid = True
            for k in range(p):
                for i in range(n):
                    # Construct A (m x rank)
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i*m+j][r] * V[j*p+k][r]
                    
                    w = solve_for_w(A)
                    if w is None:
                        valid = False
                        break
                    W[k*n+i] = w
            if not valid:
                continue
                
            # Check if W is close to half-integers
            w_is_half = True
            for ki in range(p*n):
                for r in range(rank):
                    if not is_half_integer(W[ki][r]):
                        w_is_half = False
                        break
                if not w_is_half:
                    break
            
            if not w_is_half:
                continue
                
            # Round W to half-integers
            W_rounded = [[round_to_half_integer(W[ki][r]) for r in range(rank)] for ki in range(p*n)]
            
            # Verify exact decomposition
            if verify(U, V, W_rounded, rank):
                best = {"U": U, "V": V, "W": W_rounded}
                best_rank = rank
                break # Move to next (lower) rank? No, this rank is found. 
                       # We can break out of trial loop for this rank.
                       # Since we iterate rank from low to high, the first found is minimal for this search.
                       # But we should check lower ranks first.
                       # The loop is `for rank in range(20, best_rank)`.
                       # So if we find rank 25, we update best_rank to 25.
                       # Then the loop continues to 26? No, range is fixed.
                       # We should restart the rank loop or just break and return?
                       # If we find a valid rank, it's better than schoolbook.
                       # We can try to find even lower?
                       # The loop goes 20, 21, ...
                       # If we find 25, we should continue checking 20-24?
                       # No, the loop is already at 25.
                       # We should have iterated from 20 upwards.
                       # So if we find 25, we haven't checked 20-24 yet?
                       # Wait, `for rank in range(20, best_rank)`
                       # If best_rank is 40, it checks 20, 21, ..., 39.
                       # If it finds 25, it updates best_rank to 25.
                       # But the iterator is already past 20-24.
                       # So we should restart the search from 20 to best_rank-1?
                       # Or just break and return?
                       # Given time limits, finding ANY rank < 40 is a win.
                       # Finding the MINIMAL rank is better.
                       # Let's restart the rank loop if we find a better rank.
            
            # Actually, let's just break out of the trial loop for this rank.
            # And then the outer loop will continue to higher ranks?
            # No, we want lower ranks.
            # If we find rank 25, we should check if 20-24 are possible.
            # So we should `break` the trial loop, and then `break` the rank loop?
            # And restart with new best_rank?
            
            # Let's just break out of everything and return the best found so far.
            # Or, to be thorough, we can restart the rank search.
            
            break # Break trial loop
        
        # If we found a better rank, we can restart the rank search
        # to see if we can go lower.
        # But time is limited.
        # Let's just continue to the next rank in the current loop?
        # No, if we found 25, the loop is at 25.
        # It will go to 26, 27... which are worse.
        # So we should break the rank loop and restart from 20 to best_rank-1.
        
        # Simple approach:
        # If best_rank < rank_school, restart the whole rank loop.
        if best_rank < rank_school:
             # Restart rank search with updated best_rank
             # We can do this by breaking and using a while loop or recursion?
             # Let's just break and rely on the fact that we found something.
             # To find the MINIMAL, we should search 20..best_rank-1.
             # Since we are already in a loop, let's just break and return.
             # The score is rank. Lower is better.
             # If we found 25, score is 1.0 * (40/25) = 1.6?
             # No, score is seed/value. Seed is 40?
             # "score is the rank".
             # "score is seed/value, so 1.0 equals the seed program".
             # Seed program returns rank 40. So seed value is 40.
             # If we return rank 25, value is 25. Score = 40/25 = 1.6.
             # So finding ANY rank < 40 is an improvement.
             # Finding the minimal rank gives the best score.
             
             # Let's try to restart the search for lower ranks.
             # We can use a flag or just break and let the function return.
             # To maximize score, we should find the lowest possible rank.
             
             # Let's break out of the rank loop and restart.
             break
             
    # If we broke out, we should try again with the new best_rank?
    # To keep it simple and within time, we will just return the best found.
    # If we want to be smarter, we can wrap the rank loop in a while True with a restart condition.
    
    return best
