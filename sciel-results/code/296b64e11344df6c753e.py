import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for the evaluator's 120s limit
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Coefficients for U and V
    # Using {-1, 1} is often better for Strassen-like structures than including 0, 
    # but sparsity helps solvability. We'll mix.
    
    def is_rational(matrix, max_denom=4, tol=1e-9):
        """
        Checks if all entries in matrix are close to rationals k/d 
        where d is in {1, 2, 3, 4, ..., max_denom}.
        Returns the rounded matrix if valid, else None.
        """
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                found = False
                for d in range(1, max_denom + 1):
                    num = round(val * d)
                    # Check if val is close to num/d
                    if abs(val - num / d) < tol:
                        # Store as float for now, or Fraction? 
                        # The interface expects integer/half-integer. 
                        # The problem statement says "integer/half-integer entries".
                        # So we should strictly check for d=1 and d=2.
                        # If d > 2, it's not allowed by the contract?
                        # "integer/half-integer entries" -> d=1 or d=2.
                        if d == 1 or d == 2:
                            res_row.append(float(num / d))
                            found = True
                            break
                if not found:
                    return None
            result.append(res_row)
        return result

    def verify_decomposition(U, V, W, rank):
        """
        Verifies sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k.
        U shape: (n*m, rank) -> index i*m+j
        V shape: (m*p, rank) -> index j*p+k
        W shape: (p*n, rank) -> index k*n+i
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def solve_for_W(U, V, R):
        """
        Solves for W given U and V.
        Returns W as a list of lists if a valid half-integer/integer solution is found, else None.
        """
        # A has shape (n*m*p, R*n*p)
        # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Unknowns: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..R-1
        # Total unknowns: R * p * n
        # Total equations: n * m * p
        
        num_eqs = n * m * p
        num_unk = R * p * n
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # For this equation (i, j, k), the variables involved are W[k*n+i, c] for all c
                    # Coefficient for W[k*n+i, c] is U[i*m+j, c] * V[j*p+k, c]
                    for c in range(R):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        if coeff != 0:
                            # Index in W vector: (k*n+i) * R + c
                            w_idx = (k * n + i) * R + c
                            A[eq_idx, w_idx] = coeff
                    eq_idx += 1
        
        try:
            # Use lstsq
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check residual
            if residuals and residuals[0] > 1e-8:
                return None
            
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if W_sol is integer or half-integer
            W_nice = is_rational(W_sol, max_denom=2, tol=1e-9)
            if W_nice is not None:
                return W_nice
            else:
                return None
                
        except:
            return None

    # Search ranks
    # We iterate from a reasonable lower bound up to schoolbook-1
    # Known ranks: 2x2x2 is 7. 3x3x3 is ~23.
    # 3x4x5 is likely < 40.
    # We try 30 to 59.
    
    # To maximize chances, we prioritize lower ranks but allow more time for higher ones?
    # No, we want the MINIMAL rank. So we must check low ranks first.
    # If we find rank 35, we return it. We don't care about 34 if we already have 35?
    # Wait, if we find 35, we should keep searching for 34?
    # The problem asks for "rank-minimal". If we find 35, we should try to find 34.
    # However, if we run out of time, we return the best found.
    # Strategy: Iterate R from 30 to 59.
    # For each R, do a limited number of restarts.
    # If we find a valid decomposition for R, we update best_rank and best_U/V/W.
    # We do NOT return immediately, because a lower R might exist.
    # BUT, if we are iterating R in increasing order (30, 31, 32...), 
    # and we find a valid one for R=35, we know R=30..34 failed (or were not fully explored).
    # To be safe, we should probably do a full pass for low R, then higher R.
    # Given time constraints, we will iterate R from 30 to 59.
    # We will allocate a time budget per R.
    
    ranks_to_try = list(range(30, 60))
    
    # Shuffle ranks? No, we want minimal. So 30, 31, 32...
    # But 30 might be impossible. 40 might be easy.
    # If we spend 10s on 30 and find nothing, then 10s on 31...
    # We have ~110s. 30 ranks. ~3s per rank.
    # 3s is not much.
    
    # Alternative: Randomize the order of ranks but weighted towards lower?
    # Or just try a few specific promising ranks?
    # Let's try 35, 40, 45, 50, 55, 59 first? No, that violates "minimal".
    
    # Let's stick to increasing order but with a small number of restarts per rank initially,
    # and if we find nothing, we move on.
    
    num_restarts_per_rank = 200
    
    for R in ranks_to_try:
        if R >= best_rank:
            # If we've already found a better rank, we can skip higher ones?
            # No, we are going up. If best_rank is 40, and we are at R=41, we can stop?
            # Yes, because we are looking for MINIMAL. If we found 40, 41 is worse.
            # But we are iterating 30, 31... 59.
            # If we find 40 at R=40, we update best_rank=40.
            # Next R=41. R >= best_rank (41 >= 40) is True. We break the loop?
            # We should break the outer loop because any R > best_rank is not minimal.
            break
            
        if time.time() - start_time > time_limit:
            break
            
        # Determine time budget for this rank
        # If we have spent a lot of time, reduce restarts
        time_elapsed = time.time() - start_time
        time_remaining = time_limit - time_elapsed
        # Simple heuristic: if < 10s left, do very few restarts
        if time_remaining < 5:
             break
             
        current_restarts = num_restarts_per_rank
        if time_remaining < 20:
            current_restarts = 50
        elif time_remaining < 40:
            current_restarts = 100
            
        for r_idx in range(current_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V
            # Use {-1, 1} with some probability of 0?
            # Strassen uses -1, 0, 1.
            # Let's try a mix: 70% chance of -1/1, 30% chance of 0.
            
            U_rand = [[0] * R for _ in range(nm)]
            V_rand = [[0] * R for _ in range(mp)]
            
            for i in range(nm):
                for c in range(R):
                    if random.random() < 0.7:
                        U_rand[i][c] = random.choice([-1, 1])
                    else:
                        U_rand[i][c] = 0
                        
            for j in range(mp):
                for c in range(R):
                    if random.random() < 0.7:
                        V_rand[j][c] = random.choice([-1, 1])
                    else:
                        V_rand[j][c] = 0
            
            W_sol = solve_for_W(U_rand, V_rand, R)
            if W_sol is not None:
                # Verify
                if verify_decomposition(U_rand, V_rand, W_sol, R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_sol
                    # We found a valid decomposition for rank R.
                    # Since we are iterating R in increasing order, this is the minimal so far.
                    # We should continue searching for smaller R? No, we are at R.
                    # We will break the inner loop and the outer loop will break 
                    # because next R will be > best_rank.
                    # So we can return immediately?
                    # Yes, because we are iterating 30, 31... 
                    # If we find one at 35, we return 35. 
                    # We don't need to check 36-59.
                    return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
