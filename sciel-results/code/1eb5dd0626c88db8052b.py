import numpy as np
import random
import time

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
    time_limit = 110  # seconds
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # We search for ranks from 30 up to 55.
    # We use a heuristic: Randomize U, V (sparse), solve for W, check integrality.
    # To increase success rate, we try different sparsities and more restarts.
    
    # Ranks to try. Lower is better.
    ranks_to_try = list(range(30, 56))
    
    # Shuffle ranks slightly to avoid getting stuck in a bad local search pattern?
    # No, we want the minimal rank, so we must check in increasing order.
    # However, if we fail at 30, we move to 31, etc.
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        # Number of restarts depends on time remaining
        # Estimate time per restart: building A (60x15R), lstsq.
        # For R=30, A is 60x450. Very fast.
        # Let's do 5000 restarts per rank if time permits.
        
        num_restarts = 5000
        attempts = 0
        
        while attempts < num_restarts:
            attempts += 1
            if time.time() - start_time > time_limit:
                break
                
            # Randomize U and V
            # Use sparse random matrices with values {-1, 0, 1}
            # Density 0.2 to 0.5
            density = random.uniform(0.2, 0.5)
            
            U_rand = [[0] * R for _ in range(nm)]
            V_rand = [[0] * R for _ in range(mp)]
            
            for i in range(nm):
                for c in range(R):
                    if random.random() < density:
                        U_rand[i][c] = random.choice([-1, 1])
                        
            for j in range(mp):
                for c in range(R):
                    if random.random() < density:
                        V_rand[j][c] = random.choice([-1, 1])
            
            # Build A and b
            # A shape: (n*m*p, R * n * p)
            # b shape: (n*m*p,)
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            coeff = U_rand[i * m + j][c] * V_rand[j * p + k][c]
                            if coeff != 0:
                                w_idx = (k * n + i) * R + c
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            # Solve A W_flat = b
            try:
                # Use lstsq
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if W_sol is close to half-integers
            # Round to nearest 0.5
            W_rounded = np.round(W_sol * 2) / 2
            
            # Check if the rounded solution is exact
            # We need to verify with the rounded values
            # Convert to list for verify_decomposition
            W_check = W_rounded.tolist()
            
            if verify_decomposition(U_rand, V_rand, W_check, R):
                best_rank = R
                best_U = U_rand
                best_V = V_rand
                best_W = W_check
                # Found a valid one for this rank. 
                # Since we iterate R in increasing order, this is the minimal found so far.
                # We can return immediately if we assume we want the first found minimal?
                # Yes, because we are checking 30, 31, 32...
                # If we find 30, we return. If not, we try 31.
                return {"U": best_U, "V": best_V, "W": best_W}

        # If we didn't find a solution for this R in the allocated time/restarts, move to next R
        # But we need to be careful with time.
        if time.time() - start_time > time_limit - 5:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
