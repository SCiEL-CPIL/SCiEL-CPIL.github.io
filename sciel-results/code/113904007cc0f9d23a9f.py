"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Schoolbook fallback (Rank 40)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Target Rank: Try 36 first. If not found, try 32? 
    # 36 is a reasonable guess for small improvements.
    # We use a discrete search over {-1, 0, 1}
    
    target_rank = 36
    
    # Pre-compute indices for equations
    # Equation for (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # We will track the current sum for each of the 40 equations.
    
    def initialize_random(rank):
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # Random initialization with sparse non-zeros to avoid immediate large errors
        # Density ~ 0.5
        for r in range(rank):
            # U column
            for idx in range(n * m):
                if random.random() < 0.5:
                    U[idx, r] = random.choice([-1, 1])
            # V column
            for idx in range(m * p):
                if random.random() < 0.5:
                    V[idx, r] = random.choice([-1, 1])
            # W column
            for idx in range(p * n):
                if random.random() < 0.5:
                    W[idx, r] = random.choice([-1, 1])
        return U, V, W

    def calculate_sums(U, V, W, rank):
        """
        Calculate the current sum for each equation (i, j, k).
        Returns an array of shape (n, m, p) containing the sums.
        """
        sums = np.zeros((n, m, p), dtype=int)
        # Reshape for easier access
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        for r in range(rank):
            # u: (n, m), v: (m, p), w: (p, n)
            u = U_r[:, :, r]
            v = V_r[:, :, r]
            w = W_r[:, :, r]
            
            # We need to add u[i,j] * v[j,k] * w[k,i] to sums[i,j,k]
            # Loop over i
            for i in range(n):
                ui = u[i, :] # (m,)
                wi = w[:, i] # (p,)
                # For each j
                for j in range(m):
                    uij = ui[j]
                    if uij == 0: continue
                    vj = v[j, :] # (p,)
                    # sums[i, j, :] += uij * vj * wi
                    # vj * wi is element-wise product of (p,) and (p,)
                    prod = vj * wi
                    sums[i, j, :] += uij * prod
        return sums

    def count_satisfied(sums):
        """Count how many equations are exactly equal to 1."""
        return np.sum(sums == 1)

    # Main Search Loop: Local Search on Discrete Grid
    # We try to find a solution for target_rank.
    
    best_U = None
    best_V = None
    best_W = None
    best_score = -1
    
    # Number of restarts
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        U, V, W = initialize_random(target_rank)
        sums = calculate_sums(U, V, W, target_rank)
        current_score = count_satisfied(sums)
        
        if current_score == n * m * p:
            best_U, best_V, best_W = U, V, W
            break
            
        if current_score > best_score:
            best_score = current_score
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        # Local Search: Flip one element at a time
        # Total elements: U (8*36), V (20*36), W (10*36)
        # We pick a random element and flip it (or change to 0/1/-1)
        
        # To speed up, we only update the sums affected by the change.
        # Changing U[i, j, r] affects all k for that (i, j, r).
        # Changing V[j, k, r] affects all i for that (j, k, r).
        # Changing W[k, i, r] affects all j for that (k, i, r).
        
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if current_score == n * m * p:
                best_U, best_V, best_W = U, V, W
                break
                
            # Choose which matrix to modify
            choice = random.random()
            if choice < 0.2: # U (8 vars per col)
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                r = random.randint(0, target_rank - 1)
                idx_u = i * m + j
                
                old_val = U[idx_u, r]
                # New value: random from {-1, 0, 1} excluding old
                options = [x for x in [-1, 0, 1] if x != old_val]
                new_val = random.choice(options)
                
                # Calculate delta for sums
                # sums[i, j, k] += (new_val - old_val) * V[j, p+k, r] * W[k, n+i, r]
                delta_val = new_val - old_val
                if delta_val == 0: continue
                
                v_col = V[:, r] # (m*p,)
                w_col = W[:, r] # (p*n,)
                
                # For this (i, j, r), k varies from 0 to p-1
                # V index: j*p + k
                # W index: k*n + i
                
                best_local_score = current_score
                best_local_new_val = old_val
                best_local_sums = None
                
                # Try all 3 possible values for this element
                for cand_val in [-1, 0, 1]:
                    if cand_val == old_val:
                        continue
                    d = cand_val - old_val
                    if d == 0:
                        continue
                    
                    # Apply change to a copy of sums? Or calculate new score directly?
                    # It's faster to calculate the new score for the affected equations only.
                    # Affected equations: (i, j, k) for k in 0..p-1
                    # Old sums: sums[i, j, :]
                    # New sums: sums[i, j, :] + d * (V[j*p+k, r] * W[k*n+i, r])
                    
                    affected_old = sums[i, j, :].copy()
                    # Construct the term to add
                    terms = np.zeros(p)
                    for k in range(p):
                        terms[k] = d * V[j * p + k, r] * W[k * n + i, r]
                    
                    affected_new = affected_old + terms
                    
                    # Count satisfied in affected region
                    old_sat = np.sum(affected_old == 1)
                    new_sat = np.sum(affected_new == 1)
                    
                    # Total score change
                    delta_score = new_sat - old_sat
                    
                    if delta_score > best_local_score - current_score: # Actually just track max delta
                        best_local_delta = delta_score
                        best_local_new_val = cand_val
                        best_local_affected_new = affected_new
                    
                    if delta_score > 0:
                        # Accept immediately if it improves? 
                        # Or just take the best improvement?
                        pass

                # Let's just pick the candidate with the best delta score
                # Re-doing the loop to find the best candidate cleanly
                best_delta = -100
                best_cand = old_val
                best_new_sums_part = None
                
                for cand_val in [-1, 0, 1]:
                    if cand_val == old_val:
                        continue
                    d = cand_val - old_val
                    if d == 0:
                        continue
                    
                    affected_old = sums[i, j, :]
                    terms = np.zeros(p)
                    for k in range(p):
                        terms[k] = d * V[j * p + k, r] * W[k * n + i, r]
                    affected_new = affected_old + terms
                    
                    old_sat = np.sum(affected_old == 1)
                    new_sat = np.sum(affected_new == 1)
                    delta_score = new_sat - old_sat
                    
                    if delta_score > best_delta:
                        best_delta = delta_score
                        best_cand = cand_val
                        best_new_sums_part = affected_new

                # Accept if best_delta > 0 or (best_delta == 0 and random < 0.5)
                if best_delta > 0 or (best_delta == 0 and random.random() < 0.5):
                    U[idx_u, r] = best_cand
                    sums[i, j, :] = best_new_sums_part
                    current_score += best_delta
                    if current_score > best_score:
                        best_score = current_score
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        
            elif choice < 0.5: # V (20 vars per col)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                r = random.randint(0, target_rank - 1)
                idx_v = j * p + k
                
                old_val = V[idx_v, r]
                
                best_delta = -100
                best_cand = old_val
                best_new_sums_part = None
                # Affected: (i, j, k) for i in 0..n-1
                # sums[i, j, k] += (new_val - old_val) * U[i*m+j, r] * W[k*n+i, r]
                
                for cand_val in [-1, 0, 1]:
                    if cand_val == old_val:
                        continue
                    d = cand_val - old_val
                    if d == 0:
                        continue
                    
                    terms = np.zeros(n)
                    for i in range(n):
                        terms[i] = d * U[i * m + j, r] * W[k * n + i, r]
                    
                    affected_old = sums[:, j, k] # shape (n,)
                    affected_new = affected_old + terms
                    
                    old_sat = np.sum(affected_old == 1)
                    new_sat = np.sum(affected_new == 1)
                    delta_score = new_sat - old_sat
                    
                    if delta_score > best_delta:
                        best_delta = delta_score
                        best_cand = cand_val
                        best_new_sums_part = affected_new

                if best_delta > 0 or (best_delta == 0 and random.random() < 0.5):
                    V[idx_v, r] = best_cand
                    sums[:, j, k] = best_new_sums_part
                    current_score += best_delta
                    if current_score > best_score:
                        best_score = current_score
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            else: # W (10 vars per col)
                k = random.randint(0, p - 1)
                i = random.randint(0, n - 1)
                r = random.randint(0, target_rank - 1)
                idx_w = k * n + i
                
                old_val = W[idx_w, r]
                
                best_delta = -100
                best_cand = old_val
                best_new_sums_part = None
                # Affected: (i, j, k) for j in 0..m-1
                # sums[i, j, k] += (new_val - old_val) * U[i*m+j, r] * V[j*p+k, r]
                
                for cand_val in [-1, 0, 1]:
                    if cand_val == old_val:
                        continue
                    d = cand_val - old_val
                    if d == 0:
                        continue
                    
                    terms = np.zeros(m)
                    for j in range(m):
                        terms[j] = d * U[i * m + j, r] * V[j * p + k, r]
                    
                    affected_old = sums[i, :, k] # shape (m,)
                    affected_new = affected_old + terms
                    
                    old_sat = np.sum(affected_old == 1)
                    new_sat = np.sum(affected_new == 1)
                    delta_score = new_sat - old_score if False else new_sat - old_sat # Typo guard
                    # Correct:
                    delta_score = new_sat - old_sat
                    
                    if delta_score > best_delta:
                        best_delta = delta_score
                        best_cand = cand_val
                        best_new_sums_part = affected_new

                if best_delta > 0 or (best_delta == 0 and random.random() < 0.5):
                    W[idx_w, r] = best_cand
                    sums[i, :, k] = best_new_sums_part
                    current_score += best_delta
                    if current_score > best_score:
                        best_score = current_score
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        # End local search for this restart
        if best_score == n * m * p:
            break

    # Check if we found a valid solution
    if best_score == n * m * p and best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fall back to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
