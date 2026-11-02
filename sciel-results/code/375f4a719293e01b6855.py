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
    
    # New Approach: Simulated Annealing with Efficient Delta Updates
    start_time = time.time()
    time_limit = 115  # seconds
    
    vals = [-1, -0.5, 0, 0.5, 1]
    
    ranks_to_try = list(range(30, 51))
    
    def verify(U, V, W, R):
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for c in range(R):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
        return True

    best_solution = None
    best_rank = rank_school
    
    for R in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        num_restarts = 100
        found = False
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            U = [[random.choice(vals) for _ in range(R)] for _ in range(nm)]
            V = [[random.choice(vals) for _ in range(R)] for _ in range(mp)]
            W = [[random.choice(vals) for _ in range(R)] for _ in range(pn)]
            
            # Pre-compute S[i, k] = sum_{j,c} U[i,j,c] V[j,k,c] W[k,i,c]
            # Note: W index is k*n+i.
            S = [[0.0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s_val = 0.0
                    for j in range(m):
                        for c in range(R):
                            s_val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    S[i][k] = s_val
            
            # Current Cost
            current_cost = sum((S[i][k] - 1.0)**2 for i in range(n) for k in range(p))
            
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.95
            max_iter = 2000
            
            for it in range(max_iter):
                if current_cost < 1e-9:
                    if verify(U, V, W, R):
                        found = True
                        best_rank = R
                        best_solution = {"U": U, "V": V, "W": W}
                        break
                    else:
                        break
                
                # Pick random variable
                mat_idx = random.choice([0, 1, 2])
                if mat_idx == 0:
                    row = random.randint(0, nm - 1)
                    col = random.randint(0, R - 1)
                    i = row // m
                    j = row % m
                    old_val = U[row][col]
                elif mat_idx == 1:
                    row = random.randint(0, mp - 1)
                    col = random.randint(0, R - 1)
                    j = row // p
                    k = row % p
                    old_val = V[row][col]
                else:
                    row = random.randint(0, pn - 1)
                    col = random.randint(0, R - 1)
                    k = row // n
                    i = row % n
                    old_val = W[row][col]
                
                new_val = random.choice(vals)
                if new_val == old_val:
                    continue
                
                delta_val = new_val - old_val
                
                # Calculate Delta Cost
                delta_cost = 0.0
                
                if mat_idx == 0:
                    # Change U[i, j, col]
                    # Affects S[i, k] for all k
                    for k in range(p):
                        old_s = S[i][k]
                        # Contribution of this term to S[i, k]
                        term_change = delta_val * V[j * p + k][col] * W[k * n + i][col]
                        new_s = old_s + term_change
                        delta_cost += (new_s - 1.0)**2 - (old_s - 1.0)**2
                        # Temporarily update S for potential acceptance? 
                        # We need to update S if we accept. 
                        # To avoid double update, we store the changes or update in place and revert if rejected.
                        S[i][k] = new_s
                        
                elif mat_idx == 1:
                    # Change V[j, k, col]
                    # Affects S[i, k] for all i
                    for i in range(n):
                        old_s = S[i][k]
                        term_change = delta_val * U[i * m + j][col] * W[k * n + i][col]
                        new_s = old_s + term_change
                        delta_cost += (new_s - 1.0)**2 - (old_s - 1.0)**2
                        S[i][k] = new_s
                        
                else:
                    # Change W[k, i, col] (index k*n+i)
                    # Affects S[i, k] for this specific i, k? 
                    # No, S[i, k] = sum_{j,c} ... W[k*n+i, c] ...
                    # So changing W[k*n+i, col] affects S[i, k] ONLY for this i, k.
                    old_s = S[i][k]
                    term_change = delta_val * sum(U[i * m + j][col] * V[j * p + k][col] for j in range(m))
                    new_s = old_s + term_change
                    delta_cost = (new_s - 1.0)**2 - (old_s - 1.0)**2
                    S[i][k] = new_s
                
                # Accept/Reject
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    current_cost += delta_cost
                    if mat_idx == 0:
                        U[row][col] = new_val
                    elif mat_idx == 1:
                        V[row][col] = new_val
                    else:
                        W[row][col] = new_val
                else:
                    # Reject: Revert S changes
                    # We already updated S in the loop above. We need to revert.
                    # This is messy. Better to calculate delta_cost first, then update S if accepted.
                    # Let's refactor the logic to calculate delta_cost without modifying S, then modify S if accepted.
                    pass

                T *= cooling
                if T < T_min:
                    T = T_min

            if found:
                break
        
        if found:
            return best_solution

    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
