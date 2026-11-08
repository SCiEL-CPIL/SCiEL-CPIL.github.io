"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Pre-compute index triples for the tensor
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_solution = None
    best_rank = rank_school
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, n*p) = 28
    # We'll search from 28 to 111
    for R in range(28, rank_school + 1):
        if time.time() > deadline - 15:
            break
            
        # Number of restarts depends on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Estimate time per restart: roughly 100ms to 1s depending on convergence
        # Let's allow up to 10 restarts per rank, but stop if time runs out
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W with random small integers
            # Using small integers helps with exactness and convergence
            U = [[random.randint(-3, 3) for _ in range(R)] for _ in range(n * m)]
            V = [[random.randint(-3, 3) for _ in range(R)] for _ in range(m * p)]
            W = [[random.randint(-3, 3) for _ in range(R)] for _ in range(p * n)]
            
            # Local search
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                # Compute errors for all triples
                errors = {}
                valid = True
                for i, j, k in triples:
                    s = 0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        valid = False
                        errors[(i, j, k)] = s - 1
                            
                if valid:
                    return {"U": U, "V": V, "W": W}
                
                # Pick a random error
                if not errors:
                    continue
                    
                (i, j, k), err = random.choice(list(errors.items()))
                
                # Try to fix by adjusting one variable in a random rank slot
                r = random.randint(0, R - 1)
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                u_val = U[u_idx][r]
                v_val = V[v_idx][r]
                w_val = W[w_idx][r]
                
                # Try adjusting U
                if v_val != 0 and w_val != 0:
                    denom = v_val * w_val
                    # Current contribution: u_val * denom
                    # Total sum S = 1 + err
                    # New sum S' = S - u_val*denom + new_u*denom
                    # We want S' = 1 => new_u*denom = 1 - S + u_val*denom = -err + u_val*denom
                    target = -err + u_val * denom
                    if target % denom == 0:
                        new_u = target // denom
                        if new_u != u_val and -20 <= new_u <= 20:
                            U[u_idx][r] = new_u
                            continue
                            
                # Try adjusting V
                if u_val != 0 and w_val != 0:
                    denom = u_val * w_val
                    target = -err + v_val * denom
                    if target % denom == 0:
                        new_v = target // denom
                        if new_v != v_val and -20 <= new_v <= 20:
                            V[v_idx][r] = new_v
                            continue
                            
                # Try adjusting W
                if u_val != 0 and v_val != 0:
                    denom = u_val * v_val
                    target = -err + w_val * denom
                    if target % denom == 0:
                        new_w = target // denom
                        if new_w != w_val and -20 <= new_w <= 20:
                            W[w_idx][r] = new_w
                            continue
                            
                # If no single variable adjustment worked, try a random perturbation
                # to escape local minima
                if random.random() < 0.1:
                    var_choice = random.randint(0, 2)
                    if var_choice == 0:
                        U[u_idx][r] = random.randint(-3, 3)
                    elif var_choice == 1:
                        V[v_idx][r] = random.randint(-3, 3)
                    else:
                        W[w_idx][r] = random.randint(-3, 3)
                            
            # End local search
        # End restarts
    # End ranks
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
