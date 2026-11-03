"""Quadratic assignment solver for QAPLIB tai256c."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute QAP cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_init(a, b, n, rng):
    """Greedy construction: assign facilities one by one to minimize marginal cost."""
    # Precompute row sums of a
    a_row_sums = a.sum(axis=1)
    
    # Start with facility having largest flow sum
    order = np.argsort(-a_row_sums, kind='stable')
    
    p = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    
    for fac in order:
        if assigned[fac]:
            continue
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if assigned[loc]:
                continue
            # Marginal cost of placing fac at loc
            # sum_{j} a[fac,j]*b[loc,p[j]] + sum_{i} a[i,fac]*b[p[i],loc] + a[fac,fac]*b[loc,loc]
            # For unassigned j, we don't know yet, so approximate with row sums
            marginal = 0
            for j in range(n):
                if assigned[j]:
                    marginal += a[fac, j] * b[loc, p[j]] + a[j, fac] * b[p[j], loc]
            # Diagonal term
            marginal += a[fac, fac] * b[loc, loc]
            if marginal < best_cost:
                best_cost = marginal
                best_loc = loc
        if best_loc >= 0:
            p[fac] = best_loc
            assigned[fac] = True
    
    # If any unassigned (shouldn't happen), assign randomly
    unassigned = np.where(p == -1)[0]
    if len(unassigned) > 0:
        remaining_locs = [loc for loc in range(n) if not any(p == loc for loc in range(n))]
        random.shuffle(remaining_locs)
        for idx, fac in enumerate(unassigned):
            p[fac] = remaining_locs[idx]
    
    return p

def _local_search_2opt(a, b, p, time_limit):
    """2-opt local search with vectorized delta computation."""
    n = len(p)
    p = p.copy().astype(np.int64)
    best_cost = _qap_cost(a, b, p)
    start_time = time.time()
    
    # Precompute b at p
    bp = b[np.ix_(p, p)]
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        improved = False
        # Randomize order to escape local plateaus
        pairs = list(range(n))
        random.shuffle(pairs)
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                
                pi, pj = p[i], p[j]
                if pi == pj:
                    continue
                
                # Delta for swapping i and j
                # delta = sum_k (a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj]) 
                #       + sum_k (a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi])
                #       - sum_k (a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi])
                #       - sum_k (a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj])
                #       - a[i,j]*(b[pi,pj]+b[pj,pi]) + a[j,i]*(b[pj,pi]+b[pi,pj])
                # Simplify:
                # delta = sum_k (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
                #       + sum_k (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
                #       - a[i,j]*(b[pi,pj]+b[pj,pi]) + a[j,i]*(b[pj,pi]+b[pi,pj])
                
                # Vectorized computation
                diff_a_row = a[i, :] - a[j, :]
                diff_b_cols = bp[pj, :] - bp[pi, :]
                term1 = np.dot(diff_a_row, diff_b_cols)
                
                diff_a_col = a[:, i] - a[:, j]
                diff_b_rows = bp[:, pj] - bp[:, pi]
                term2 = np.dot(diff_a_col, diff_b_rows)
                
                # Direct i-j terms
                direct = -a[i, j] * (b[pi, pj] + b[pj, pi]) + a[j, i] * (b[pj, pi] + b[pi, pj])
                
                delta = int(term1 + term2 + direct)
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = pj, pi
                    best_cost += delta
                    bp = b[np.ix_(p, p)]
                    improved = True
                    break  # Restart inner loop after improvement
            
            if improved:
                break
        
        if not improved:
            break
    
    return p.tolist(), best_cost

def _simulated_annealing(a, b, p, time_limit):
    """Simulated annealing for QAP."""
    n = len(p)
    p = p.copy().astype(np.int64)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    T = 1000.0  # Initial temperature
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        # Choose random pair to swap
        i, j = random.sample(range(n), 2)
        if i == j:
            continue
        
        pi, pj = p[i], p[j]
        
        # Compute delta
        diff_a_row = a[i, :] - a[j, :]
        bp = b[np.ix_(p, p)]
        diff_b_cols = bp[pj, :] - bp[pi, :]
        term1 = np.dot(diff_a_row, diff_b_cols)
        
        diff_a_col = a[:, i] - a[:, j]
        diff_b_rows = bp[:, pj] - bp[:, pi]
        term2 = np.dot(diff_a_col, diff_b_rows)
        
        direct = -a[i, j] * (b[pi, pj] + b[pj, pi]) + a[j, i] * (b[pj, pi] + b[pi, pj])
        delta = int(term1 + term2 + direct)
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            p[i], p[j] = pj, pi
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling_rate
        if T < 1.0:
            break
    
    return best_p.tolist(), best_cost

def solve(a, b):
    """Solve the QAP instance. Returns permutation p where facility i is at location p[i]."""
    n = len(a)
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Step 1: Greedy initialization
    p_init = _greedy_init(a, b, n, rng)
    best_p = np.array(p_init, dtype=np.int64)
    best_cost = _qap_cost(a, b, best_p)
    
    # Step 2: Local search from greedy start
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 30:
        p_ls, cost_ls = _local_search_2opt(a, b, best_p, time_limit=min(45, remaining - 20))
        if cost_ls < best_cost:
            best_p = np.array(p_ls, dtype=np.int64)
            best_cost = cost_ls
    
    # Step 3: If time remains, try additional restarts
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 20:
        # Try a few random restarts with local search
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            # Perturb the best solution
            p_pert = best_p.copy()
            num_swaps = rng.randint(2, 5)
            for _ in range(num_swaps):
                i, j = rng.sample(range(n), 2)
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            time_left = deadline - time.time() - 10
            if time_left < 5:
                break
            
            p_new, cost_new = _local_search_2opt(a, b, p_pert, time_limit=max(5, time_left))
            if cost_new < best_cost:
                best_p = np.array(p_new, dtype=np.int64)
                best_cost = cost_new
    
    # Final check
    if time.time() < deadline - 5:
        # One more local search from current best
        p_final, cost_final = _local_search_2opt(a, b, best_p, time_limit=deadline - time.time() - 2)
        if cost_final < best_cost:
            best_p = np.array(p_final, dtype=np.int64)
            best_cost = cost_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
