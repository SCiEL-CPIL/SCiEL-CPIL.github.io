import numpy as np
import time

def solve(a, b):
    n = len(a)
    S = a + a.T
    
    # Vectorized Greedy Initialization
    p = np.full(n, -1, dtype=np.int64)
    assigned_locs = []
    assigned_facs = []
    
    order = np.argsort(-a.sum(axis=1), kind="stable")
    
    for idx in order:
        if len(assigned_locs) == 0:
            # First facility, cost is 0 for all locations? 
            # Or just pick first available.
            best_loc = 0
            for l in range(n):
                if p[l] == -1:
                    best_loc = l
                    break
        else:
            # Vectorized cost calculation for all unassigned locations
            # V[loc] = sum_{j in assigned} (a[idx, j] * b[loc, p[j]] + a[j, idx] * b[p[j], loc])
            
            # a_idx_j = a[idx, assigned_facs]
            # a_j_idx = a[assigned_facs, idx]
            # p_j = p[assigned_facs]
            
            # Term 1: sum_j a[idx, j] * b[loc, p[j]]
            # b_loc_pj = b[:, p_j] # shape (n, k)
            # term1 = (b_loc_pj * a_idx_j).sum(axis=1) # shape (n,)
            
            # Term 2: sum_j a[j, idx] * b[p[j], loc]
            # b_pj_loc = b[p_j, :] # shape (k, n)
            # term2 = (b_pj_loc * a_j_idx[:, None]).sum(axis=0) # shape (n,)
            
            # total = term1 + term2
            
            # Mask assigned locations
            mask = np.ones(n, dtype=bool)
            mask[assigned_locs] = False
            
            # Calculate only for unassigned? 
            # Calculating for all is fine, then mask.
            
            a_idx_j = a[idx, assigned_facs]
            a_j_idx = a[assigned_facs, idx]
            p_j = p[assigned_facs]
            
            # Efficient dot products
            # term1 = b[:, p_j] @ a_idx_j
            term1 = np.dot(b[:, p_j], a_idx_j)
            
            # term2 = b[p_j, :] @ a_j_idx
            term2 = np.dot(b[p_j, :], a_j_idx)
            
            total_cost = term1 + term2
            total_cost[assigned_locs] = np.inf
            
            best_loc = np.argmin(total_cost)
            
        p[idx] = best_loc
        assigned_locs.append(best_loc)
        assigned_facs.append(idx)
        
    # Calculate initial cost
    b_p = b[np.ix_(p, p)]
    current_cost = np.sum(a * b_p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing
    T = 500.0
    cooling = 0.9995
    min_T = 1e-6
    
    start_time = time.time()
    time_limit = 110.0
    
    rng = np.random.default_rng(42)
    
    stuck_counter = 0
    
    while time.time() - start_time < time_limit:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        diff_S = S[i, :] - S[j, :]
        diff_b = b[v, :] - b[u, :]
        
        delta = np.dot(diff_S, diff_b)
        delta -= diff_S[i] * diff_b[i]
        delta -= diff_S[j] * diff_b[j]
        
        new_cost = current_cost + delta
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0:
                # Check for overflow
                if delta / T > 700: # exp(-700) is tiny
                    prob = 0.0
                else:
                    prob = np.exp(-delta / T)
                accept = rng.random() < prob
        
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            stuck_counter += 1

        T *= cooling
        if T < min_T:
            T = min_T
            
        if stuck_counter > 5000:
            # Perturb
            for _ in range(10):
                ri = rng.integers(0, n)
                rj = rng.integers(0, n)
                if ri != rj:
                    p[ri], p[rj] = p[rj], p[ri]
            b_p = b[np.ix_(p, p)]
            current_cost = np.sum(a * b_p)
            T = 100.0
            stuck_counter = 0

    return best_p.tolist()
