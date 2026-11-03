"""QAP Solver using Iterated Linearization with Multi-Swap (3-opt/4-opt) Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """
    Compute delta for swapping positions i and j in permutation p.
    a_sym[i, k] = a[i, k] + a[k, i]
    """
    x = p[i]
    y = p[j]
    
    # diff_a[k] = a_sym[i, k] - a_sym[j, k]
    diff_a = a_sym[i, :] - a_sym[j, :]
    
    # diff_b[k] = b[y, k] - b[x, k]
    diff_b = b[y, :] - b[x, :]
    
    # Calculate sum over all k: sum_k diff_a[k] * diff_b[p[k]]
    # We need to exclude k=i and k=j from the sum
    temp = diff_a * diff_b[p]
    
    # Subtract terms for k=i and k=j
    # For k=i: diff_a[i] * diff_b[p[i]] = diff_a[i] * diff_b[x]
    # For k=j: diff_a[j] * diff_b[p[j]] = diff_a[j] * diff_b[y]
    
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _cyclic_shift_delta(a_sym, b, p, i, j, k):
    """
    Compute delta for cyclic shift of positions i, j, k.
    p[i] -> p[j], p[j] -> p[k], p[k] -> p[i]
    This is equivalent to swapping i,j then swapping i,k in the new config?
    Actually, let's derive directly or simulate.
    Simulating is O(n) for cost calculation if we don't have O(1) delta for 3-swaps.
    However, we can compute the cost change by:
    1. Save old cost contribution of i, j, k with all others.
    2. Apply shift.
    3. Compute new cost contribution.
    4. Difference.
    
    To keep it fast, we can use the fact that only interactions involving i, j, k change.
    Let S = {i, j, k}.
    Old Cost = sum_{u,v in S} a[u,v]b[p[u],p[v]] + sum_{u in S, v not in S} (a[u,v]b[p[u],p[v]] + a[v,u]b[p[v],p[u]])
    New Cost = same with p' where p'[i]=p[k], p'[j]=p[i], p'[k]=p[j] (assuming shift i<-k<-j<-i? 
    Let's define the move clearly:
    Cycle: i -> j -> k -> i
    New p:
    p_new[i] = p[k]
    p_new[j] = p[i]
    p_new[k] = p[j]
    Others unchanged.
    
    Cost Change = New - Old.
    Terms involving only S:
    Old_S = a[i,j]b[p[i],p[j]] + a[j,i]b[p[j],p[i]] + a[j,k]b[p[j],p[k]] + a[k,j]b[p[k],p[j]] + a[k,i]b[p[k],p[i]] + a[i,k]b[p[i],p[k]]
    Since a,b symmetric: 2 * (a[i,j]b[p[i],p[j]] + a[j,k]b[p[j],p[k]] + a[k,i]b[p[k],p[i]])
    
    New_S = 2 * (a[i,j]b[p[k],p[i]] + a[j,k]b[p[i],p[j]] + a[k,i]b[p[j],p[k]])
    
    Terms involving S and Outside (O):
    For each u in S, v in O:
    Old: a[u,v]b[p[u],p[v]] + a[v,u]b[p[v],p[u]] = 2*a[u,v]*b[p[u],p[v]]
    New: 2*a[u,v]*b[p_new[u],p[v]]
    
    Delta = New_S - Old_S + sum_{u in S, v in O} 2*a[u,v]*(b[p_new[u],p[v]] - b[p[u],p[v]])
    
    This is O(n) to compute the sum over O.
    """
    # This might be too slow in Python loop for 3-opt if not vectorized well.
    # Let's stick to 2-opt for speed but add a "kick" using random 3-swaps if stuck?
    # Or just use the Hungarian step as the "kick".
    pass

def _hungarian_step(a, b, p):
    """
    Perform a global reassignment using the Hungarian algorithm.
    """
    n = len(a)
    M = b[:, p]  # M[x, j] = b[x, p[j]]
    C = a @ M.T  # C[i, x] = sum_j a[i, j] b[x, p[j]]
    
    row_ind, col_ind = linear_sum_assignment(C)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p

def _local_search_2opt(a, b, a_sym, p, time_limit, start_time, max_iter=50):
    """
    Perform 2-opt local search. Optimized for speed.
    """
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        if time.time() - start_time > time_limit:
            break
            
        # Vectorized-ish approach for 2-opt pass?
        # Hard to vectorize the "if delta < 0 then swap" loop easily without GPU.
        # We will iterate pairs.
        for i in range(n):
            if time.time() - start_time > time_limit:
                return p, current_cost
            for j in range(i + 1, n):
                delta = _swap_delta(a_sym, b, p, i, j)
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # First improvement restart is often faster for QAP?
                    # Let's do full pass for now to ensure convergence.
                    
    return p, current_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0 # Safety margin for 150s limit
    
    # Precompute symmetric a for fast delta
    a_sym = a + a.T
    
    # Initial solution: Rank matching (greedy)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Main Loop: Iterated Linearization
    # We alternate between:
    # 1. Hungarian Global Reassignment (Large Jump)
    # 2. 2-opt Local Search (Polishing)
    # 3. If stuck (no improvement in Hungarian), apply random perturbations (Kicks)
    
    stuck_count = 0
    kick_strength = 5 # Number of random swaps for kick
    
    while time.time() - start_time < time_limit:
        # 1. Hungarian Step
        new_p = _hungarian_step(a, b, p)
        new_cost = _qap_cost(a, b, new_p)
        
        # Accept if better or equal (to allow diversification)
        if new_cost <= current_cost:
            p = new_p.copy()
            current_cost = new_cost
            stuck_count = 0
        else:
            stuck_count += 1
            
            # If stuck for a few Hungarian steps, apply a kick to p
            if stuck_count > 3:
                # Random kick: swap 'kick_strength' pairs
                for _ in range(kick_strength):
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                # Recalculate cost after kick
                current_cost = _qap_cost(a, b, p)
                stuck_count = 0
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # 2. 2-opt Polishing
        # Limit time for polishing to ensure we have time for more Hungarian steps
        # We give 2-opt a budget of up to 10s, but it exits early if no improvement.
        polish_limit = min(time_limit, time.time() + 10.0)
        p, current_cost = _local_search_2opt(a, b, a_sym, p, polish_limit, start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Check time
        if time.time() - start_time > time_limit - 5.0:
            break
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
