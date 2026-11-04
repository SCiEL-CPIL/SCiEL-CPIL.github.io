import numpy as np
import time
import random
import sys

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_swap_deltas(a, b, p):
    """
    Compute cost deltas for all 2-opt swaps (i, j) in permutation p.
    Returns a matrix D where D[i, j] is the change in cost if we swap 
    positions i and j in p.
    """
    n = len(p)
    C = b[np.ix_(p, p)]
    S1 = a.T @ C
    T1 = a @ C
    
    D_S1 = np.diag(S1)
    D_T1 = np.diag(T1)
    
    # Term 1: Sum of flows from i to others * distances
    # Term 2: Sum of flows to i from others * distances
    # The formula derived in the current program is efficient.
    Term1 = S1 - D_S1[:, None] - D_S1[None, :] + S1.T
    Term2 = T1 - D_T1[:, None] - D_T1[None, :] + T1.T
    
    Raw = Term1 + Term2
    
    # Error term correction
    Error = -2.0 * (a + a.T) * C
    
    Delta = Raw - Error
    np.fill_diagonal(Delta, 0.0)
    
    return Delta

def _compute_oropt_deltas(a, b, p, k=3):
    """
    Compute cost deltas for moving a block of size k from position i to position j.
    This is more complex than swaps. For simplicity and speed in this implementation,
    we will primarily rely on 2-opt swaps and random re-insertions for perturbation,
    but we can implement a fast check for 3-opt moves if needed.
    However, calculating all O(n^2) or-opt deltas is O(n^3) or high constant factor.
    Instead, we will use a hybrid local search:
    1. Fast 2-opt steepest descent.
    2. Random Or-opt (move a block of size 1, 2, or 3) to escape plateaus.
    
    To keep it fast, we will implement a specific Or-opt move evaluation on demand
    rather than pre-computing all deltas, or just use 2-opt and rely on SA perturbation.
    Given the 150s limit, 2-opt is the workhorse. We will add a 'kick' using random 
    re-insertion of a block.
    """
    pass # Placeholder, we will implement the logic inside the search loop

def _local_search_2opt(a, b, p, max_iters=200, time_limit=None, start_time=None):
    """
    Steepest descent 2-opt local search.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        deltas = _compute_swap_deltas(a, b, p)
        
        # Find the best improving move
        min_delta = np.min(deltas)
        
        if min_delta >= 0:
            break # Local minimum reached
            
        # Find indices of the best move
        i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
    return p, current_cost

def _perturb(p, strength=5):
    """
    Perturb the permutation by moving a block of size 1, 2, or 3 
    to a random location. This is a standard ILS kick.
    """
    n = len(p)
    p = p.copy()
    
    # Choose block size
    k = random.choice([1, 2, 3])
    
    # Choose start position of the block
    if k > n:
        k = n
    start = random.randint(0, n - k)
    
    # Extract block
    block = p[start:start+k].copy()
    
    # Remove block from original position
    # The remaining elements shift left
    p[start:n-k] = p[start+k:n]
    
    # Choose insertion position
    # The array is now size n-k. We can insert at 0 to n-k
    # If we insert at the same spot, it's a no-op (mostly)
    ins_pos = random.randint(0, n - k)
    
    # Insert block
    if ins_pos > start:
        # Inserting after the original position
        # The "gap" is effectively at 'start' but the array shifted.
        # We need to be careful with indices.
        # Easier way:
        # 1. Remove block
        # 2. Create new array
        # Let's do it cleanly:
        p_new = np.concatenate([p[:ins_pos], block, p[ins_pos:]])
        # Wait, p was modified in place. Let's restart perturb logic cleanly.
        
    # Clean implementation:
    p = p.copy()
    k = random.choice([1, 2, 3])
    if k > n:
        k = n
    start = random.randint(0, n - k)
    block = p[start:start+k].copy()
    
    # Remove block
    p_reduced = np.concatenate([p[:start], p[start+k:]])
    
    # Insert at random position in p_reduced (length n-k)
    # Possible positions: 0 to n-k
    ins_pos = random.randint(0, n - k)
    
    p_final = np.concatenate([p_reduced[:ins_pos], block, p_reduced[ins_pos:]])
    
    return p_final

def _sa_polish(a, b, p, start_temp=1000, end_temp=1, cool=0.99, max_iters=500, time_limit=None, start_time=None):
    """
    Simulated Annealing polish to escape local minima.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    n = len(p)
    current_p = p.copy()
    current_cost = _qap_cost(a, b, current_p)
    best_p = current_p.copy()
    best_cost = current_cost
    
    T = start_temp
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if T <= end_temp:
            break
            
        # Random swap
        i, j = np.random.choice(n, 2, replace=False)
        if i == j:
            continue
            
        # Compute delta for this specific swap
        # To save time, we don't recompute all deltas, just this one.
        # But computing delta for one swap is O(n).
        # cost_new = cost_old + delta
        # delta = sum_k a[i,k](b[p[k],p[j]] - b[p[k],p[i]]) + sum_k a[j,k](b[p[k],p[i]] - b[p[k],p[j]])
        #       + sum_k a[k,i](b[p[i],p[k]] - b[p[j],p[k]]) + sum_k a[k,j](b[p[i],p[k]] - b[p[j],p[k]])
        #       - 2*a[i,j]*b[p[i],p[j]] - 2*a[j,i]*b[p[j],p[i]] + 2*a[i,j]*b[p[j],p[i]] + 2*a[j,i]*b[p[i],p[j]]
        # Actually, it's easier to just calculate the new cost if n is small, but n=100.
        # O(n) calculation is fast.
        
        pi = current_p[i]
        pj = current_p[j]
        
        # We can use the precomputed delta matrix if we had it, but we don't in SA loop.
        # Let's just do a fast O(n) delta calculation.
        # Let C = b[np.ix_(current_p, current_p)]
        # We need delta for swapping positions i and j in current_p.
        
        # To avoid O(n^2) per step, we stick to 2-opt local search for the "polish"
        # and use SA only for the "perturbation" phase? No, SA is for escaping.
        # Let's use a simple random swap delta.
        
        # Actually, for n=100, O(n) is very fast.
        # Let's precompute C once? No, p changes.
        
        # Let's just do the O(n) calculation.
        c_i = b[np.ix_(current_p[i], current_p)] # b[p[i], p[*]]
        c_j = b[np.ix_(current_p[j], current_p)] # b[p[j], p[*]]
        
        # Flows involving i or j
        # The change in cost is:
        # sum_{k != i,j} a[i,k] * (b[p[k], p[j]] - b[p[k], p[i]]) 
        # + sum_{k != i,j} a[j,k] * (b[p[k], p[i]] - b[p[k], p[j]])
        # + sum_{k != i,j} a[k,i] * (b[p[i], p[k]] - b[p[j], p[k]])
        # + sum_{k != i,j} a[k,j] * (b[p[i], p[k]] - b[p[j], p[k]])
        # + (a[i,j] * b[p[j], p[i]] + a[j,i] * b[p[i], p[j]]) - (a[i,j] * b[p[i], p[j]] + a[j,i] * b[p[j], p[i]])
        
        # Since b is symmetric, b[p[i], p[j]] == b[p[j], p[i]].
        # Last term: a[i,j]*b + a[j,i]*b - a[i,j]*b - a[j,i]*b = 0.
        # So the diagonal swap term cancels out if b is symmetric.
        # Wait, the term is a[i,j] * b[new_i, new_j] where new_i=p[j], new_j=p[i].
        # Old: a[i,j] * b[p[i], p[j]].
        # New: a[i,j] * b[p[j], p[i]].
        # If b is symmetric, these are equal. So no change from the direct i-j interaction.
        
        # Let's compute the sums.
        # We need b[p[k], p[i]] and b[p[k], p[j]] for all k.
        # Let col_i = b[:, p[i]]
        # Let col_j = b[:, p[j]]
        # We need sum_{k != i,j} a[i,k] * (col_i[current_p[k]] - col_j[current_p[k]])
        
        # This is getting complex to vectorize quickly for a single swap in a tight loop.
        # Alternative: Use the _compute_swap_deltas function but only use it every few steps?
        # Or, just accept the O(n^2) cost of recomputing all deltas every few steps?
        
        # For this implementation, to ensure speed, we will use a "greedy" SA:
        # 1. Compute all deltas (O(n^2))
        # 2. Pick a random move, not just the best.
        # 3. Accept if delta < 0 or with probability exp(-delta/T).
        
        deltas = _compute_swap_deltas(a, b, current_p)
        
        # Pick a random pair
        i, j = np.random.choice(n, 2, replace=False)
        delta = deltas[i, j]
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                
        T *= cool
        
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 145.0 # 5s buffer for safety
    start_time = time.time()
    
    # ILS Parameters
    num_restarts = 20 # We will loop until time runs out
    current_best_p = None
    current_best_cost = float('inf')
    
    # Initial solution: Greedy
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Initial Local Search
    p, c = _local_search_2opt(a, b, p, max_iters=100, time_limit=total_deadline, start_time=start_time)
    
    if c < current_best_cost:
        current_best_cost = c
        current_best_p = p.copy()
    
    # ILS Loop
    while time.time() - start_time < total_deadline - 5.0:
        # 1. Perturb
        # Strength of perturbation can increase if we are stuck
        # For simplicity, fixed strength
        p_perturbed = _perturb(p, strength=3)
        
        # 2. Local Search (Polish)
        # We alternate between 2-opt and SA polish?
        # Let's do 2-opt first, then a short SA if time permits.
        
        p_local, c_local = _local_search_2opt(a, b, p_perturbed, max_iters=100, time_limit=total_deadline, start_time=start_time)
        
        # If we have time, do a quick SA polish on the result
        if time.time() - start_time < total_deadline - 10.0:
            p_sa, c_sa = _sa_polish(a, b, p_local, start_temp=500, end_temp=1, cool=0.95, max_iters=100, time_limit=total_deadline, start_time=start_time)
            if c_sa < c_local:
                p_local = p_sa
                c_local = c_sa
        
        # 3. Acceptance Criterion
        # Standard ILS: Accept if better, or with some probability to escape.
        # For QAP, simple "accept if better" (First Improvement or Best Improvement) is often robust.
        # Let's use: Accept if c_local < current_best_cost.
        # To maintain diversity, we can also accept with probability if it's close.
        # But to ensure we converge, let's stick to best-so-far for the global best,
        # and update the current working solution 'p' based on ILS rules.
        
        if c_local < current_best_cost:
            current_best_cost = c_local
            current_best_p = p_local.copy()
            p = p_local.copy() # Continue from the new best
        else:
            # Accept with probability to explore
            # Simple rule: Accept if delta is small, or random
            # For now, let's just move to the local solution to explore around it
            # unless it's much worse.
            if c_local < current_best_cost * 1.1: # Within 10%
                p = p_local.copy()
            else:
                # Revert to best
                p = current_best_p.copy()
                
    return current_best_p.tolist()

# EVOLVE-BLOCK-END
