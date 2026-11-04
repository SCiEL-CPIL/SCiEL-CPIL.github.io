import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initial Solution: Rank Matching
    # Heavy flows (high sum of a) should be assigned to central locations (low sum of b)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    def calc_cost(p):
        # C = sum a[i,j] * b[p[i], p[j]]
        # This is equivalent to sum over i of a[i, :] dot b[p[i], p]
        # Using einsum or simple loops might be slow. 
        # Vectorized: 
        # b_perm = b[np.ix_(p, p)]
        # return np.sum(a * b_perm)
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    # Simulated Annealing Parameters
    T = 1000.0 # Initial temperature. Cost scale for wil100 is ~270,000.
    cooling = 0.999 # Cooling rate
    min_T = 0.1
    
    # Helper to calculate delta cost for swapping indices i and j in permutation p
    # i, j are facility indices. p[i] and p[j] are their current locations.
    def delta_cost(i, j, p):
        li = p[i]
        lj = p[j]
        # Sum over all k != i, j
        # Terms:
        # Old: a[i,k]b[li, p[k]] + a[k,i]b[p[k], li] + a[j,k]b[lj, p[k]] + a[k,j]b[p[k], lj]
        # New: a[i,k]b[lj, p[k]] + a[k,i]b[p[k], lj] + a[j,k]b[li, p[k]] + a[k,j]b[p[k], li]
        # Delta = New - Old
        
        # We can compute this vectorized for all k at once to speed up, 
        # but n=100 is small, so a loop or simple vector ops on slices is fine.
        # Let's use vectorized ops on the whole arrays and subtract i,j contributions if needed, 
        # or just loop k. n=100 loop is ~100 ops. 100k steps/s is possible.
        
        # Vectorized approach:
        # d = (a[i, :] * (b[lj, p] - b[li, p]) + (a[:, i]) * (b[p, lj] - b[p, li]) + 
        #       a[j, :] * (b[li, p] - b[lj, p]) + (a[:, j]) * (b[p, li] - b[p, lj]))
        # This sums over all k. We need to exclude k=i and k=j?
        # The formula above includes k=i and k=j.
        # Let's check the terms for k=i:
        # a[i,i]... usually 0 in QAP. If not, need care. QAPLIB usually has 0 diagonal.
        # Assuming a[i,i]=0 and b[i,i]=0.
        # If we sum over all k, the terms for k=i and k=j in the "Old" and "New" 
        # parts related to the swap of i and j themselves need specific handling.
        
        # Let's just do the O(n) loop for clarity and correctness, it's fast enough for n=100.
        delta = 0.0
        # Interactions with i and j themselves
        # Old: a[i,j]b[li, lj] + a[j,i]b[lj, li]
        # New: a[i,j]b[lj, li] + a[j,i]b[li, lj]
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        for k in range(n):
            if k == i or k == j:
                continue
            lk = p[k]
            # Old contributions involving i or k, or j or k
            # a[i,k] b[li, lk] + a[k,i] b[lk, li]
            # a[j,k] b[lj, lk] + a[k,j] b[lk, lj]
            old = a[i, k] * b[li, lk] + a[k, i] * b[lk, li] + a[j, k] * b[lj, lk] + a[k, j] * b[lk, lj]
            
            # New contributions (i takes lj, j takes li)
            # a[i,k] b[lj, lk] + a[k,i] b[lk, lj]
            # a[j,k] b[li, lk] + a[k,j] b[lk, li]
            new = a[i, k] * b[lj, lk] + a[k, i] * b[lk, lj] + a[j, k] * b[li, lk] + a[k, j] * b[lk, li]
            
            delta += new - old
            
        return delta

    # Main SA Loop
    while time.time() - start_time < time_limit:
        # Pick random pair
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        dc = delta_cost(i, j, p)
        
        # Acceptance criterion
        if dc < 0:
            # Accept improvement
            p[i], p[j] = p[j], p[i]
            current_cost += dc
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Accept worsening with probability exp(-dc/T)
            if T > 0:
                prob = np.exp(-dc / T)
                if random.random() < prob:
                    p[i], p[j] = p[j], p[i]
                    current_cost += dc
        
        # Cool down
        T *= cooling
        if T < min_T:
            # Optional: Reheat or stop? Let's just stop cooling and keep swapping if possible?
            # Better to just break if T is very low, but we have time.
            # Let's reset T to a small value to allow some exploration? 
            # Or just continue with T=min_T.
            pass

    return best_p.tolist()
