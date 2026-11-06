import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 512
    # Build adjacency list as bitmasks using vectorized approach
    adj = [0] * n
    
    # Precompute lcm matrix more efficiently
    # For each pair (a,b), check if lcm(a,b) <= 512
    # lcm(a,b) = a*b/gcd(a,b) <= 512
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            g = gcd(a, b)
            if a * b // g <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute neighbor counts for faster pivot selection
    neighbor_counts = [adj[i].bit_count() for i in range(n)]
    
    # Find a good initial clique by trying multiple highly composite numbers
    # 360 = 2^3 * 3^2 * 5 (24 divisors)
    # 2520 = 2^3 * 3^2 * 5 * 7 (48 divisors) - too large
    # Let's try 360, 240, 180, 120
    candidates_for_init = [360, 240, 180, 120, 72, 60]
    best_initial = []
    for L in candidates_for_init:
        divs = [i for i in range(1, min(L, 512) + 1) if L % i == 0]
        if len(divs) > len(best_initial):
            best_initial = divs
    
    best_clique = best_initial
    best_size = len(best_clique)
    
    all_nodes = (1 << n) - 1
    
    # Use a counter to avoid excessive time checking
    time_check_counter = 0
    
    def bronk(R, P, X):
        nonlocal best_clique, best_size, time_check_counter
        
        # Time check every 500 calls (more frequent)
        time_check_counter += 1
        if time_check_counter % 500 == 0:
            if time.time() > deadline:
                return
        
        # Bound check: Can we beat the best?
        if R.bit_count() + P.bit_count() <= best_size:
            return
        
        if not P:
            if not X:
                # Found a maximal clique
                rc = R.bit_count()
                if rc > best_size:
                    best_clique = [i + 1 for i in range(n) if (R >> i) & 1]
                    best_size = rc
            return
        
        # Pivot selection: choose u in P | X to maximize |P & adj[u]|
        union = P | X
        if union == 0:
            return
        
        # Fast pivot: try to find a node that covers all of P
        pivot = -1
        p_count = P.bit_count()
        
        # Quick check: if any node in P has all other P nodes as neighbors
        temp_p = P
        while temp_p and pivot == -1:
            lsb = temp_p & (-temp_p)
            u = lsb.bit_length() - 1
            temp_p ^= lsb
            if (P & adj[u]).bit_count() >= p_count - 1:
                pivot = u
                break
        
        # If not found, sample from X or use first node
        if pivot == -1:
            # Try a few nodes from union, prioritizing high-degree nodes
            temp = union
            best_count = -1
            # Sample up to 15 candidates
            for _ in range(min(15, temp.bit_count())):
                if not temp:
                    break
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                temp ^= lsb
                count = (P & adj[u]).bit_count()
                if count > best_count:
                    best_count = count
                    pivot = u
                if count >= p_count:
                    break
            if pivot == -1:
                # Fallback: just take first node
                lsb = union & (-union)
                pivot = lsb.bit_length() - 1
        
        # Iterate over candidates P \ N(pivot)
        candidates = P & ~adj[pivot]
        
        # Order candidates by degree (greedy heuristic) - process high-degree first
        cand_list = []
        temp = candidates
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            cand_list.append((neighbor_counts[v], v))
        
        # Sort by degree descending
        cand_list.sort(reverse=True)
        
        for _, v in cand_list:
            if time.time() > deadline:
                return
            
            lsb = 1 << v
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bronk(R | lsb, P_new, X_new)
            
            P ^= lsb
            X |= lsb
    
    bronk(0, all_nodes, 0)
    
    if not best_clique:
        return [1]
        
    return best_clique
