"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    N = 256
    
    # 1. Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    # 2. Precompute compatibility matrix (coprime check)
    # compat[i][j] is True if gcd(i, j) == 1
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if not (prime_factors[i] & prime_factors[j]):
                compat[i][j] = True
                compat[j][i] = True
    
    # 3. Sort vertices by weight (value) descending
    # We index 1..N. 
    # We will process them in this order for B&B.
    vertices = list(range(N, 0, -1))
    
    best_sum = 0
    best_set = []
    
    # Greedy Initialization for a strong initial lower bound
    current_greedy = []
    current_greedy_sum = 0
    # Try to add largest numbers first if they fit
    for v in vertices:
        can_add = True
        for u in current_greedy:
            if not compat[u][v]:
                can_add = False
                break
        if can_add:
            current_greedy.append(v)
            current_greedy_sum += v
    
    best_sum = current_greedy_sum
    best_set = list(current_greedy)
    
    # Branch and Bound
    # State: 
    #   idx: index in the sorted `vertices` list
    #   current_set: list of selected vertices
    #   current_sum: sum of selected vertices
    #   candidates: list of vertices from idx..end that are compatible with current_set
    
    # To speed up, we maintain a list of available candidates.
    # Since N=256, we can use a bitmask or a list.
    # A list of candidates is simpler.
    
    def bnb(idx, current_set, current_sum, candidates):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning: If current_sum + sum of all weights in candidates <= best_sum, prune
        # Note: candidates are already filtered to be compatible with current_set.
        # However, candidates might conflict with each other.
        # So sum(candidates) is a valid upper bound.
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            return
            
        upper_bound = current_sum + sum(candidates)
        if upper_bound <= best_sum:
            return
            
        # Heuristic: Pick the vertex with highest weight from candidates
        # In our pre-sorted `vertices`, the candidates are not necessarily sorted by weight 
        # if we just filter them. But `vertices` is sorted desc.
        # So `candidates` contains a subset of `vertices[idx:]`.
        # The first element in `candidates` is the largest available weight.
        
        v = candidates[0]
        
        # Option 1: Include v
        # New candidates: those in candidates[1:] that are compatible with v
        new_candidates = []
        for u in candidates[1:]:
            if compat[v][u]:
                new_candidates.append(u)
                
        # Recurse with v included
        current_set.append(v)
        bnb(idx + 1, current_set, current_sum + v, new_candidates)
        current_set.pop()
        
        # Option 2: Exclude v
        # New candidates: candidates[1:]
        # We can also prune if current_sum + sum(candidates[1:]) <= best_sum
        if time.time() > deadline:
            return
        bnb(idx + 1, current_set, current_sum, candidates[1:])
        
    # Start B&B
    # Initial candidates are all vertices
    # We pass the full sorted list.
    bnb(0, [], 0, vertices)
    
    # Final check: Ensure distinct integers and types
    final_set = list(set(best_set))
    final_set = [int(x) for x in final_set]
    
    return final_set

# EVOLVE-BLOCK-END
