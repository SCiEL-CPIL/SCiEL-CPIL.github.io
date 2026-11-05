import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Identify bad triples and build conflict_pairs
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
                    
    best_set = set()
    best_size = 0
    
    # Run many restarts
    # We can also try different strategies:
    # 1. Random tie-break in greedy removal.
    # 2. Random order in add-back.
    # 3. Maybe try "Min Degree" removal? No, Max Degree removal is standard for MIS.
    
    for seed in range(100000): # High limit, will break on time
        if time.time() > deadline:
            break
            
        random.seed(seed)
        S = set(range(1, N + 1))
        
        # Initial degrees
        degrees = {v: len(conflict_pairs[v]) for v in S}
        
        # Greedy Removal
        while True:
            if not S:
                break
            
            max_deg = max(degrees.values())
            if max_deg == 0:
                break
            
            # Candidates
            candidates = [v for v in S if degrees[v] == max_deg]
            v_to_remove = random.choice(candidates)
            
            S.remove(v_to_remove)
            
            # Update degrees
            for a, b in conflict_pairs[v_to_remove]:
                if a in S:
                    degrees[a] -= 1
                if b in S:
                    degrees[b] -= 1
            
        # Add-back Phase
        R = set(range(1, N + 1)) - S
        R_list = list(R)
        random.shuffle(R_list)
        
        for v in R_list:
            if time.time() > deadline:
                break
                
            can_add = True
            for a, b in conflict_pairs[v]:
                if a in S and b in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(v)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
            # If we found a very large set, we can try to be more patient or just continue.
            # The theoretical max is likely around 100-120? 
            # Pairwise coprimality max is much smaller.
            # Equal gcd triples are rarer than non-coprime triples.
            # So the set should be large.
            
    if not best_set:
        return [1]
        
    return list(best_set)

# EVOLVE-BLOCK-END
