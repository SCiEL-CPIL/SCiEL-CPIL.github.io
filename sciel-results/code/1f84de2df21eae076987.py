import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    # Precompute forbidden triples
    forbidden_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            l_ab = lcm(a, b)
            if l_ab > N:
                continue
            for c in range(b + 1, N + 1):
                if time.time() > deadline - 15.0:
                    break
                if lcm(a, c) == l_ab and lcm(b, c) == l_ab:
                    forbidden_triples.add((a, b, c))
                    
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in forbidden_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    best_set = set()
    
    # Run multiple greedy attempts
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Different starting sets
        if attempt == 0:
            S = set(range(65, N + 1))
        elif attempt == 1:
            S = set(range(1, N + 1, 2)) # Odds
        elif attempt == 2:
            S = set()
        else:
            # Random subset
            S = set(random.sample(range(1, N + 1), random.randint(40, 60)))
            
        # Greedy Addition
        # Sort candidates by "popularity" (number of conflicts) to add safe ones first?
        # Or just iterate 1..N
        for x in range(1, N + 1):
            if x in S:
                continue
            can_add = True
            for (a, b) in conflict_pairs[x]:
                if a in S and b in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                
        if len(S) > len(best_set):
            best_set = S.copy()
            
    # Local search on best_set
    S = best_set
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for x in range(1, N + 1):
            if x in S:
                continue
            can_add = True
            for (a, b) in conflict_pairs[x]:
                if a in S and b in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                improved = True
                break
        if improved:
            continue
        # Try simple swaps: remove y, add x
        # This is O(N^2) per check, too slow if done exhaustively.
        # Skip complex local search if time is tight.
        
    return list(S)
