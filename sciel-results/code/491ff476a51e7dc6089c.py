import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # 1. Generate all bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(b,c) == gcd(c,a)
    # We assume a < b < c to avoid duplicates
    bad_triples = []
    triples_for_elem = [[] for _ in range(129)] # 1-indexed
    
    for a in range(1, 129):
        for b in range(a + 1, 129):
            g_ab = gcd(a, b)
            for c in range(b + 1, 129):
                g_bc = gcd(b, c)
                if g_ab == g_bc:
                    g_ca = gcd(c, a)
                    if g_ab == g_ca:
                        bad_triples.append((a, b, c))
                        triples_for_elem[a].append(len(bad_triples) - 1)
                        triples_for_elem[b].append(len(bad_triples) - 1)
                        triples_for_elem[c].append(len(bad_triples) - 1)
                        
    if not bad_triples:
        return list(range(1, 129))

    # 2. Greedy Hitting Set
    # We want to remove the minimum number of elements to hit all bad triples.
    # This maximizes the remaining set.
    
    active_triples = [True] * len(bad_triples)
    deg = [0] * 129
    for t_idx in range(len(bad_triples)):
        a, b, c = bad_triples[t_idx]
        deg[a] += 1
        deg[b] += 1
        deg[c] += 1
        
    removed = [False] * 129
    active_count = len(bad_triples)
    
    # To handle ties and potential local optima, we can randomize the choice among max-degree elements
    # or use a simple greedy. Let's try a few random restarts with slight randomization in tie-breaking.
    
    best_size = 0
    best_set = set()
    
    # Run multiple iterations with random tie-breaking
    num_iters = 50
    for _ in range(num_iters):
        if time.time() > deadline:
            break
            
        current_active = [True] * len(bad_triples)
        current_deg = [0] * 129
        for t_idx in range(len(bad_triples)):
            a, b, c = bad_triples[t_idx]
            current_deg[a] += 1
            current_deg[b] += 1
            current_deg[c] += 1
            
        current_removed = [False] * 129
        active_count = len(bad_triples)
        
        while active_count > 0:
            # Find an active triple
            # Instead of scanning all, we can just pick a random active triple?
            # Or pick the one with highest sum of degrees?
            # Greedy: Pick an active triple, then pick the element in it with highest degree.
            
            # Optimization: Just iterate through active triples to find one, or keep a list of active indices.
            # For simplicity and speed, we scan. 128^3 is small.
            found_triple = -1
            for t_idx in range(len(bad_triples)):
                if current_active[t_idx]:
                    found_triple = t_idx
                    break
            
            if found_triple == -1:
                break
                
            a, b, c = bad_triples[found_triple]
            
            # Choose element to remove: max degree, random tie-break
            max_deg = -1
            candidates = []
            for x in (a, b, c):
                if not current_removed[x]:
                    if current_deg[x] > max_deg:
                        max_deg = current_deg[x]
                        candidates = [x]
                    elif current_deg[x] == max_deg:
                        candidates.append(x)
            
            # Pick randomly from candidates
            x = random.choice(candidates)
            current_removed[x] = True
            
            # Update degrees and active status for all triples containing x
            for t_idx in triples_for_elem[x]:
                if current_active[t_idx]:
                    current_active[t_idx] = False
                    active_count -= 1
                    u, v, w = bad_triples[t_idx]
                    for y in (u, v, w):
                        if y != x:
                            current_deg[y] -= 1
        
        # Calculate size
        size = 128 - sum(1 for r in current_removed if r)
        if size > best_size:
            best_size = size
            best_set = set(range(1, 129))
            for i in range(1, 129):
                if current_removed[i]:
                    best_set.remove(i)

    return list(best_set)
