import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Identify all bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return numbers # Fallback
        
    # 2. Precompute conflict structures
    # conflicts[x] = list of pairs (u, v) such that (x, u, v) is a bad triple
    # This allows fast checking: can we add x to set S?
    # Yes, if no pair (u, v) in conflicts[x] is fully contained in S.
    
    conflicts = {x: [] for x in numbers}
    for a, b, c in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
        
    # Helper function to check if x can be added to current_set
    def can_add(x, current_set):
        for u, v in conflicts[x]:
            if u in current_set and v in current_set:
                return False
        return True

    best_set = set()
    best_size = -1
    
    num_restarts = 100 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Strategy: Greedy Vertex Cover with Randomized Tie-Breaking
        # We start with all numbers and remove elements that are in the most active bad triples.
        
        S = set(numbers)
        counts = {x: 0 for x in numbers}
        for t in bad_triples:
            for x in t:
                counts[x] += 1
        
        active_triples = [True] * len(bad_triples)
        
        while True:
            max_count = 0
            candidates = []
            
            for x in S:
                if counts[x] > max_count:
                    max_count = counts[x]
                    candidates = [x]
                elif counts[x] == max_count and max_count > 0:
                    candidates.append(x)
            
            if not candidates or max_count == 0:
                break
                
            # Randomly pick from candidates to break ties
            target = random.choice(candidates)
            
            S.remove(target)
            
            for i, t in enumerate(bad_triples):
                if not active_triples[i]:
                    continue
                if target in t:
                    active_triples[i] = False
                    for x in t:
                        if x != target:
                            counts[x] -= 1
        
        # Local Search: Try to increase size by removing 1 and adding many
        improved = True
        max_local_iters = 50 # Cap local search iterations per restart
        
        local_iter = 0
        while improved and local_iter < max_local_iters:
            improved = False
            local_iter += 1
            if time.time() > deadline - 5.0:
                break
                
            # Try removing each y in S
            # Randomize order to escape local optima
            s_list = list(S)
            random.shuffle(s_list)
            
            for y in s_list:
                if time.time() > deadline - 5.0:
                    break
                    
                S_tentative = S - {y}
                
                # Try to add any element not in S
                # We should try to add as many as possible
                # Order of addition matters. Let's try random order.
                
                candidates_to_add = [x for x in numbers if x not in S]
                random.shuffle(candidates_to_add)
                
                added_count = 0
                for z in candidates_to_add:
                    if can_add(z, S_tentative):
                        S_tentative.add(z)
                        added_count += 1
                        
                # If we added more elements than we removed (1), size increased
                if len(S_tentative) > len(S):
                    S = S_tentative
                    improved = True
                    break # Restart the while loop with the new S
                    
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # Early exit if we found a very large set
        if best_size > 120:
             pass

    if best_size == -1:
        return [1]
        
    return list(best_set)
