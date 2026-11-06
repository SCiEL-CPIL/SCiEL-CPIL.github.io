"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    N = 128
    
    # 1. Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Identify all bad triples in [1, N]
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples_by_element = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    bad_triples_by_element[a].append((b, c))
                    bad_triples_by_element[b].append((a, c))
                    bad_triples_by_element[c].append((a, b))

    def is_valid_insertion(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for y, z in bad_triples_by_element[x]:
            if y in S and z in S:
                return False
        return True

    def greedy_construct(ordering):
        """Build a set by iterating through a specific ordering."""
        S = set()
        for x in ordering:
            if is_valid_insertion(x, S):
                S.add(x)
        return S

    # Phase 1: Multiple Randomized Greedy Constructions
    best_S = set()
    best_size = 0
    start_time = time.time()
    time_limit = 55.0
    
    # Run many random greedy constructions
    # 128 elements, is_valid is fast. We can run hundreds of these.
    for i in range(500):
        if time.time() - start_time > time_limit * 0.4:
            break
        random.seed(i)
        ordering = list(range(1, N + 1))
        random.shuffle(ordering)
        S = greedy_construct(ordering)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # Phase 2: Local Search on the best set found
    # We try to add any element from the complement.
    # If stuck, we shake.
    
    shake_count = 0
    while time.time() - start_time < time_limit:
        current_S = best_S
        complement = list(set(range(1, N + 1)) - current_S)
        
        improved = False
        # Try to add any element from the complement
        # Shuffle to avoid bias
        random.shuffle(complement)
        for x in complement:
            if is_valid_insertion(x, current_S):
                current_S.add(x)
                improved = True
        
        if improved:
            best_S = current_S
            best_size = len(best_S)
            shake_count = 0
            continue
            
        # If no simple improvement, perform a shake
        # Remove k random elements and try to re-insert everything in random order
        k = 3 + (shake_count // 50)
        k = min(k, 10)
        shake_count += 1
        
        if len(current_S) < k:
            break
        to_shake = random.sample(list(current_S), k)
        
        new_S = set(current_S)
        for x in to_shake:
            new_S.remove(x)
        
        # The pool of candidates to re-insert is the shaken elements + original complement
        candidates = list(to_shake) + [x for x in complement if x not in to_shake]
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_insertion(x, new_S):
                new_S.add(x)
                
        if len(new_S) > best_size:
            best_S = new_S
            best_size = len(best_S)
            shake_count = 0

    return list(best_S)
# EVOLVE-BLOCK-END
