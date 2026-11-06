import time
from itertools import combinations
from math import gcd, lcm

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0  # safety margin before 90s kill

    # Precompute lcm for all pairs
    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            lcm_table[a][b] = lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]

    # Find all bad triples: three distinct elements {a,b,c} where
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    bad_triples.add((a, b, c))

    # Build conflict graph: two elements conflict if they appear together in some bad triple
    # Actually, we need a hypergraph approach. Let's use a different strategy.
    # 
    # Better approach: Use a greedy + local search (maximizing independent set in 3-uniform hypergraph)
    # 
    # Strategy: 
    # 1. Start with all elements
    # 2. Iteratively remove elements from bad triples (the one that's in the most bad triples)
    # 3. Then try to add back removed elements if possible
    # 4. Use simulated annealing / local search for improvement

    all_elements = set(range(1, n + 1))

    def count_bad_triples_in_set(S):
        """Count bad triples within set S."""
        S_list = sorted(S)
        count = 0
        for triple in bad_triples:
            if all(x in S for x in triple):
                count += 1
        return count

    def is_valid_set(S):
        """Check if set S contains no bad triple."""
        for triple in bad_triples:
            if all(x in S for x in triple):
                return False
        return True

    # Initial greedy: start with all, remove elements that are in most bad triples
    S = set(all_elements)
    
    # Count how many bad triples each element participates in
    elem_triple_count = {i: 0 for i in all_elements}
    for triple in bad_triples:
        for x in triple:
            elem_triple_count[x] += 1

    # Iteratively remove the element in the most bad triples (that are still "active")
    while count_bad_triples_in_set(S) > 0 and time.time() < DEADLINE:
        # Find active bad triples
        active_bad = []
        for triple in bad_triples:
            if all(x in S for x in triple):
                active_bad.append(triple)
        
        if not active_bad:
            break
        
        # Count participation in active bad triples
        active_count = {}
        for triple in active_bad:
            for x in triple:
                active_count[x] = active_count.get(x, 0) + 1
        
        # Remove the element with highest active count
        worst_elem = max(active_count, key=active_count.get)
        S.discard(worst_elem)

    best_S = set(S)
    best_size = len(S)

    # Local search: try swapping or adding/removing to improve
    # Phase 1: Try to add back removed elements
    removed = all_elements - S
    
    # Try adding each removed element back
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for elem in sorted(removed):
            trial_S = S | {elem}
            if is_valid_set(trial_S):
                S = trial_S
                removed.discard(elem)
                improved = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)

    # Phase 2: Local search - swap elements
    # For each pair (in_S, not_in_S), try swapping
    for _iteration in range(200):
        if time.time() > DEADLINE:
            break
        found_swap = False
        S_list = sorted(S)
        R_list = sorted(removed)
        for a in S_list:
            for b in R_list:
                trial_S = (S - {a}) | {b}
                if is_valid_set(trial_S):
                    # Check if this is an improvement (should be same size, but let's keep it)
                    S = trial_S
                    removed = all_elements - S
                    found_swap = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    break
            if found_swap:
                break
        if not found_swap:
            break

    # Phase 3: More aggressive local search with random restarts
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > DEADLINE:
            break
        
        # Start from a random subset biased toward larger sets
        # Try random greedy with different orderings
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S_rand = set()
        for elem in order:
            trial = S_rand | {elem}
            if is_valid_set(trial):
                S_rand = trial
        
        if len(S_rand) > best_size:
            best_S = set(S_rand)
            best_size = len(S_rand)
        
        # Local search from S_rand
        S_ls = set(S_rand)
        removed_ls = all_elements - S_ls
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for elem in sorted(removed_ls):
                trial = S_ls | {elem}
                if is_valid_set(trial):
                    S_ls = trial
                    removed_ls.discard(elem)
                    improved = True
        
        if len(S_ls) > best_size:
            best_S = set(S_ls)
            best_size = len(S_ls)

    return sorted(best_S) if best_S else [1]
