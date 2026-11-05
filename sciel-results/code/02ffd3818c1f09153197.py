import random
from itertools import combinations
from math import lcm, gcd

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute forbidden pairs for each x
    # conflict_pairs[x] contains tuple (a, b) such that lcm(a,b) == lcm(a,x) == lcm(b,x)
    # We store sorted tuples (a,b) with a < b
    conflict_pairs = {x: set() for x in numbers}
    
    for a, b, c in combinations(numbers, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            # Add (a,b) to conflict_pairs[c], etc.
            conflict_pairs[c].add(tuple(sorted((a, b))))
            conflict_pairs[b].add(tuple(sorted((a, c))))
            conflict_pairs[a].add(tuple(sorted((b, c))))

    def greedy_construction(order):
        S = set()
        S_pairs = set()
        
        for x in order:
            # Check if x conflicts with any pair in S
            # x can be added if conflict_pairs[x] is disjoint from S_pairs
            if S_pairs.isdisjoint(conflict_pairs[x]):
                S.add(x)
                # Update S_pairs with new pairs involving x
                for y in S:
                    if y != x:
                        S_pairs.add(tuple(sorted((x, y))))
        return S

    def local_search(S):
        S = set(S)
        S_pairs = set(tuple(sorted(p)) for p in combinations(S, 2))
        all_numbers = set(range(1, N + 1))
        
        # Try to improve by adding or swapping
        # We iterate a fixed number of times to avoid infinite loops
        for _ in range(200):
            improved = False
            
            # Phase 1: Try to add any element that fits
            candidates_to_add = all_numbers - S
            for x in candidates_to_add:
                if S_pairs.isdisjoint(conflict_pairs[x]):
                    S.add(x)
                    for y in S:
                        if y != x:
                            S_pairs.add(tuple(sorted((x, y))))
                    improved = True
                    break # Restart phase 1 with new S
            
            if improved:
                continue
                
            # Phase 2: Try to swap an element y in S with x not in S
            # We look for a swap that is valid.
            # To make it deterministic and avoid cycling, we can prioritize swaps
            # that replace a 'high degree' node with a 'low degree' node? 
            # Or just find any valid swap.
            
            found_swap = False
            for x in candidates_to_add:
                if found_swap:
                    break
                # Find pairs in S that conflict with x
                conf_pairs = S_pairs.intersection(conflict_pairs[x])
                if not conf_pairs:
                    # Should have been added in Phase 1
                    continue
                
                # Identify elements involved in these conflicts
                involved = set()
                for a, b in conf_pairs:
                    involved.add(a)
                    involved.add(b)
                    
                # Try removing each y in involved
                for y in involved:
                    # Check if S - {y} + {x} is valid
                    # New S_pairs would be S_pairs minus pairs involving y, plus pairs involving x (with S-{y})
                    # We need to check if conflict_pairs[x] is disjoint from (S_pairs - pairs_involving_y)
                    
                    # Pairs involving y in S_pairs
                    pairs_to_remove = set()
                    for p in S_pairs:
                        if y in p:
                            pairs_to_remove.add(p)
                            
                    new_S_pairs = S_pairs - pairs_to_remove
                    
                    # Check validity of x against new_S_pairs
                    if new_S_pairs.isdisjoint(conflict_pairs[x]):
                        # Perform swap
                        S.remove(y)
                        S.add(x)
                        
                        # Update S_pairs
                        S_pairs = new_S_pairs
                        for z in S:
                            if z != x:
                                S_pairs.add(tuple(sorted((x, z))))
                        
                        found_swap = True
                        improved = True
                        break
                        
            if not improved:
                break
                
        return S

    best_set = set()
    
    # 1. Deterministic Greedy
    best_set = greedy_construction(numbers)
    best_set = local_search(best_set)
    
    best_size = len(best_set)
    
    # 2. Reverse Greedy
    S_rev = greedy_construction(sorted(numbers, reverse=True))
    S_rev = local_search(S_rev)
    if len(S_rev) > best_size:
        best_set = S_rev
        best_size = len(S_rev)
        
    # 3. Randomized Greedy + Local Search
    for _ in range(100):
        random.shuffle(numbers)
        S_rand = greedy_construction(numbers)
        S_rand = local_search(S_rand)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
            
    if not best_set:
        return [1]
        
    return list(best_set)
