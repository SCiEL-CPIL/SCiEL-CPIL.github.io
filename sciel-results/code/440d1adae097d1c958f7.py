import random
from itertools import combinations
from math import lcm

# Precompute bad triples
# conflicts[x] is a set of tuples (a, b) such that {a, b, x} is a bad triple
# We assume a < b for the pair to avoid duplicates in the set, but for checking we just need existence.
conflicts = {i: set() for i in range(1, 129)}

def is_bad_triple(a, b, c):
    if a == b or b == c or a == c:
        return False
    l_ab = lcm(a, b)
    l_ac = lcm(a, c)
    l_bc = lcm(b, c)
    return l_ab == l_ac == l_bc

# Precompute conflicts
for a in range(1, 129):
    for b in range(a + 1, 129):
        for c in range(b + 1, 129):
            if is_bad_triple(a, b, c):
                # Add (b, c) to conflicts[a]
                conflicts[a].add((b, c))
                # Add (a, c) to conflicts[b]
                conflicts[b].add((a, c))
                # Add (a, b) to conflicts[c]
                conflicts[c].add((a, b))

def construct():
    best_set = set()
    
    # Strategy: Start with full set, remove elements to resolve conflicts, then try to add back.
    # Run multiple times with different randomization to find the maximum size.
    
    for _ in range(100): # Increase iterations if time allows, 100 is safe for 60s? 
        # 128 elements, conflicts are sparse. 
        # Finding a bad triple takes O(N * avg_deg). 
        # Removal loop runs at most N times. 
        # So one run is O(N^2 * avg_deg). 
        # 100 runs should be fast.
        
        S = set(range(1, 129))
        removed = []
        
        # Phase 1: Remove elements until no bad triples
        while True:
            bad_triple_found = None
            # Find any bad triple
            # To make it more effective, we can pick the element with highest conflict degree in S
            # But finding it requires checking all.
            # Let's just find one bad triple quickly.
            # Iterate through elements in S in random order to diversify
            items = list(S)
            random.shuffle(items)
            
            for x in items:
                # Check if x is in any bad triple with elements in S
                for a, b in conflicts[x]:
                    if a in S and b in S:
                        bad_triple_found = (x, a, b)
                        break
                if bad_triple_found:
                    break
            
            if not bad_triple_found:
                break
            
            x, a, b = bad_triple_found
            # Decide which element to remove from {x, a, b}
            # Heuristic: Remove the one with the most conflicts within S
            # Calculate degree for x, a, b
            def get_degree_in_S(y):
                count = 0
                for p, q in conflicts[y]:
                    if p in S and q in S:
                        count += 1
                return count
            
            deg_x = get_degree_in_S(x)
            deg_a = get_degree_in_S(a)
            deg_b = get_degree_in_S(b)
            
            # Pick the one with max degree, tie-break randomly
            candidates = [(deg_x, x), (deg_a, a), (deg_b, b)]
            candidates.sort(key=lambda t: t[0], reverse=True)
            
            # If top two have same degree, pick randomly between them
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                to_remove = random.choice([candidates[0][1], candidates[1][1]])
            else:
                to_remove = candidates[0][1]
                
            S.remove(to_remove)
            removed.append(to_remove)
        
        # Phase 2: Try to add back removed elements
        # Sort removed elements. Maybe those with lower original degree are easier to add?
        # Or just random order.
        random.shuffle(removed)
        
        for x in removed:
            if x in S:
                continue
            # Check if adding x creates a bad triple
            can_add = True
            for a, b in conflicts[x]:
                if a in S and b in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
        
        if len(S) > len(best_set):
            best_set = S.copy()
            
        # Optional: Local Search / Swap
        # Try swapping an element in S with one in Removed?
        # This is complex. The add-back phase is a good approximation.
        
    return list(best_set)
