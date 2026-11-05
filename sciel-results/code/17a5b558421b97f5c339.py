"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    n = 256
    
    # Precompute conflict information
    # For each a, and for each pair (b,c) with a < b <= c <= n,
    # check if a divides b+c
    # We'll build a "forbidden pairs" structure: for each a, which (b,c) pairs conflict
    
    # Better approach: for each a, compute all multiples k*a where k>=3 and k*a <= 512
    # Then for each such sum s = k*a, find all pairs (b,c) in [a+1, 256] with b<=c and b+c=s
    # These pairs are "forbidden" when a is in the set
    
    # For greedy independent set: we want to select a subset S such that
    # for no a in S, and no b,c in S with a<b<=c, does a | (b+c)
    
    # Let's use a different formulation:
    # A set S is valid iff for all a in S, for all b,c in S with a<b<=c, a does not divide b+c
    
    # Greedy approach with random restarts:
    # 1. Start with all elements [1, 256]
    # 2. Repeatedly remove the element that is involved in the most conflicts
    # 3. Try different orderings / random perturbations
    
    # Actually, let's think of it as: we want to find the largest valid subset.
    # This is like a maximum independent set in a hypergraph.
    
    # Let's use a simpler greedy: build the set element by element
    # For each candidate element x (in some order), check if adding it to current set S
    # creates any conflict. If not, add it.
    
    # Conflict check: adding x to S creates a conflict if:
    # - There exist b,c in S with x < b <= c and x | (b+c)
    # - There exist a in S with a < x <= c and a | (x+c) for some c in S with c >= x
    # - There exist a in S with a < b <= x and a | (b+x) for some b in S with b <= x and b > a
    
    # More precisely, when we add x to S:
    # 1. For each a in S with a < x: check if there exists c in S with c >= x such that a | (x+c)
    #    (here b=x, c>=x, and a<b<=c means a<x<=c)
    # 2. For each a in S with a < x: check if there exists b in S with a < b < x such that a | (b+x)
    #    (here b<x, c=x, and a<b<c means a<b<x)
    # 3. For each b in S with x < b: check if there exists c in S with c >= b such that x | (b+c)
    #    (here a=x, b>x, c>=b, and a<b<=c means x<b<=c)
    # 4. Also need: for a=x, b=c=x: x | (2x) is always true! Wait...
    #    a < b <= c, so if a=x, then b > x, so b != x. So we don't have b=c=x case.
    #    But we do have b=c > x: a=x, b=c, x | (2b). So for each b in S with b > x,
    #    check if x | (2b).
    
    # Let me re-examine: the condition is "for every a<b<=c in the set, a must NOT divide b+c"
    # So when we add x to S, new triples involving x are:
    # - x is the smallest: x < b <= c, both b,c in S. Check x | (b+c) for all b,c in S with x < b <= c
    # - x is the middle: a < x <= c, a,c in S, a < x. Check a | (x+c) for all a,c in S with a < x <= c
    #   Note: x <= c means c >= x. And a < x.
    #   Sub-case: c = x: a < x, c=x. Check a | (x+x) = 2x. For all a in S with a < x.
    #   Sub-case: c > x: a < x < c. Check a | (x+c). For all a,c in S with a < x < c.
    # - x is the largest: a < b <= x, a,b in S. Check a | (b+x) for all a,b in S with a < b <= x
    #   Sub-case: b = x: a < x, b=x. Check a | (x+x) = 2x. For all a in S with a < x.
    #   Sub-case: b < x: a < b < x. Check a | (b+x). For all a,b in S with a < b < x.
    
    # So the conflicts when adding x are:
    # 1. x | (b+c) for b,c in S, x < b <= c
    # 2. a | (2x) for a in S, a < x  [from c=x or b=x cases]
    # 3. a | (x+c) for a,c in S, a < x < c
    # 4. a | (b+x) for a,b in S, a < b < x
    
    # This is expensive to check naively. Let's use a different approach.
    
    # Alternative: precompute for each x, which elements conflict with it.
    # Two elements can't directly conflict; it's about triples.
    
    # Let me use a simpler but effective approach:
    # Start with the known good set [171, 256] and try to add elements from below,
    # removing conflicting elements if the net gain is positive.
    
    # Or: use local search. Start with [171,256], try adding each element from 1 to 170,
    # and if it conflicts, try removing a minimal set of conflicting elements.
    
    # Let me implement a clean greedy with random restarts:
    
    def is_valid_set(S):
        """Check if S is a valid set."""
        S_sorted = sorted(S)
        S_set = set(S)
        for i, a in enumerate(S_sorted):
            for j in range(i+1, len(S_sorted)):
                b = S_sorted[j]
                for k in range(j, len(S_sorted)):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def count_conflicts_adding(x, S):
        """Count how many triples in S would become invalid if we add x."""
        S_sorted = sorted(S)
        conflicts = 0
        for i, a in enumerate(S_sorted):
            for j in range(i+1, len(S_sorted)):
                b = S_sorted[j]
                for k in range(j, len(S_sorted)):
                    c = S_sorted[k]
                    # Check if adding x creates a conflict with this triple,
                    # or if x is part of a new conflicting triple
                    # Actually, we need to check all triples that include x
                    pass
        # This approach is too complex. Let me just check validity directly.
        return None
    
    # Simpler: try to build a valid set greedily with different strategies
    
    best_set = list(range(171, 257))
    best_size = len(best_set)
    
    # Strategy 1: Greedy addition from the tail, trying to add smaller elements
    # For each x from 170 down to 1, try adding x to the current set.
    # If it creates conflicts, identify which elements to remove.
    
    def check_add(x, S):
        """Check if adding x to S creates any conflict. Return list of conflicting elements or None."""
        S_sorted = sorted(S)
        S_set = set(S)
        conflicting = set()
        
        # Check all triples that include x
        # Case 1: x is smallest: x < b <= c, b,c in S
        for b in S_sorted:
            if b <= x:
                continue
            for c in S_sorted:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    conflicting.add(b)
                    conflicting.add(c)
        
        # Case 2: x is middle or largest
        # a < x <= c: check a | (x+c)
        for a in S_sorted:
            if a >= x:
                continue
            # c >= x, c in S
            for c in S_sorted:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    conflicting.add(a)
                    conflicting.add(c)
        
        # a < b <= x: check a | (b+x)
        for a in S_sorted:
            if a >= x:
                continue
            for b in S_sorted:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    conflicting.add(a)
                    conflicting.add(b)
        
        return conflicting if conflicting else None
    
    # Try adding elements from 170 down to 1
    current_set = set(range(171, 257))
    
    for x in range(170, 0, -1):
        if time.time() > deadline - 5:
            break
        conf = check_add(x, current_set)
        if conf is None:
            current_set.add(x)
        elif len(conf) == 1:
            # Only one element conflicts, remove it and add x
            current_set.discard(conf.pop())
            current_set.add(x)
        elif len(conf) == 2:
            # Two elements conflict. Net zero change, skip.
            pass
    
    if len(current_set) > best_size:
        best_set = list(current_set)
        best_size = len(best_set)
    
    # Strategy 2: Random restarts with different initial orderings
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a random priority for elements
        priorities = {x: random.random() for x in range(1, 257)}
        
        # Greedy: process elements in order of priority, add if valid
        current_set = set()
        elements = sorted(range(1, 257), key=lambda x: priorities[x])
        
        for x in elements:
            if time.time() > deadline - 2:
                break
            conf = check_add(x, current_set)
            if conf is None:
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(best_set)
    
    # Strategy 3: Start from best_set and do local search (swap operations)
    if best_size > 86:
        # Try to improve further
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            improved = False
            current_set = set(best_set)
            
            for x in range(1, 171):
                if x in current_set:
                    continue
                conf = check_add(x, current_set)
                if conf is None:
                    current_set.add(x)
                    improved = True
                elif len(conf) <= 1:
                    for el in list(conf):
                        current_set.discard(el)
                    current_set.add(x)
                    improved = True
                    break
            
            if not improved:
                break
            
            if len(current_set) > best_size:
                best_set = list(current_set)
                best_size = len(best_set)
    
    return best_set
# EVOLVE-BLOCK-END
