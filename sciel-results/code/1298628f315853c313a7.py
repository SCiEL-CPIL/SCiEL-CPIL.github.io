"""
Rewrite using a Max-SAT style greedy construction with conflict resolution.
"""
import math
import random

def construct():
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c, all in [1, N]
    # such that 1/a = 1/b + 1/c  <=>  b*c = a*(b+c)
    # Note: a < b and a < c (since 1/a > 1/b and 1/a > 1/c)
    # Also b < c w.l.o.g. (symmetric in b, c)
    
    # For efficiency, iterate over possible (b, c) pairs and compute a
    # a = (b*c) / (b+c)
    # We need a to be an integer and a < b (since a is the smallest)
    # Also a >= 1 and a <= N
    
    forbidden = []  # list of (a, b, c)
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a < b:
                    forbidden.append((a, b, c))
    
    # Map each element to the list of triples it belongs to
    # triples[x] = list of indices of triples containing x
    triples = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(forbidden):
        triples[a].append(i)
        triples[b].append(i)
        triples[c].append(i)
    
    # Greedy Conflict-Driven Removal
    # Start with all elements included
    S = set(range(1, N + 1))
    
    # To speed up conflict checking, we maintain a count of how many elements of each triple are in S
    # triple_count[i] = number of elements of forbidden[i] that are in S
    triple_count = [0] * len(forbidden)
    for i, (a, b, c) in enumerate(forbidden):
        triple_count[i] = 3 # Initially all in S
        
    # We need to efficiently find conflicts and decide what to remove.
    # A conflict exists if triple_count[i] == 3.
    
    # We will iteratively remove elements.
    # Heuristic: Remove the element involved in the most active conflicts.
    
    # To make this efficient, we can track the "weight" of each element:
    # weight[x] = number of triples containing x that are currently fully in S (conflicts)
    # plus maybe a fraction of those partially in S?
    # Let's stick to: weight[x] = number of conflicts (triple_count == 3) containing x.
    
    # We need a way to update weights efficiently.
    # When we remove x, we update triple_count for all triples containing x.
    # If a triple drops from 3 to 2, it is no longer a conflict.
    # So for each triple containing x, if it was a conflict, we decrement weight of a, b, c.
    
    # Let's implement this.
    
    # Initialize weights
    weights = [0] * (N + 1)
    # Initially, all triples with count 3 are conflicts.
    # But wait, do all triples have count 3 initially? Yes, S is full.
    # So weight[x] is just the number of triples containing x.
    for x in range(1, N + 1):
        weights[x] = len(triples[x])
        
    # We need a priority queue or just scan for max weight?
    # Scanning 180 elements is cheap.
    
    while True:
        # Find if any conflict exists
        # A conflict is a triple with count 3.
        # We can check this by seeing if any weight > 0?
        # No, weight > 0 means it's in at least one conflict.
        # But if weights are all 0, no conflicts.
        
        # Find element with max weight
        max_w = -1
        best_x = -1
        for x in range(1, N + 1):
            if x in S and weights[x] > max_w:
                max_w = weights[x]
                best_x = x
                
        if max_w <= 0:
            break
            
        # Remove best_x
        S.remove(best_x)
        
        # Update triple counts and weights
        for t_idx in triples[best_x]:
            # This triple had best_x in it.
            # Since best_x was in S, and we are removing it, count decreases by 1.
            triple_count[t_idx] -= 1
            
            # If it was a conflict (count was 3) and is now 2, it stops being a conflict.
            # We need to decrement weights of the other two elements in this triple.
            if triple_count[t_idx] == 2:
                a, b, c = forbidden[t_idx]
                # The other two elements are the ones in S
                for y in (a, b, c):
                    if y != best_x and y in S:
                        weights[y] -= 1
                        # Note: weights[y] cannot go negative if logic is correct,
                        # because we only decrement when a conflict is resolved.
                        
    # Now S is a valid set.
    # Try to improve by adding back removed elements?
    # This is the "Local Search" part.
    
    # List of removed elements
    removed = [x for x in range(1, N + 1) if x not in S]
    
    # We can try to add them back in a random order or sorted by "potential".
    # Potential: how many conflicts would it cause? (Should be 0 if we can add it).
    # We can try to add any removed element that doesn't create a conflict.
    
    # To check if x can be added:
    # For each triple containing x, check if the other two are in S.
    
    def can_add(x, current_S):
        for t_idx in triples[x]:
            a, b, c = forbidden[t_idx]
            # Check if the other two are in current_S
            others = [y for y in (a, b, c) if y != x]
            if others[0] in current_S and others[1] in current_S:
                return False
        return True

    # Try to add back removed elements
    # We can do this iteratively.
    # Sort removed by some heuristic? Maybe by original weight?
    # Or just random shuffle for diversity?
    
    # Let's try adding them in descending order of "degree" (len(triples[x]))
    # Higher degree elements are more constrained, so maybe try them first?
    # Or lower degree first?
    # Let's try random restarts of the addition phase.
    
    best_S = set(S)
    best_size = len(S)
    
    # Run a few iterations of local search
    for _ in range(50):
        S_curr = set(S)
        # Shuffle removed
        rem = [x for x in range(1, N + 1) if x not in S_curr]
        random.shuffle(rem)
        
        for x in rem:
            if can_add(x, S_curr):
                S_curr.add(x)
                
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_S = set(S_curr)
            
    # One more pass: try to swap elements?
    # If we can't add x, maybe we can remove y and add x?
    # This is complex. The simple add-back is usually good enough for this density.
    
    return list(best_S)
