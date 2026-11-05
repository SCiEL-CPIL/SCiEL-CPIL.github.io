"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 192
    all_elements = list(range(1, n + 1))
    
    # Precompute: for each element a, which 3-subsets of OTHER elements sum to multiple of a
    # conflict_mask[a] is a set of frozensets (or we use a different approach)
    # Actually, let's use a smarter approach:
    # For a given selected set S, element a in S is "bad" if there exist b,c,d in S\{a} with (b+c+d) % a == 0
    
    # Precompute all 3-sums for each possible a
    # For efficiency, precompute sum3_mod[a] = set of all (b,c,d) triples from [1..n]\{a} where (b+c+d)%a==0
    # But that's huge. Instead, let's precompute for each a, the list of b values that could form such triples.
    
    # Better approach: precompute for each pair (a, b) where b != a, 
    # and for the current selected set, we can check conflicts incrementally.
    
    # Let's use a different strategy: 
    # Precompute all 3-element subsets and their sums
    all_triples = []
    for a, b, c in combinations(range(1, n + 1), 3):
        all_triples.append((a, b, c, a + b + c))
    
    # For each element x, precompute which triples (not containing x) have sum divisible by x
    # conflict_triples[x] = list of (a,b,c) where a,b,c != x and (a+b+c) % x == 0
    conflict_triples = {x: [] for x in range(1, n + 1)}
    for a, b, c, s in all_triples:
        for x in range(1, n + 1):
            if x not in (a, b, c) and s % x == 0:
                conflict_triples[x].append((a, b, c))
    
    # Now for a selected set S, element x in S is valid iff no triple in conflict_triples[x] is fully contained in S
    # We can check this by: for each triple (a,b,c) in conflict_triples[x], check if a,b,c all in S
    
    # For fast checking, let's use a boolean array for membership
    # selected_bool[i] = True if i in selected
    
    # Greedy initial solution
    selected = set()
    for x in range(n, 0, -1):
        # Check if x can be added
        can_add = True
        for a, b, c in conflict_triples[x]:
            if a in selected and b in selected and c in selected:
                can_add = False
                break
        if can_add:
            # Also check that existing elements remain valid with x added
            for y in selected:
                for a, b, c in conflict_triples[y]:
                    if a == x and b in selected and c in selected:
                        can_add = False
                        break
                    elif b == x and a in selected and c in selected:
                        can_add = False
                        break
                    elif c == x and a in selected and b in selected:
                        can_add = False
                        break
                if not can_add:
                    break
        if can_add:
            selected.add(x)
    
    best_set = set(selected)
    best_size = len(selected)
    
    # Simulated annealing
    if time.time() < deadline - 5:
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        # Precompute for faster conflict checking
        # For each x, store conflict_triples[x] as a list of tuples
        
        while time.time() < deadline - 2:
            if T < T_min:
                break
            
            # Random move: remove one element and/or add one
            r = random.random()
            if r < 0.4:
                # Remove one element
                if len(selected) > 0:
                    x = random.choice(list(selected))
                    new_set = selected - {x}
                    # Check validity of new_set
                    valid = True
                    for y in new_set:
                        for a, b, c in conflict_triples[y]:
                            if a in new_set and b in new_set and c in new_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_size = len(new_set)
                        delta = new_size - len(selected)
                        if delta >= 0 or random.random() < min(1.0, delta / T):
                            selected = new_set
                            if len(selected) > best_size:
                                best_size = len(selected)
                                best_set = set(selected)
            elif r < 0.8:
                # Add one element
                candidates = [x for x in range(1, n + 1) if x not in selected]
                if candidates:
                    x = random.choice(candidates)
                    new_set = selected | {x}
                    # Check validity
                    valid = True
                    # Check x
                    for a, b, c in conflict_triples[x]:
                        if a in new_set and b in new_set and c in new_set:
                            valid = False
                            break
                    if valid:
                        # Check existing elements
                        for y in selected:
                            for a, b, c in conflict_triples[y]:
                                if a == x and b in new_set and c in new_set:
                                    valid = False
                                    break
                                elif b == x and a in new_set and c in new_set:
                                    valid = False
                                    break
                                elif c == x and a in new_set and b in new_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        new_size = len(new_set)
                        delta = new_size - len(selected)
                        if delta >= 0 or random.random() < min(1.0, delta / T):
                            selected = new_set
                            if len(selected) > best_size:
                                best_size = len(selected)
                                best_set = set(selected)
            else:
                # Swap: remove one, add one
                if len(selected) > 0:
                    x = random.choice(list(selected))
                    candidates = [y for y in range(1, n + 1) if y not in selected]
                    if candidates:
                        y = random.choice(candidates)
                        new_set = (selected - {x}) | {y}
                        # Check validity
                        valid = True
                        for z in new_set:
                            for a, b, c in conflict_triples[z]:
                                if a in new_set and b in new_set and c in new_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            new_size = len(new_set)
                            delta = new_size - len(selected)
                            if delta >= 0 or random.random() < min(1.0, delta / T):
                                selected = new_set
                                if len(selected) > best_size:
                                    best_size = len(selected)
                                    best_set = set(selected)
            
            T *= cooling
    
    return list(best_set)
# EVOLVE-BLOCK-END
