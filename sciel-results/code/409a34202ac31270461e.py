import numpy as np
import itertools
import random
import time

def construct():
    n = 192
    max_time = 55.0 # Leave 5s buffer
    
    # Initial solution: Greedy
    selected = []
    for x in range(n, 0, -1):
        if is_valid_add(x, selected):
            selected.append(x)
    
    best_set = set(selected)
    best_size = len(selected)
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        current_set = list(best_set)
        random.shuffle(current_set) # Randomize order for tie-breaking or exploration? 
        # Actually, we work with the set directly.
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any element
            added = False
            # Sort candidates by some heuristic? Or just random?
            # Let's try adding elements in random order to explore different paths
            candidates = [x for x in range(1, n+1) if x not in best_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_add(x, list(best_set)):
                    best_set.add(x)
                    best_size += 1
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # 2. If no single add, try a swap: remove one, add two
            # Pick a random element to remove
            if not best_set:
                break
            y = random.choice(list(best_set))
            new_set = best_set - {y}
            
            # Try to find two elements x1, x2 not in new_set that can be added
            candidates = [x for x in range(1, n+1) if x not in new_set and x != y]
            random.shuffle(candidates)
            
            # We need to check if x1 and x2 can both be added to new_set
            # This is O(N^2) checks. N=192. 192^2 = 36k checks.
            # Each check is O(S^3) ~ 100k ops. 36k * 100k = 3.6e9 ops. Too slow.
            
            # Optimization:
            # First, filter candidates that are individually valid with new_set?
            # No, x1 and x2 interact.
            
            # Better approach:
            # Try to add x1 to new_set. If valid, try to add x2 to new_set U {x1}.
            # We can pre-filter x1 that are valid with new_set.
            valid_x1 = []
            for x in candidates:
                if is_valid_add(x, list(new_set)):
                    valid_x1.append(x)
            
            if len(valid_x1) < 2:
                continue # Can't even find 2 individually valid ones? 
                # Wait, x1 might be invalid with new_set alone?
                # If x1 is invalid with new_set, it can't be in the final set.
                # So this filter is correct.
                
            # Now try pairs from valid_x1
            found_pair = False
            for x1, x2 in itertools.combinations(valid_x1, 2):
                # Check if x2 is valid with new_set U {x1}
                # Note: is_valid_add(x2, new_set U {x1}) checks:
                # 1. x2 vs triples in new_set U {x1}
                # 2. elements in new_set U {x1} vs pairs involving x2
                if is_valid_add(x2, list(new_set) + [x1]):
                    best_set = new_set | {x1, x2}
                    best_size += 1
                    improved = True
                    found_pair = True
                    break
            
            if found_pair:
                continue
        
        # If no improvement in this pass, we might be stuck.
        # The "improved" loop breaks if no add and no swap found.
        
    return list(best_set)

def is_valid_add(x, selected_list):
    # selected_list is a list of ints
    # x is the new int
    # Check 1: x does not divide sum of any 3 in selected_list
    if len(selected_list) >= 3:
        # Use itertools for clarity, but might be slow
        # Let's try to optimize this
        for u, v, w in itertools.combinations(selected_list, 3):
            if (u + v + w) % x == 0:
                return False
                
    # Check 2: For each a in selected_list, a does not divide x + u + v
    # where u, v are distinct in selected_list and u,v != a
    for a in selected_list:
        # Get elements other than a
        others = [z for z in selected_list if z != a]
        if len(others) < 2:
            continue
        for u, v in itertools.combinations(others, 2):
            if (x + u + v) % a == 0:
                return False
    return True
