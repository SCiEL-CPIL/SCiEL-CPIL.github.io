import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin (90s limit, 35s buffer)
    
    n = 192
    k = 3
    
    # Precompute: for each pair (i,j) with i < j, compute i+j
    # We need to check if any selected a divides (b+c+d) where b,c,d are other selected
    
    # Strategy: Greedy construction with multiple random orderings, then local improvement
    # Key insight: larger numbers are less likely to divide sums of 3 others
    # Numbers > 96: sum of 3 numbers each <= 192 is at most 576, so only divisors matter
    
    # Precompute conflict: a conflicts with (b,c,d) if b+c+d is divisible by a and b,c,d != a
    # But we check incrementally during construction
    
    best_solution = []
    best_size = 0
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if trial % 3 == 0:
            # Descending order (largest first)
            order = list(range(n, 0, -1))
        elif trial % 3 == 1:
            # Random shuffle
            order = list(range(1, n + 1))
            rng.shuffle(order)
        else:
            # Weighted: prefer larger numbers
            order = list(range(1, n + 1))
            rng.shuffle(order)
            order.sort(key=lambda x: -x)
        
        selected = []
        selected_set = set()
        
        for candidate in order:
            if time.time() > deadline:
                break
            
            # Check if adding candidate violates the condition
            valid = True
            
            # For each a in selected + [candidate], check all triples of distinct others
            # This is expensive, so let's be smart
            
            # When we add 'candidate', we need to check:
            # 1. For a = candidate: no triple from selected sums to multiple of candidate
            # 2. For each a in selected: no triple from (selected \ {a}) ∪ {candidate} sums to multiple of a
            
            # Check 1: candidate divides sum of any 3 from selected
            if len(selected) >= 3:
                sel_list = list(selected)
                for i in range(len(sel_list)):
                    for j in range(i+1, len(sel_list)):
                        for l in range(j+1, len(sel_list)):
                            if (sel_list[i] + sel_list[j] + sel_list[l]) % candidate == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
            
            if not valid:
                continue
            
            # Check 2: for each a in selected, check if candidate + 2 from selected (excluding a) 
            # or 3 from selected (excluding a) sums to multiple of a
            # Actually we need: for each a in selected, all triples from (selected \ {a}) ∪ {candidate}
            # These triples are either:
            #   - 3 from selected \ {a} (already checked before)
            #   - 2 from selected \ {a} + candidate
            #   - 1 from selected \ {a} + ... no, need exactly 3
            # So new triples involving candidate: 2 from selected\{a} + candidate
            
            for a in selected:
                # Check triples: candidate + x + y where x,y in selected, x != a, y != a, x != y
                if len(selected) >= 2:
                    for i, x in enumerate(selected):
                        if x == a:
                            continue
                        for j, y in enumerate(selected):
                            if j <= i:
                                continue
                            if y == a or x == a:
                                continue
                            if (candidate + x + y) % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
            
            if valid:
                selected.append(candidate)
                selected_set.add(candidate)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_solution = list(selected)
        
        # Early termination if we've found a good solution
        if best_size >= 40:
            break
    
    # Local improvement: try to swap elements to increase size
    if time.time() < deadline - 5:
        # Try removing one element and adding two others
        for trial2 in range(50):
            if time.time() > deadline - 5:
                break
            
            current = list(best_solution)
            current_set = set(current)
            improved = False
            
            # Try removing each element and adding a new one
            for idx in range(len(current)):
                if time.time() > deadline - 5:
                    break
                
                removed = current[idx]
                remaining = current[:idx] + current[idx+1:]
                remaining_set = set(remaining)
                
                # Try adding each candidate not in remaining
                for cand in range(1, n + 1):
                    if cand in remaining_set:
                        continue
                    
                    new_set = remaining + [cand]
                    new_set_arr = list(new_set)
                    
                    # Validate
                    valid = True
                    for a in new_set_arr:
                        others = [v for v in new_set_arr if v != a]
                        if len(others) >= 3:
                            for tri in combinations(others, 3):
                                if sum(tri) % a == 0:
                                    valid = False
                                    break
                        if not valid:
                            break
                    
                    if valid and len(new_set_arr) > len(best_solution):
                        best_solution = new_set_arr
                        best_size = len(new_set_arr)
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_solution
