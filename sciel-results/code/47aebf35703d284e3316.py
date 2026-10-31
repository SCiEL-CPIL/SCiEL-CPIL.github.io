import math
import random
import copy

def calculate_score(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    
    # Convert to set for faster lookup? No, we need to compute sums/diffs.
    # Using sets is fine for small N.
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    val = 1 + (math.log(ratio) / denom)
    return val

def construct():
    # Candidate sets known to be good or starting points
    candidates = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 4, 10],
        [0, 2, 4, 8],
        [0, 1, 2, 4, 8],
    ]
    
    best_set = candidates[0]
    best_score = calculate_score(best_set)
    
    for c in candidates[1:]:
        s = calculate_score(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    # Local Search / Hill Climbing
    # Try to improve by swapping elements or adding/removing
    # We limit the range of values to keep it fast
    max_val = 50
    current_set = sorted(best_set)
    current_score = best_score
    
    for _ in range(1000): # Iterations
        improved = False
        # Try adding a new element
        # Pick a random candidate from 1..max_val not in set
        available = [x for x in range(1, max_val + 1) if x not in current_set]
        if available:
            new_elem = random.choice(available)
            new_set = sorted(current_set + [new_elem])
            new_score = calculate_score(new_set)
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                
        # Try removing an element (if size > 2)
        if len(current_set) > 2:
            # Remove a random non-zero element
            non_zero = [x for x in current_set if x != 0]
            if non_zero:
                rem_elem = random.choice(non_zero)
                new_set = [x for x in current_set if x != rem_elem]
                new_score = calculate_score(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    
        # Try swapping an element
        if len(current_set) > 2:
            non_zero = [x for x in current_set if x != 0]
            if non_zero:
                swap_idx = random.choice(range(len(non_zero)))
                old_elem = non_zero[swap_idx]
                new_elem = random.randint(1, max_val)
                if new_elem not in current_set:
                    new_set = [x for x in current_set if x != old_elem]
                    new_set.append(new_elem)
                    new_set.sort()
                    new_score = calculate_score(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        
        if not improved:
            # If stuck, try a random restart or break
            # For simplicity, break if no improvement in a few iterations
            # Here we just break the loop to save time if stuck
            break
            
    if current_score > best_score:
        best_set = current_set
        best_score = current_score
        
    return best_set
