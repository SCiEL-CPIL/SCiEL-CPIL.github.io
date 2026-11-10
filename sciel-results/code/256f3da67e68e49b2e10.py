import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # 1. Greedy Construction
    selected = []
    for x in range(192, 0, -1):
        candidate = selected + [x]
        # Check validity
        valid = True
        # Check x against existing
        # x divides sum of 3 others?
        for a, b, c in itertools.combinations(selected, 3):
            if (a + b + c) % x == 0:
                valid = False
                break
        if valid:
            # Check existing a against x + 2 others
            for a in selected:
                others = [v for v in selected if v != a]
                for b, c in itertools.combinations(others, 2):
                    if (x + b + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            selected.append(x)
            
    best_set = selected[:]
    best_size = len(best_set)
    
    # 2. Backtracking Search for improvement
    # We try to find a set of size best_size + 1, then +2, etc.
    
    def is_valid_addition(x, S):
        # S is list
        # 1. x does not divide sum of any 3 from S
        for a, b, c in itertools.combinations(S, 3):
            if (a + b + c) % x == 0:
                return False
        
        # 2. For each a in S, a does not divide x + b + c for any b, c in S \ {a}
        for a in S:
            others = [v for v in S if v != a]
            for b, c in itertools.combinations(others, 2):
                if (x + b + c) % a == 0:
                    return False
        return True

    def search(start_idx, current_set, current_size, target_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
            
        if current_size == target_size:
            best_set = current_set[:]
            best_size = current_size
            return True
            
        # Pruning
        # Max possible size is current_size + (192 - start_idx + 1)
        if current_size + (192 - start_idx + 1) < target_size:
            return False
            
        # Iterate candidates from start_idx to 192
        # We want to try large numbers first? 
        # The greedy went 192 down to 1.
        # In backtracking, if we go 192 down to 1, we are consistent.
        # But 'start_idx' usually implies increasing index.
        # Let's map indices such that higher values are tried first.
        # Range is 1..192.
        # If we use range(192, 0, -1), start_idx is the current value.
        
        for x in range(start_idx, 0, -1):
            if is_valid_addition(x, current_set):
                if search(x - 1, current_set + [x], current_size + 1, target_size):
                    return True
        return False

    # Try to improve
    target = best_size + 1
    while target <= 48:
        # Reset current set to empty? 
        # Or start from a promising subset?
        # Starting from empty is safer but slower.
        # Starting from greedy set's prefix?
        # Let's start from empty to avoid bias, but use the order 192..1.
        found = search(192, [], 0, target)
        if found:
            target += 1
        else:
            break

    return best_set
