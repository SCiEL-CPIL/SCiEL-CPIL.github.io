import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 128
    # Start with upper block
    best_set = set(range(65, 129))  # 64 elements
    
    def is_valid(s):
        """Check if set s has no forbidden triple."""
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                for k in range(j+1, len(lst)):
                    a, b, c = lst[i], lst[j], lst[k]
                    if (a+b) in s and (a+c) in s and (b+c) in s:
                        return False
        return True
    
    def can_add(s, x):
        """Check if adding x to s keeps it valid (only check triples involving x)."""
        # Triples where x is one of a,b,c
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a, b = lst[i], lst[j]
                # x, a, b
                if (x+a) in s and (x+b) in s and (a+b) in s:
                    return False
                # a, b, x is same as above
        return True
    
    # Greedy add: try to add lower elements in descending order
    for x in range(64, 0, -1):
        if time.time() > deadline:
            break
        if can_add(best_set, x):
            best_set.add(x)
    
    best_size = len(best_set)
    
    # Local search: try swaps (remove one, add another)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        lst = sorted(best_set)
        for i in range(len(lst)):
            if time.time() > deadline:
                break
            removed = lst[i]
            trial = best_set - {removed}
            # Try to add elements not in trial
            for x in range(1, 129):
                if time.time() > deadline:
                    break
                if x in trial or x == removed:
                    continue
                if can_add(trial, x):
                    new_set = trial | {x}
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_size = len(best_set)
                        improved = True
                        break
    
    # Final validation
    result = sorted(best_set)
    assert is_valid(set(result)), "Final set is invalid!"
    assert all(isinstance(x, int) for x in result), "Non-int in result"
    assert all(1 <= x <= 128 for x in result), "Out of range"
    assert len(set(result)) == len(result), "Duplicates"
    
    return result
