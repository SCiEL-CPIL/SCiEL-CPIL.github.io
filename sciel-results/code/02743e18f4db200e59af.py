import time
import random
import sys

def _check_collision(lst, sums_set, x):
    """
    Check if adding x to lst (with existing sums in sums_set) creates a collision.
    lst is sorted.
    """
    n = len(lst)
    
    # 3x
    if 3 * x in sums_set:
        return True
        
    # 2x + a
    # x + 2a
    for a in lst:
        if 2 * x + a in sums_set:
            return True
        if x + 2 * a in sums_set:
            return True
            
    # x + a + b
    for i in range(n):
        ai = lst[i]
        for j in range(i + 1, n):
            if x + ai + lst[j] in sums_set:
                return True
                
    return False

def _add_to_set(lst, sums_set, x):
    """
    Add x to lst and update sums_set.
    Assumes x is valid.
    lst is sorted. We append and sort? Or use bisect?
    For small n, append and sort is fine, but bisect.insort is O(n) vs O(n log n).
    Let's use bisect.insort.
    """
    import bisect
    bisect.insort(lst, x)
    
    n = len(lst) - 1 # length before insertion
    
    sums_set.add(3 * x)
    for i in range(n):
        a = lst[i]
        sums_set.add(2 * x + a)
        sums_set.add(x + 2 * a)
        
    for i in range(n):
        ai = lst[i]
        for j in range(i + 1, n):
            sums_set.add(x + ai + lst[j])

def _remove_from_set(lst, sums_set, x):
    """
    Remove x from lst and update sums_set.
    This is tricky because we need to know which sums to remove.
    It's easier to recompute sums_set from scratch for small n.
    """
    lst.remove(x)
    # Recompute sums
    new_sums = set()
    n = len(lst)
    for i in range(n):
        new_sums.add(3 * lst[i])
        for j in range(i, n):
            new_sums.add(2 * lst[i] + lst[j])
            for k in range(j, n):
                new_sums.add(lst[i] + lst[j] + lst[k])
    sums_set.clear()
    sums_set.update(new_sums)

def _compute_sums(lst):
    sums = set()
    n = len(lst)
    for i in range(n):
        sums.add(3 * lst[i])
        for j in range(i, n):
            sums.add(2 * lst[i] + lst[j])
            for k in range(j, n):
                sums.add(lst[i] + lst[j] + lst[k])
    return sums

def _greedy_fill(lst, sums_set, start_val=1, max_val=4096):
    """
    Try to add as many numbers as possible starting from start_val.
    Returns the updated lst and sums_set.
    """
    current_lst = list(lst)
    current_sums = set(sums_set)
    
    for x in range(start_val, max_val + 1):
        if x in current_sums: # x itself is not in sums, x is an element. 
            # Check if x is already in list
            # Since we iterate sequentially and list is sorted, 
            # if x > current_lst[-1] (or list empty), it's new.
            pass
            
        # Check if x is already in list
        # Simple check:
        if x in current_lst:
            continue
            
        if _check_collision(current_lst, current_sums, x):
            continue
            
        _add_to_set(current_lst, current_sums, x)
        
    return current_lst, current_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_lst = []
    best_sums = set()
    best_size = 0
    
    rng = random.Random(42)
    
    # Initial Seed: Powers of 4
    # 1, 4, 16, 64, 256, 1024, 4096
    seed = [1, 4, 16, 64, 256, 1024, 4096]
    # Validate seed
    seed_sums = _compute_sums(seed)
    if len(seed_sums) == len(list(seed_sums)): # Always true for set
         if _is_valid(seed):
             best_lst = list(seed)
             best_sums = seed_sums
             best_size = len(seed)
             
    # Strategy 1: Greedy from 1
    lst1, sums1 = _greedy_fill([], set(), 1, 4096)
    if len(lst1) > best_size:
        best_lst = lst1
        best_sums = sums1
        best_size = len(lst1)
        
    # Strategy 2: Local Search / Kick
    current_lst = list(best_lst)
    current_sums = set(best_sums)
    current_size = len(current_lst)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Try to add any number
        added = False
        # Randomly pick a range to search for additions?
        # Or just scan 1..4096?
        # Scanning is fast.
        for x in range(1, 4097):
            if x in current_lst:
                continue
            if _check_collision(current_lst, current_sums, x):
                continue
            _add_to_set(current_lst, current_sums, x)
            current_size += 1
            added = True
            break
            
        if added:
            if current_size > best_size:
                best_lst = list(current_lst)
                best_sums = set(current_sums)
                best_size = current_size
                no_improve_count = 0
            else:
                # Size didn't increase best, but we added something.
                # This shouldn't happen if we update best_size immediately.
                pass
            continue
            
        # If no addition possible, perform a kick
        if current_size > 0:
            # Remove a random element
            idx = rng.randint(0, current_size - 1)
            removed_val = current_lst.pop(idx)
            _remove_from_set(current_lst, current_sums, removed_val)
            current_size -= 1
            
            # Try to refill
            # 1. Try to add smaller numbers first
            # 2. Then larger numbers
            # We can just call _greedy_fill on the current state
            new_lst, new_sums = _greedy_fill(current_lst, current_sums, 1, 4096)
            
            if len(new_lst) > current_size:
                current_lst = new_lst
                current_sums = new_sums
                current_size = len(new_lst)
                
                if current_size > best_size:
                    best_lst = list(current_lst)
                    best_sums = set(current_sums)
                    best_size = current_size
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Refill didn't even restore size
                current_lst = new_lst
                current_sums = new_sums
                current_size = len(new_lst)
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                # Reset to best and try a different random seed for kicks?
                # Or just break and try a new random restart
                break
        else:
            break
            
    return best_lst

def _is_valid(lst):
    sums = _compute_sums(lst)
    # Check if any sum appears more than once
    # Since sums is a set, we just need to check if the number of generated sums
    # equals the size of the set.
    # But _compute_sums adds to a set, so duplicates are removed.
    # We need to count total triples.
    n = len(lst)
    total_triples = 0
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                total_triples += 1
    return len(sums) == total_triples

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(f"Valid: {_is_valid(result)}")
