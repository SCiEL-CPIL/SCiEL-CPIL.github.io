import time
import random

def _has_4ap_in_set(s):
    """Check if set s contains a nonconstant 4-term AP."""
    lst = sorted(s)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            d = b - a
            if d == 0:
                continue
            c = b + d
            if c in s:
                e = c + d
                if e in s:
                    return True
    return False

def _is_valid_insertion(x, s_set, s_list_sorted):
    """Check if adding x to s_set creates a 4-AP.
    x can be in position 0,1,2,3 of the 4-AP."""
    # x as term 0: x, x+d, x+2d, x+3d
    for d in range(1, (512 - x) // 3 + 1):
        if (x + d) in s_set and (x + 2*d) in s_set and (x + 3*d) in s_set:
            return False
    # x as term 1: x-d, x, x+d, x+2d
    for d in range(1, x):
        if (x - d) in s_set and (x + d) in s_set and (x + 2*d) in s_set:
            return False
    # x as term 2: x-2d, x-d, x, x+d
    for d in range(1, x // 2 + 1):
        if (x - 2*d) in s_set and (x - d) in s_set and (x + d) in s_set:
            return False
    # x as term 3: x-3d, x-2d, x-d, x
    for d in range(1, x // 3 + 1):
        if (x - 3*d) in s_set and (x - 2*d) in s_set and (x - d) in s_set:
            return False
    return True

def _greedy_construct(order):
    """Greedy: add elements in given order if valid."""
    s_set = set()
    for x in order:
        if _is_valid_insertion(x, s_set, None):
            s_set.add(x)
    return s_set

def _local_search_improve(s_set, max_time=50.0):
    """Improve via swaps: remove one, try to add others."""
    start_time = time.time()
    best = s_set
    best_size = len(s_set)
    
    while time.time() - start_time < max_time:
        improved = False
        s_list = sorted(s_set)
        n = len(s_list)
        
        for i in range(n):
            if time.time() - start_time > max_time:
                break
            removed = s_list[i]
            trial = s_set - {removed}
            
            # Try adding each element not in trial
            for x in range(1, 513):
                if x in trial:
                    continue
                if _is_valid_insertion(x, trial, None):
                    new_set = trial | {x}
                    if len(new_set) > best_size:
                        best = new_set
                        best_size = len(new_set)
                        s_set = new_set
                        improved = True
                        break  # restart with new set
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _randomized_greedy(order_seed, time_limit=30.0):
    """Randomized greedy with multiple orderings."""
    start_time = time.time()
    best_set = set()
    best_size = 0
    
    rng = random.Random(order_seed)
    
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Create a random ordering with some structure
        base = list(range(1, 513))
        # Mix in some bias
        if rng.random() < 0.5:
            rng.shuffle(base)
        else:
            # Sort by some modular property
            mod = rng.randint(2, 10)
            base.sort(key=lambda x: (x % mod, x))
            # Partial shuffle
            for _ in range(rng.randint(10, 50)):
                i = rng.randint(0, 511)
                j = rng.randint(0, 511)
                base[i], base[j] = base[j], base[i]
        
        s_set = _greedy_construct(base)
        if len(s_set) > best_size:
            best_set = s_set
            best_size = len(s_set)
    
    return best_set

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    # Phase 1: Multiple greedy passes with different orderings
    best_set = set()
    best_size = 0
    
    # Standard ascending greedy
    s1 = _greedy_construct(range(1, 513))
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
    
    # Descending greedy
    s2 = _greedy_construct(range(512, 0, -1))
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
    
    # Randomized greedy
    if time.time() < deadline:
        s3 = _randomized_greedy(42, time_limit=20.0)
        if len(s3) > best_size:
            best_set = s3
            best_size = len(s3)
    
    # Phase 2: Local search improvement
    if time.time() < deadline:
        remaining = deadline - time.time() - 5.0  # 5s safety
        if remaining > 5:
            best_set = _local_search_improve(best_set, max_time=remaining)
    
    # Ensure nonempty
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
