import time
import random
from itertools import combinations

def _is_valid(S):
    """
    Check if S is a valid set.
    For every three distinct members a, b, c in S, a does not divide b*c.
    """
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if i == j:
                continue
            b = s_list[j]
            for k in range(n):
                if i == k or j == k:
                    continue
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S, x, exclude=None):
    """
    Check if x can be added to S (optionally excluding element 'exclude' from consideration).
    S is a set of ints. x is an int not in S.
    """
    s_list = [a for a in S if a != exclude]
    n = len(s_list)
    
    # 1. x divides a*b for distinct a,b in S
    for i in range(n):
        a = s_list[i]
        for j in range(i+1, n):
            b = s_list[j]
            if (a * b) % x == 0:
                return False
            
    # 2. a divides x*b for distinct a,b in S
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if i == j:
                continue
            b = s_list[j]
            if (x * b) % a == 0:
                return False
                
    return True

def _greedy_construct(order):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S = set()
    for x in order:
        if _can_add(S, x):
            S.add(x)
    return S

def _polish(S, deadline):
    """
    Repeatedly try to add any element that can be added until no more can be added.
    Returns the polished set.
    """
    all_nums = list(range(2, 257))
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in all_nums:
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    changed = True
    return S

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    start_time = time.time()
    
    # Pre-compute all numbers
    all_nums = list(range(2, 257))
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple Random Greedy Constructions
    # Try a few specific orders first
    orders = [
        list(range(2, 257)),
        list(range(256, 1, -1)),
    ]
    
    # Add prime-heavy order
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0: return False
        return True
            
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    orders.append(primes + composites)
    orders.append(composites + primes)
    
    for _ in range(20):
        random.shuffle(all_nums)
        orders.append(all_nums.copy())
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S = _greedy_construct(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Phase 2: Polish the best set
    best_set = _polish(best_set, deadline)
    best_size = len(best_set)
    
    # Phase 3: Local Search with strategic swaps
    current_set = best_set.copy()
    current_size = best_size
    
    # Precompute non-members for faster selection
    non_members = [x for x in all_nums if x not in current_set]
    
    # Local search loop
    while time.time() < deadline - 2.0:
        if not non_members:
            break
        
        # First, try to add any non-member that can be added
        added_any = False
        for idx in range(len(non_members) - 1, -1, -1):
            x = non_members[idx]
            if _can_add(current_set, x):
                current_set.add(x)
                current_size += 1
                non_members.pop(idx)
                added_any = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
        
        if added_any:
            # After adding, re-polish
            current_set = _polish(current_set, deadline)
            current_size = len(current_set)
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
            non_members = [x for x in all_nums if x not in current_set]
            continue
        
        # No additions possible, try swaps
        # Strategy: For each non-member x, find which elements in current_set block it.
        # Then try swapping one blocking element with x.
        
        # Pick a random non-member to focus on
        if not non_members:
            break
        x = random.choice(non_members)
        
        # Find which elements in current_set block x
        s_list = list(current_set)
        blocking = []
        for y in s_list:
            # Check if x can be added to current_set without y
            if _can_add(current_set, x, exclude=y):
                # y is not the blocker (or there are multiple blockers)
                pass
            else:
                # y might be a blocker
                blocking.append(y)
        
        # If removing any single element allows x to be added, do the swap
        swapped = False
        for y in blocking:
            if _can_add(current_set, x, exclude=y):
                # Swap: remove y, add x
                current_set.remove(y)
                current_set.add(x)
                non_members.remove(x)
                non_members.append(y)
                swapped = True
                break
        
        if not swapped and len(current_set) > 0:
            # Try random swap as fallback
            y = random.choice(list(current_set))
            if _can_add(current_set, x, exclude=y):
                current_set.remove(y)
                current_set.add(x)
                non_members.remove(x)
                non_members.append(y)
    
    # Final polish
    best_set = _polish(best_set, deadline)
    best_size = len(best_set)
    
    # Ensure valid Python int types
    result = [int(x) for x in best_set]
    
    return result
