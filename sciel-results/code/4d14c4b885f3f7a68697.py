import time
import random
from itertools import combinations
from collections import defaultdict

def _is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0: return False
    return True

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

def _can_add(S, x):
    """
    Check if x can be added to S.
    S is a set of ints. x is an int not in S.
    """
    s_list = list(S)
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

def _greedy_construct(order, deadline=None):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S = set()
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(S, all_nums, deadline, max_iterations=5000):
    """
    Local search to improve set S.
    Tries to add elements and perform swaps.
    """
    current_set = S.copy()
    current_size = len(current_set)
    non_members = [x for x in all_nums if x not in current_set]
    best_local = current_set.copy()
    best_local_size = current_size
    rng = random.Random(12345)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        if not non_members:
            break
        
        # Try to add a random non-member
        if rng.random() < 0.4:
            idx = rng.randint(0, len(non_members) - 1)
            x = non_members[idx]
            if _can_add(current_set, x):
                current_set.add(x)
                current_size += 1
                non_members.pop(idx)
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local = current_set.copy()
        else:
            # Try a swap: remove y from current_set, add x from non_members
            if len(current_set) > 0:
                y = rng.choice(list(current_set))
                idx = rng.randint(0, len(non_members) - 1)
                x = non_members[idx]
                
                # Check if x can be added to S \ {y}
                temp_set = current_set - {y}
                if _can_add(temp_set, x):
                    # Swap accepted
                    current_set = temp_set | {x}
                    non_members.remove(x)
                    non_members.append(y)
                    # Size unchanged, but structure changed
                    # Check if this allows future improvements
                    # We don't update best_local here since size is same
    
    return best_local

def _polish(S, all_nums, deadline):
    """
    Systematic polish: try every possible addition, then every 1-swap.
    Repeat until no improvement or time runs out.
    """
    current_set = S.copy()
    current_size = len(current_set)
    best_set = current_set.copy()
    best_size = current_size
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        non_members = [x for x in all_nums if x not in current_set]
        
        # Phase A: try all additions
        for x in non_members:
            if time.time() > deadline:
                break
            if _can_add(current_set, x):
                current_set.add(x)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
        
        # Phase B: try all 1-swaps (remove one, add one)
        if time.time() < deadline:
            non_members = [x for x in all_nums if x not in current_set]
            members = list(current_set)
            for y in members:
                if time.time() > deadline:
                    break
                temp_set = current_set - {y}
                for x in non_members:
                    if time.time() > deadline:
                        break
                    if _can_add(temp_set, x):
                        # Accept swap if it enables a future addition
                        new_set = temp_set | {x}
                        # Check if we can add anything after this swap
                        can_add_more = False
                        for z in non_members:
                            if z == x:
                                continue
                            if _can_add(new_set, z):
                                can_add_more = True
                                break
                        if can_add_more:
                            current_set = new_set
                            improved = True
                            break
                if time.time() > deadline:
                    break
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set.copy()
    
    return best_set

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    start_time = time.time()
    
    # Pre-compute all numbers
    all_nums = list(range(2, 257))
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple Greedy Constructions with different orders
    # Try deterministic orders first
    orders = [
        list(range(2, 257)),           # Ascending
        list(range(256, 1, -1)),       # Descending
    ]
    
    # Prime-heavy orders
    primes = [x for x in all_nums if _is_prime(x)]
    composites = [x for x in all_nums if not _is_prime(x)]
    orders.append(primes + composites)
    orders.append(composites + primes)
    
    # Add random orders
    random_seed = 42
    rng = random.Random(random_seed)
    for _ in range(30):
        shuffled = all_nums.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)
        
    # Try all greedy constructions
    for order in orders:
        if time.time() > deadline - 15.0:  # Leave time for local search
            break
            
        S = _greedy_construct(order, deadline=deadline - 15.0)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Phase 2: Systematic polish on the best set
    if time.time() < deadline - 5.0:
        polished = _polish(best_set, all_nums, deadline - 5.0)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished
    
    # Phase 3: Local search from multiple starting points
    if time.time() < deadline - 2.0:
        for attempt in range(8):
            if time.time() > deadline - 2.0:
                break
            shuffled = all_nums.copy()
            rng.shuffle(shuffled)
            S = _greedy_construct(shuffled, deadline=deadline - 2.0)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
            # Try local search on this set
            if time.time() < deadline - 1.0:
                improved = _local_search(S, all_nums, deadline - 1.0, max_iterations=3000)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved
    
    # Phase 4: Final polish on the best set found
    if time.time() < deadline - 0.5:
        final_polished = _polish(best_set, all_nums, deadline - 0.5)
        if len(final_polished) > best_size:
            best_size = len(final_polished)
            best_set = final_polished
    
    # Final validation
    if not _is_valid(best_set):
        # Fallback: return primes (always valid)
        best_set = set(x for x in all_nums if _is_prime(x))
    
    return list(best_set)
