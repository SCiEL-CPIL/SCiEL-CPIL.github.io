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
        random.shuffle(all_nums)
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
    
    # Phase 3: Restart and Polish with perturbations
    while time.time() < deadline - 5.0:
        # Create a perturbed version of the best set
        # Remove k random elements
        k = random.randint(1, 5)
        perturbed = best_set.copy()
        if len(perturbed) > k:
            to_remove = random.sample(list(perturbed), k)
            for x in to_remove:
                perturbed.remove(x)
            
        # Polish the perturbed set
        perturbed = _polish(perturbed, deadline)
        
        if len(perturbed) > best_size:
            best_size = len(perturbed)
            best_set = perturbed.copy()
            
        # If we found a better set, continue. If not, we might be stuck.
        # To avoid getting stuck, we can also try random restarts from scratch.
        if random.random() < 0.2:
            # Random restart
            random.shuffle(all_nums)
            S = _greedy_construct(all_nums)
            S = _polish(S, deadline)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()

    # Final polish
    best_set = _polish(best_set, deadline)
    best_size = len(best_set)
    
    # Ensure valid Python int types
    result = [int(x) for x in best_set]
    
    return result
