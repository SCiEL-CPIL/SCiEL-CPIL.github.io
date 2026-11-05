import time
import random
from itertools import combinations

# Global deadline
START_TIME = time.time()
TIME_LIMIT = 55.0  # Safety margin for 60s limit

def can_add(S_list, x):
    """
    Check if x can be added to valid set S_list.
    Condition:
    1. x does not divide sum of any 3 distinct elements in S_list.
    2. No a in S_list divides sum of x and any 2 distinct elements in S_list \ {a}.
    """
    n = len(S_list)
    
    # Check 1: x divides sum of 3 distinct in S_list
    if n >= 3:
        for b, c, d in combinations(S_list, 3):
            if (b + c + d) % x == 0:
                return False

    # Check 2: a in S_list divides sum of x and 2 distinct in S_list \ {a}
    if n >= 3:
        for i, a in enumerate(S_list):
            # Create list of others
            others = [S_list[j] for j in range(n) if j != i]
            for b, c in combinations(others, 2):
                if (x + b + c) % a == 0:
                    return False

    return True

def greedy_construct(numbers, order_type="random"):
    """
    Perform a single greedy construction pass.
    order_type: 'random', 'desc', 'asc'
    """
    current_list = []
    candidates = list(numbers)
    
    if order_type == 'desc':
        candidates.sort(reverse=True)
    elif order_type == 'asc':
        candidates.sort()
    else:
        random.shuffle(candidates)
        
    for x in candidates:
        if can_add(current_list, x):
            current_list.append(x)
            
    return current_list

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_set = set()
    best_size = 0
    
    numbers = list(range(1, 193))
    
    # Try a few deterministic orders first to get a good baseline quickly
    # Descending is often good because large numbers are less likely to divide sums
    for order in ['desc', 'asc', 'random']:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        current_list = greedy_construct(numbers, order)
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_set = set(current_list)
            
    # Then do many random restarts
    while time.time() - START_TIME < TIME_LIMIT:
        current_list = greedy_construct(numbers, 'random')
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_set = set(current_list)
            
    return list(best_set)
