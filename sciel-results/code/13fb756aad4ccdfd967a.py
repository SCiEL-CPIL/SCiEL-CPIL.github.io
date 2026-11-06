from math import gcd
from functools import reduce
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

L = reduce(_lcm, range(1, 23), 1)

def _check_valid(S):
    """Check if all reciprocal subset sums are distinct using integer weights."""
    if not S:
        return True
    weights = [L // x for x in S]
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            # Check collision with existing sums
            if ns in sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def construct():
    # 1. Greedy construction with multiple random permutations
    best_set = []
    best_size = 0
    
    # Try a few deterministic orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        sorted(range(1, 23), key=lambda x: random.random())
    ]
    
    # Add some random permutations
    for _ in range(50):
        perm = list(range(1, 23))
        random.shuffle(perm)
        orders.append(perm)
        
    for perm in orders:
        current_set = []
        for x in perm:
            if x in current_set:
                continue
            new_set = current_set + [x]
            if _check_valid(new_set):
                current_set = new_set
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # 2. Local Search / Randomized Improvement
    # Try to increase size by 1
    # We can try to find a set of size best_size + 1 by random sampling
    # Or, we can try to modify best_set.
    
    # Let's try random sampling for larger sizes
    # If best_size is 16, try to find 17.
    # 2^17 = 131,072 subset sums. This is fast.
    # 2^18 = 262,144.
    # 2^19 = 524,288.
    # 2^20 = 1,048,576.
    
    target_size = best_size + 1
    max_attempts = 1000
    
    for _ in range(max_attempts):
        # Sample a random set of size target_size
        S = random.sample(range(1, 23), target_size)
        if _check_valid(S):
            best_set = S
            best_size = target_size
            # Try to go bigger
            target_size += 1
            if target_size > 18: # Cap to avoid slow checks
                break
            max_attempts = 1000 # Reset attempts for new size
            
    # 3. Final attempt: Try to add to best_set if possible
    # This is already covered by greedy, but maybe a different order helps?
    # The greedy approach is quite strong.
    
    return best_set
