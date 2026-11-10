import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N = 512
    
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs
    
    def is_valid_set(s):
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                if (a * b) // gcd(a, b) > N:
                    return False
        return True
    
    # Find best divisor set
    best_set = set()
    best_size = 0
    
    for L in range(1, N + 1):
        divs = get_divisors(L)
        if len(divs) > best_size:
            best_size = len(divs)
            best_set = divs.copy()
    
    # Try to augment best_set greedily
    # Sort candidates by how "compatible" they are
    current_set = set(best_set)
    remaining = set(range(1, N + 1)) - current_set
    
    # Try adding elements in order of increasing value (smaller numbers tend to have smaller LCMs)
    for x in sorted(remaining):
        if time.time() > deadline:
            break
        # Check if x can be added
        can_add = True
        for y in current_set:
            if (x * y) // gcd(x, y) > N:
                can_add = False
                break
        if can_add:
            current_set.add(x)
    
    # Try multiple starting points: divisor sets of various L values
    # Also try local search / hill climbing
    
    # Try starting from each L with many divisors and augment
    all_candidates = []
    for L in range(1, N + 1):
        divs = get_divisors(L)
        if len(divs) >= best_size - 2:  # Only consider near-optimal starts
            all_candidates.append(divs)
    
    # For each candidate, try greedy augmentation
    for divs in all_candidates:
        if time.time() > deadline:
            break
        current = set(divs)
        remaining = set(range(1, N + 1)) - current
        
        # Try adding in different orders
        for x in sorted(remaining):
            if time.time() > deadline:
                break
            can_add = True
            for y in current:
                if (x * y) // gcd(x, y) > N:
                    can_add = False
                    break
            if can_add:
                current.add(x)
        
        if len(current) > len(current_set):
            current_set = current
    
    # Return as list
    return list(current_set)
