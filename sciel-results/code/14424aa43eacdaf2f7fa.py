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

def _greedy_construct(order):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S = set()
    # Optimization: Keep a list of elements in S for faster validation checks
    # But _is_valid is O(n^3), so we just re-validate.
    # For n~50, n^3 is 125k ops. 
    # If we do 1000 greedy runs, that's 125M ops. Might be tight in Python.
    # Let's optimize _is_valid for incremental addition?
    # Check: Does x conflict with S?
    # Conflicts for adding x to S:
    # 1. x | a*b for distinct a,b in S
    # 2. a | x*b for distinct a,b in S (where b is the other element)
    # 3. a | b*x is same as 2.
    
    # Let's write a faster check for addition.
    s_list = list(S)
    n = len(s_list)
    
    # Check 1: x divides a*b
    for i in range(n):
        a = s_list[i]
        for j in range(i+1, n):
            b = s_list[j]
            if (a * b) % x == 0:
                return False # x is bad divisor
        
    # Check 2: a divides x*b (for all a in S, b in S, a!=b)
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if i == j:
                continue
            b = s_list[j]
            if (x * b) % a == 0:
                return False # a is bad divisor of x*b
            
    # Check 3: b divides x*a (symmetric to Check 2, covered)
    
    return True

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
            
        S = set()
        for x in order:
            # Check if x can be added to S
            # Use the fast check
            if _can_add(S, x):
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            # print(f"New best size: {best_size}")

    # Phase 2: Local Search on the best set
    # We try to improve best_set using swaps and adds.
    current_set = best_set.copy()
    current_size = best_size
    
    # Precompute non-members for faster selection
    non_members = [x for x in all_nums if x not in current_set]
    
    # Local search loop
    while time.time() < deadline - 2.0:
        if not non_members:
            break
            
        # Try random Add
        if random.random() < 0.5:
            # Try to add a random non-member
            idx = random.randint(0, len(non_members) - 1)
            x = non_members[idx]
            if _can_add(current_set, x):
                current_set.add(x)
                current_size += 1
                non_members.pop(idx)
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
        else:
            # Try a Swap: remove random y from current_set, add random x from non_members
            # Actually, just try to add. If it fails, try to remove one element to make room?
            # A simpler swap: pick y in current_set, pick x in non_members.
            # If (current_set - {y}) U {x} is valid, swap.
            
            # To make this efficient, we don't re-validate the whole set.
            # We check if x can be added to (current_set - {y}).
            
            if len(current_set) > 0:
                y = random.choice(list(current_set))
                idx = random.randint(0, len(non_members) - 1)
                x = non_members[idx]
                
                # Check if x can be added to S_temp = current_set - {y}
                # We need to modify _can_add to take a set and an element, 
                # but it validates against the whole set.
                # So we create a temporary set? That's O(N) copy.
                # Or we pass y as an "excluded" element?
                
                # Let's just do the full validation for the swap candidate.
                # candidate = current_set - {y}
                # candidate.add(x)
                # if _is_valid(candidate): ...
                
                # Optimization: 
                # 1. Check if x conflicts with current_set (ignoring y)
                # 2. Check if y was hiding a conflict that x exposes? No.
                # If x is compatible with (S \ {y}), then (S \ {y}) U {x} is valid 
                # IF (S \ {y}) is valid (which it is, since S is valid and removing elements preserves validity).
                
                # So we just need to check if x is compatible with S \ {y}.
                # We can reuse _can_add but we need to skip y in the loops.
                
                # Let's write a specific check for this case or just do the full check.
                # Given N is small (~50), full check is fine.
                
                candidate = current_set.copy()
                candidate.remove(y)
                candidate.add(x)
                
                if _is_valid(candidate):
                    # Swap accepted
                    current_set = candidate
                    # Update non_members
                    non_members.remove(x)
                    non_members.append(y)
                    
                    # Size is unchanged, but structure changed.
                    # This might allow future adds.
                    
    return list(best_set)

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
