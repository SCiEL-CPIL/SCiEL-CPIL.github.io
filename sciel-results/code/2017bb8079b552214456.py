import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    all_numbers = list(range(2, 41))
    
    # Fast validity check using incremental subset products
    def is_valid_set(S):
        n = len(S)
        if n > 22:  # 2^22 is too large
            return False
        # Compute all subset products
        prods = [1]
        for x in S:
            # New products are old products * x
            new_prods = [p * x for p in prods]
            prods.extend(new_prods)
            if len(prods) != len(set(prods)):
                return False
        return True
    
    # Can we add x to set S?
    def can_add(S, x):
        if x in S:
            return False
        n = len(S) + 1
        if n > 22:
            return False
        # Current subset products
        prods = [1]
        for y in S:
            new_prods = [p * y for p in prods]
            prods.extend(new_prods)
        # Check if any prod * x is already in prods
        prod_set = set(prods)
        for p in prods:
            if p * x in prod_set:
                return False
        return True
    
    # Known good base set of 16
    base_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    current_best = base_set[:]
    best_size = len(current_best)
    
    # Greedy addition
    candidates = [x for x in all_numbers if x not in current_best]
    for x in sorted(candidates):
        if can_add(current_best, x):
            current_best.append(x)
            best_size = len(current_best)
    
    # Branch and bound to find larger sets
    remaining = [x for x in all_numbers if x not in current_best]
    
    def bb(S, cand_list, best_sz):
        nonlocal current_best, best_size
        if time.time() > deadline:
            return
        if len(S) + len(cand_list) <= best_sz:
            return
        if not cand_list:
            if len(S) > best_sz:
                current_best = S[:]
                best_size = len(S)
            return
        
        # Try including first candidate
        x = cand_list[0]
        rest = cand_list[1:]
        
        if can_add(S, x):
            bb(S + [x], rest, best_sz)
        
        # Try excluding
        if len(S) + len(rest) > best_sz:
            bb(S, rest, best_sz)
    
    if time.time() < deadline - 10:
        bb(current_best[:], remaining, best_size)
    
    # Final validation
    if not is_valid_set(current_best):
        return base_set[:]
    
    return current_best
