import time
from math import gcd
from functools import lru_cache

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute all bad triples
    # A triple (a,b,c) with a<b<c is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == lab and _lcm(b, c) == lab:
                    bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    T = len(bad_triples)
    
    # Precompute for each element, the list of triple indices it appears in
    elem_to_triples = [[] for _ in range(n)]
    for ti, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(ti)
        elem_to_triples[b].append(ti)
        elem_to_triples[c].append(ti)
    
    # For each triple, store the three elements as a bitmask
    triple_mask = [0] * T
    for ti, (a, b, c) in enumerate(bad_triples):
        triple_mask[ti] = (1 << a) | (1 << b) | (1 << c)
    
    # Precompute for each element, which other elements it's in a triple with
    # This helps with faster swap validation
    # For a swap (remove r, add a), we need to check:
    # 1. After removing r, is the mask still valid? (it was valid before, removing can't create violations)
    # 2. After adding a to (mask without r), is it valid?
    #    Only triples containing a AND (mask without r) matter
    #    A triple (a, x, y) is violated if x,y are both in (mask without r)
    #    So we need: for all triples containing a, not both other elements are in mask_without_r
    
    def can_add_to_mask(mask, elem):
        """Check if adding elem to mask keeps it valid. mask must be valid."""
        bit = 1 << elem
        for ti in elem_to_triples[elem]:
            tm = triple_mask[ti]
            # Check if the other two elements of this triple are both in mask
            if (mask & tm) == (tm & ~bit):
                return False
        return True
    
    def is_valid(mask):
        """Check if mask represents a valid set"""
        for ti in range(T):
            if (mask & triple_mask[ti]) == triple_mask[ti]:
                return False
        return True
    
    def count_violations(mask):
        """Count how many triples are fully contained in mask"""
        cnt = 0
        for ti in range(T):
            if (mask & triple_mask[ti]) == triple_mask[ti]:
                cnt += 1
        return cnt
    
    # Phase 1: Greedy construction with multiple orderings
    import random
    rng = random.Random(42)
    
    best_mask = 0
    best_size = 0
    
    # Multiple restarts with different strategies
    strategies = [
        list(range(n)),                          # natural order
        sorted(range(n), key=lambda x: len(elem_to_triples[x])),  # low degree first
        sorted(range(n), key=lambda x: -len(elem_to_triples[x])), # high degree first
    ]
    
    for _ in range(50):
        strategies.append(rng.sample(range(n), n))
    
    for order in strategies:
        if time.time() > deadline:
            break
        mask = 0
        for elem in order:
            if can_add_to_mask(mask, elem):
                mask |= (1 << elem)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Enhanced local search
    def local_search(mask, max_iters=10000):
        current = mask
        current_size = bin(current).count('1')
        improved = True
        iters = 0
        no_improve_count = 0
        max_no_improve = 50
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try 1-for-0: add a removed element
            for elem in range(n):
                if not (current & (1 << elem)):
                    if can_add_to_mask(current, elem):
                        current |= (1 << elem)
                        current_size += 1
                        improved = True
                        no_improve_count = 0
                        break
            
            # Try 1-for-1 swap: remove one present, add one absent
            if not improved:
                present = [e for e in range(n) if current & (1 << e)]
                absent = [e for e in range(n) if not (current & (1 << e))]
                
                # Try each swap, pick the first valid one
                # But to be smarter, try swaps that remove high-degree elements first
                present_sorted = sorted(present, key=lambda x: len(elem_to_triples[x]), reverse=True)
                
                found_swap = False
                for rem in present_sorted:
                    if time.time() > deadline:
                        return current
                    rem_bit = 1 << rem
                    base_mask = current & ~rem_bit
                    for add in absent:
                        if can_add_to_mask(base_mask, add):
                            current = base_mask | (1 << add)
                            # size stays the same
                            improved = True
                            found_swap = True
                            no_improve_count = 0
                            break
                    if found_swap:
                        break
            
            # Try 2-for-1: remove two present, add one absent (net -1, but might enable more adds)
            # Actually, let's try 2-for-2: remove two, add two (net 0) or 1-for-2 (net +1)
            # 1-for-2 is most valuable: remove one present, add two absent
            
            if not improved:
                present = [e for e in range(n) if current & (1 << e)]
                absent = [e for e in range(n) if not (current & (1 << e))]
                
                # Try 1-for-2: for each present rem, try pairs of absent adds
                # This is expensive, so limit it
                if len(absent) >= 2 and len(present) <= 60:
                    present_sorted = sorted(present, key=lambda x: len(elem_to_triples[x]), reverse=True)
                    found_1for2 = False
                    for rem in present_sorted[:30]:  # limit to top 30
                        if time.time() > deadline:
                            return current
                        rem_bit = 1 << rem
                        base_mask = current & ~rem_bit
                        # Try pairs of absent elements
                        for i in range(len(absent)):
                            if time.time() > deadline:
                                return current
                            a1 = absent[i]
                            if not can_add_to_mask(base_mask, a1):
                                continue
                            mid_mask = base_mask | (1 << a1)
                            for j in range(i + 1, len(absent)):
                                a2 = absent[j]
                                if can_add_to_mask(mid_mask, a2):
                                    current = mid_mask | (1 << a2)
                                    current_size += 1
                                    improved = True
                                    found_1for2 = True
                                    no_improve_count = 0
                                    break
                            if found_1for2:
                                break
                        if found_1for2:
                            break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        return current
    
    best_mask = local_search(best_mask)
    best_size = bin(best_mask).count('1')
    
    # Phase 3: More restarts with local search
    for restart in range(200):
        if time.time() > deadline:
            break
        # Random subset as starting point
        mask = 0
        for e in range(n):
            if rng.random() < 0.5:
                mask |= (1 << e)
        # Fix: remove elements to make valid
        # Greedy removal
        while True:
            if time.time() > deadline:
                break
            # Find a violating triple
            viol = -1
            for ti in range(T):
                if (mask & triple_mask[ti]) == triple_mask[ti]:
                    viol = ti
                    break
            if viol == -1:
                break
            # Remove one element from this triple
            a, b, c = bad_triples[viol]
            # Remove the one with highest degree
            degs = [(len(elem_to_triples[a]), a), (len(elem_to_triples[b]), b), (len(elem_to_triples[c]), c)]
            degs.sort(reverse=True)
            mask &= ~(1 << degs[0][1])
        
        sz = bin(mask).count('1')
        if sz > best_size:
            best_mask = local_search(mask)
            best_size = bin(best_mask).count('1')
    
    # Final polish: try one more local search on the best
    if time.time() < deadline - 5:
        polished = local_search(best_mask, max_iters=20000)
        if bin(polished).count('1') > best_size:
            best_mask = polished
            best_size = bin(best_mask).count('1')
    
    # Convert to list
    result = [e + 1 for e in range(n) if best_mask & (1 << e)]
    if not result:
        result = [1]
    return result
