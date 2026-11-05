import time
import random
import sys

def construct():
    deadline = time.time() + 145
    seed_elements = [1, 2, 5, 11]
    max_val = 284
    
    def compute_sums_bitset(a):
        bits = 0
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                bits |= 1 << (a[i] + a[j])
        return bits

    def can_add(x, a, sums_bits):
        if (sums_bits >> (2 * x)) & 1:
            return False
        for y in a:
            s = x + y
            if (sums_bits >> s) & 1:
                return False
        return True

    def greedy_extend(a, sums_bits):
        a = list(a)
        sums_bits = sums_bits
        for x in range(1, max_val + 1):
            if x in a:
                continue
            if can_add(x, a, sums_bits):
                a.append(x)
                a.sort()
                sums_bits |= 1 << (2 * x)
                for y in a:
                    if y != x:
                        sums_bits |= 1 << (x + y)
        return a, sums_bits

    def backtrack(a, sums_bits, target_size, deadline):
        if len(a) == target_size:
            return a
        if time.time() > deadline:
            return None
        
        # Heuristic: try to add elements that leave room for others
        # We can use a simple greedy choice with randomization
        candidates = []
        for x in range(1, max_val + 1):
            if x not in a:
                if can_add(x, a, sums_bits):
                    candidates.append(x)
        
        if not candidates:
            return None
        
        random.shuffle(candidates)
        
        # Try a few candidates
        for x in candidates[:10]:
            new_a = a + [x]
            new_a.sort()
            new_sums = sums_bits | (1 << (2 * x))
            for y in a:
                new_sums |= 1 << (x + y)
            
            res = backtrack(new_a, new_sums, target_size, deadline)
            if res:
                return res
        return None

    best_a = list(seed_elements)
    best_bits = compute_sums_bitset(best_a)
    best_a, best_bits = greedy_extend(best_a, best_bits)
    
    random.seed(42)
    
    # Try to reach size 20
    found_20 = None
    for i in range(50):
        if time.time() > deadline - 10:
            break
        a_try = list(seed_elements)
        bits_try = compute_sums_bitset(a_try)
        
        # Randomized greedy to get a base
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        for x in candidates:
            if x in a_try:
                continue
            if can_add(x, a_try, bits_try):
                a_try.append(x)
                a_try.sort()
                bits_try |= 1 << (2 * x)
                for y in a_try:
                    if y != x:
                        bits_try |= 1 << (x + y)
                if len(a_try) >= 18:
                    break
                    
        # Backtrack to fill to 20
        if len(a_try) < 20:
            res = backtrack(a_try, bits_try, 20, deadline - 5)
            if res:
                found_20 = res
                break
                
    if found_20:
        final_a = sorted(found_20)
    else:
        # Fallback to best greedy result (likely 18 or 19)
        final_a = sorted(best_a)
        
    # Verification
    assert all(x in final_a for x in seed_elements)
    assert all(1 <= x <= 284 for x in final_a)
    assert len(set(final_a)) == len(final_a)
    
    sums = set()
    valid = True
    for i in range(len(final_a)):
        for j in range(i, len(final_a)):
            s = final_a[i] + final_a[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    assert valid
    
    return final_a
