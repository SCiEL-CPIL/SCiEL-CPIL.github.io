import time
import random

_P0 = time.time()
_DEADLINE = _P0 + 80.0

def _subset_products_set(nums):
    """Compute all subset products of nums."""
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods

def _is_valid_set(nums):
    """Check if all 2^k subset products are distinct."""
    prods = {1}
    for x in nums:
        new_prods = set(prods)
        for p in prods:
            new_prods.add(p * x)
        if len(new_prods) != 2 * len(prods):
            return False
        prods = new_prods
    return True

def _greedy_build(candidates, deadline):
    """Greedy build: add candidates in order if valid."""
    selected = []
    prods = {1}
    for x in candidates:
        if time.time() > deadline:
            break
        # Check if x can be added
        can_add = True
        for p in prods:
            if (p * x) in prods:
                can_add = False
                break
        if can_add:
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            selected.append(x)
            prods = new_prods
    return selected

def construct():
    deadline = _DEADLINE
    
    # Strategy: Use pure powers of primes with superincreasing exponents
    # Prime 2: 2^1, 2^2, 2^4, 2^8, 2^16 = 2, 4, 16, 256... wait 2^8=256 > 40
    # Let me recalculate: powers of 2 in [2,40]: 2,4,8,16,32
    # Exponents: 1,2,3,4,5 - NOT superincreasing!
    # For superincreasing: 1,2,4,8,16 -> values: 2,4,16,256,65536
    # 256 > 40, so only 2,4,16 fit with superincreasing exponents 1,2,4
    
    # Actually, let me think more carefully.
    # We need the exponent vectors to be linearly independent in the sense that
    # no two distinct subsets have the same sum of exponent vectors.
    # For a single prime, we need the exponents to form a set with distinct subset sums.
    # A sufficient condition is superincreasing, but not necessary.
    # For {1,2,3,4,5}: subset sums range from 0 to 15, but are they all distinct?
    # 1+4=5, and 5 is in the set. So {1,4,5} and {5} both give sum 5? No, {1,4} gives 5, and {5} gives 5. Collision!
    # So {1,2,3,4,5} does NOT have distinct subset sums.
    
    # What about {1,2,4}? Subset sums: 0,1,2,3,4,5,6,7 - all distinct. Good.
    # {1,2,4,8}? Subset sums: 0..15 - all distinct. Good.
    # So for prime 2: use exponents {1,2,4} -> values {2,4,16}
    # For prime 3: exponents {1,2,4} -> values {3,9,27}
    # For prime 5: exponents {1,2} -> values {5,25}
    # For prime 7: exponent {1} -> value {7}
    # Other primes: just themselves
    
    # Candidate set:
    candidate_18 = [2, 4, 16, 3, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # That's 17 elements. Can we add one more?
    # What about 8? 8=2^3. If we add 8 to {2,4,16}, exponents become {1,2,3,4}.
    # Subset sums of {1,2,3,4}: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 - but 1+4=5 and 5 is in set? 
    # Wait, 5 is not in {1,2,3,4}. Let me check: can two different subsets of {1,2,3,4} have same sum?
    # 1+4=5, 2+3=5. Yes! Collision. So {1,2,3,4} does NOT have distinct subset sums.
    
    # What about adding 32=2^5 to {2,4,16}? Exponents {1,2,4,5}.
    # 1+4=5, and 5 is in the set. Collision: {1,4} and {5} both give 5.
    
    # So we can't easily add more powers of 2.
    
    # Let's verify the 17-element candidate:
    if _is_valid_set(candidate_18):
        return candidate_18
    
    # Fallback: try greedy on all numbers
    all_nums = list(range(2, 41))
    best_set = _greedy_build(all_nums, deadline)
    
    # Try to improve with local search
    if len(best_set) < 18 and time.time() < deadline:
        rng = random.Random(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Try swapping: remove one, add another
            if best_set:
                idx = rng.randrange(len(best_set))
                removed = best_set[idx]
                temp = best_set[:idx] + best_set[idx+1:]
                # Try adding a number not in temp
                available = [x for x in range(2, 41) if x not in temp]
                if available:
                    add_x = rng.choice(available)
                    trial = temp + [add_x]
                    if _is_valid_set(trial):
                        best_set = trial
    
    return best_set
