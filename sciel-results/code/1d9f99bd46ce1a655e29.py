import time
import random
from math import gcd
from functools import reduce
from fractions import Fraction

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    def check_validity(S):
        if len(S) <= 1:
            return True
        S_list = list(S)
        for i, x in enumerate(S_list):
            others = [y for y in S_list if y != x]
            if not others:
                continue
            L = reduce(_lcm, others, 1)
            target = L // x
            if target == 0:
                continue
            vals = [L // y for y in others]
            n = len(vals)
            half = n // 2
            left = vals[:half]
            right = vals[half:]
            left_sums = set()
            for mask in range(1 << len(left)):
                s = 0
                for j in range(len(left)):
                    if mask & (1 << j):
                        s += left[j]
                left_sums.add(s)
            right_sums = set()
            for mask in range(1 << len(right)):
                s = 0
                for j in range(len(right)):
                    if mask & (1 << j):
                        s += right[j]
                right_sums.add(s)
            for ls in left_sums:
                if (target - ls) in right_sums:
                    return False
        return True

    def greedy_fast_build(candidates):
        """
        Builds a valid set from candidates using the monotonicity property.
        Candidates must be processed in decreasing order for the fast check to be valid.
        """
        # Ensure we process in decreasing order
        sorted_cands = sorted(candidates, reverse=True)
        S = []
        sums = {Fraction(0, 1)}
        
        for x in sorted_cands:
            if time.time() > deadline:
                break
            target = Fraction(1, x)
            if target not in sums:
                S.append(x)
                # Update sums
                # Optimization: if sums becomes too large, we might slow down.
                # But for N=24, max size is 2^24 which is too big for Python set of Fractions in a tight loop if we do it many times.
                # However, we only add if target is NOT in sums.
                # The number of distinct sums grows, but hopefully not to 2^k for k=24.
                # If it does, we might time out. 
                # To be safe, we can limit the size of sums or fall back to check_validity if it gets too big.
                # But for now, let's try.
                new_sums = set()
                for s in sums:
                    new_sums.add(s + target)
                sums.update(new_sums)
                
        return S

    best = list(range(13, 25))
    best_size = len(best)
    
    # 1. Run multiple fast greedy builds with randomized exclusions
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Create a random subset of candidates to explore different structures
        # We always include the largest few? Or just random mask?
        # Let's random mask with probability 0.1 to exclude
        candidates = [x for x in range(1, 25) if random.random() > 0.1]
        
        # Ensure we have at least some candidates
        if not candidates:
            candidates = [24]
            
        S = greedy_fast_build(candidates)
        
        if len(S) > best_size:
            best = S
            best_size = len(S)
            
    # 2. Local search on the best set found
    # Try to add elements
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        candidates = [x for x in range(1, 25) if x not in best]
        random.shuffle(candidates)
        for x in candidates:
            new_S = best + [x]
            if check_validity(new_S):
                best = new_S
                improved = True
                break
        if not improved:
            break
            
    # 3. Try swap local search
    # Remove one element, try to add two? Or just swap to allow a better configuration?
    # Given time, simple add is often sufficient if greedy was good.
    
    return best
