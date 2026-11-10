import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # Safety margin before 60s kill

    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    
    # Precompute all pair sums for quick checks
    # For a set S, for each x in S, we need to check if 1/x = sum of subset of {1/y : y in S, y != x}
    
    def can_make_sum_mitm(target, others):
        """Meet-in-the-middle: check if target can be formed as sum of subset of others."""
        n = len(others)
        if n == 0:
            return target == 0
        if target == 0:
            return True
        
        # Quick prune
        total = sum(others)
        if total < target:
            return False
        
        # If target equals any single element, return True
        for v in others:
            if v == target:
                return True
        
        # If target is smaller than all elements, can't form it (all positive)
        if min(others) > target:
            return False
        
        # Convert to integer arithmetic via LCM for speed
        from math import gcd
        def lcm(a, b):
            return a // gcd(a, b) * b
        
        # Compute LCM of all denominators
        l = 1
        for v in others:
            l = lcm(l, v.denominator)
        
        target_int = target.numerator * (l // target.denominator)
        vals = [v.numerator * (l // v.denominator) for v in others]
        
        # Meet-in-the-middle
        mid = n // 2
        left = vals[:mid]
        right = vals[mid:]
        
        # Generate all subset sums for left
        left_sums = {0}
        for val in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + val)
            left_sums.update(new_sums)
        
        # Generate all subset sums for right and check
        right_sums = {0}
        for val in right:
            new_sums = set()
            for s in right_sums:
                new_sums.add(s + val)
            right_sums.update(new_sums)
        
        # Check if any pair sums to target
        for s in left_sums:
            if (target_int - s) in right_sums:
                return True
        return False

    def is_valid(S):
        """Check if set S is reciprocal-subsum-free."""
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return True
        
        for x in S_list:
            target = recip[x]
            others = [recip[y] for y in S_list if y != x]
            
            # Quick prune: if sum of all others < target, skip
            total_sum = sum(others)
            if total_sum < target:
                continue
            
            # Check if target can be formed from subset of others
            if can_make_sum_mitm(target, others):
                return False
        return True

    # Start with known good seed: {9, 10, ..., 24} size 16
    best_set = list(range(9, 25))
    best_size = len(best_set)
    
    # Verify seed is valid (should be, but be safe)
    if not is_valid(set(best_set)):
        # Fallback to smaller known valid set
        best_set = list(range(13, 25))
        best_size = len(best_set)
    
    elements = list(range(1, 25))
    
    # Multiple randomized greedy passes
    while time.time() < deadline:
        random.shuffle(elements)
        S = set()
        
        for x in elements:
            if time.time() > deadline:
                break
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
            # If we found a large set, try local search to improve
            if time.time() < deadline - 10:
                # Local search: try swapping elements
                current = set(best_set)
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for x in list(current):
                        if time.time() > deadline - 5:
                            break
                        for y in range(1, 25):
                            if y in current:
                                continue
                            new_S = (current - {x}) | {y}
                            if is_valid(new_S) and len(new_S) == len(current):
                                # Same size swap might help for future additions
                                pass
                            elif is_valid(new_S):
                                current = new_S
                                improved = True
                                break
                        if improved:
                            break
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)
                
                # Try adding remaining elements to improved set
                remaining = [x for x in range(1, 25) if x not in current]
                random.shuffle(remaining)
                for x in remaining:
                    if time.time() > deadline - 2:
                        break
                    new_S = current | {x}
                    if is_valid(new_S):
                        current = new_S
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = list(current)
                
                # Try adding pairs of remaining elements
                if time.time() < deadline - 3:
                    remaining2 = [x for x in range(1, 25) if x not in current]
                    random.shuffle(remaining2)
                    pair_found = True
                    while pair_found and time.time() < deadline - 2:
                        pair_found = False
                        for i in range(len(remaining2)):
                            if time.time() > deadline - 2:
                                break
                            for j in range(i + 1, len(remaining2)):
                                new_S = current | {remaining2[i], remaining2[j]}
                                if is_valid(new_S):
                                    current = new_S
                                    remaining2 = [x for x in range(1, 25) if x not in current]
                                    random.shuffle(remaining2)
                                    if len(current) > best_size:
                                        best_size = len(current)
                                        best_set = list(current)
                                    pair_found = True
                                    break
                            if pair_found:
                                break
    
    return best_set
