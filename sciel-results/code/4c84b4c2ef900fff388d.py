import time
from math import gcd
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # 255 = 3 * 5 * 17
    # All elements must share a factor with 255, so divisible by 3, 5, or 17
    
    # Start with multiples of 3 (size 85)
    S3 = [x for x in range(1, 256) if x % 3 == 0]
    best = set(S3)
    best_size = len(best)
    
    # Try to see if we can add any element not in S3
    # An element x not in S3 is divisible by 5 or 17 but not 3
    # For x to join, it must share gcd > 1 with every element in the set
    # Since x is not divisible by 3, it can only share 5 or 17
    # So every element in the set must be divisible by 5 or 17 (or both)
    # This means we'd need to remove all elements divisible by 3 but not by 5 or 17
    
    # Let's check: how many elements are divisible by 3 but NOT by 5 or 17?
    # These are the ones we'd need to remove to add any non-3-multiple
    removable = [x for x in S3 if x % 5 != 0 and x % 17 != 0]
    
    # Elements divisible by 5 or 17 but not 3 (potential additions)
    additions = [x for x in range(1, 256) if x % 3 != 0 and (x % 5 == 0 or x % 17 == 0)]
    
    # If we remove `removable` and add `additions`, what's the new size?
    # But we also need all pairs in the new set to have gcd > 1
    # The new set would be: (S3 - removable) ∪ additions
    # S3 - removable = elements divisible by 3 AND (divisible by 5 or 17)
    # = multiples of 15 or 51
    
    new_core = [x for x in S3 if x % 5 == 0 or x % 17 == 0]
    candidate_set = set(new_core) | set(additions)
    
    # Check if candidate_set is valid
    def is_valid(s):
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if gcd(lst[i], lst[j]) == 1:
                    return False
        return True
    
    if is_valid(candidate_set) and len(candidate_set) > best_size:
        best = candidate_set
        best_size = len(best)
    
    # Also try: just multiples of 5
    S5 = [x for x in range(1, 256) if x % 5 == 0]
    if len(S5) > best_size:
        best = set(S5)
        best_size = len(best)
    
    # Try simulated annealing for a few iterations
    if time.time() < deadline - 5:
        random.seed(123)
        current = set(best)
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Random move: add or remove an element
            if random.random() < 0.5:
                # Try adding
                not_in = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0 and x not in current]
                if not not_in:
                    continue
                x = random.choice(not_in)
                conflicts = [y for y in current if gcd(x, y) == 1]
                if len(conflicts) == 0:
                    new_set = current | {x}
                    if len(new_set) > len(best):
                        best = new_set
                        current = new_set
            else:
                # Try removing and re-adding
                if len(current) > 1:
                    y = random.choice(list(current))
                    if y == 255:
                        continue
                    new_set = current - {y}
                    # Try to add something
                    not_in = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0 and x not in new_set]
                    if not_in:
                        x = random.choice(not_in)
                        if all(gcd(x, z) > 1 for z in new_set):
                            new_set.add(x)
                            if len(new_set) > len(best):
                                best = new_set
                                current = new_set
    
    return list(best)
