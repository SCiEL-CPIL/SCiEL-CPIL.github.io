import time
import random
from math import gcd
from functools import reduce

# Precomputation
N = 24
# LCM of 1..24
def lcm(a, b):
    return a * b // gcd(a, b)

L = reduce(lcm, range(1, N + 1), 1)
A = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = L // i

def is_subset_sum_possible(target, others):
    # Check if target can be formed by a subset of others
    # others is a list of integers
    if target == 0:
        return True
    if not others:
        return False
    
    # Optimization: if sum(others) < target, impossible
    if sum(others) < target:
        return False
    
    # Use a set of reachable sums
    # Since numbers are positive, we can prune sums > target
    sums = {0}
    for b in others:
        if b > target:
            continue
        new_sums = set()
        for s in sums:
            ns = s + b
            if ns == target:
                return True
            if ns < target:
                new_sums.add(ns)
        sums.update(new_sums)
        # Optional: if target not in sums and we've processed all, return False
    return target in sums

def is_valid(S):
    # S is a list of elements from 1..24
    S_set = set(S)
    for x in S:
        target = A[x]
        others = [A[t] for t in S_set if t != x]
        if is_subset_sum_possible(target, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Initialize with a greedy set
    # Heuristic: Try to add elements. 
    # Start with empty, add elements in random order? 
    # Or start with a known good pattern?
    # Let's start with a random set of size 10 and grow.
    
    # Greedy initialization
    current_set = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for c in candidates:
        new_set = current_set.union([c])
        if is_valid(list(new_set)):
            current_set = new_set
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Local Search
    # Move types: Add, Remove, Swap
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.5 and len(current_set) < N:
            # Try to add
            avail = [c for c in range(1, N + 1) if c not in current_set]
            if avail:
                c = random.choice(avail)
                new_set = current_set.union([c])
                if is_valid(list(new_set)):
                    current_set = new_set
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
        elif move_type < 0.7 and len(current_set) > 0:
            # Try to remove
            c = random.choice(list(current_set))
            new_set = current_set - {c}
            # Removal always preserves validity if current is valid? 
            # Yes, removing an element cannot create a new subsum relation for remaining elements.
            # Wait. The condition is "no reciprocal of a selected element...".
            # If we remove x, we don't need to check x anymore.
            # For remaining y, the set of "others" shrinks.
            # If 1/y was NOT a sum of others before, it still is NOT a sum of a subset of others (since the set of others is smaller).
            # So validity is preserved.
            current_set = new_set
        elif move_type < 0.9 and len(current_set) > 0:
            # Try to swap: remove one, add one
            c_remove = random.choice(list(current_set))
            avail = [c for c in range(1, N + 1) if c not in current_set]
            if avail:
                c_add = random.choice(avail)
                new_set = (current_set - {c_remove}).union([c_add])
                if is_valid(list(new_set)):
                    current_set = new_set
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
        else:
            # Try 2-for-1 or 1-for-2 swaps to escape local minima
            # 1-for-2: remove 1, add 2
            if len(current_set) > 0:
                c_remove = random.choice(list(current_set))
                avail = [c for c in range(1, N + 1) if c not in current_set]
                if len(avail) >= 2:
                    c_add1, c_add2 = random.sample(avail, 2)
                    new_set = (current_set - {c_remove}).union([c_add1, c_add2])
                    if is_valid(list(new_set)):
                        current_set = new_set
                        if len(current_set) > best_size:
                            best_set = set(current_set)
                            best_size = len(current_set)
            # 2-for-1: remove 2, add 1
            elif len(current_set) >= 2:
                c_remove1, c_remove2 = random.sample(list(current_set), 2)
                avail = [c for c in range(1, N + 1) if c not in current_set]
                if avail:
                    c_add = random.choice(avail)
                    new_set = (current_set - {c_remove1, c_remove2}).union([c_add])
                    if is_valid(list(new_set)):
                        current_set = new_set
                        if len(current_set) > best_size:
                            best_set = set(current_set)
                            best_size = len(current_set)

    return list(best_set)
