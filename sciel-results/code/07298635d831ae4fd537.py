import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
N = 24

# Precompute L//x for each element (index = x-1)
WEIGHTS = [L // x for x in range(1, 25)]

# Precompute subset sums for each element's "others" set
# For element at index i, others = all indices except i
# Store as sorted list for binary search
SUBSET_SUMS = []
for i in range(N):
    others = [j for j in range(N) if j != i]
    sums = {0}
    for j in others:
        w = WEIGHTS[j]
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns <= L:  # Can't exceed L (max possible sum of all reciprocals)
                new_sums.add(ns)
        sums = sums | new_sums
    SUBSET_SUMS.append(sorted(sums))

def is_valid(S):
    """Check if set S is reciprocal-subsum-free using precomputed subset sums."""
    if not S:
        return True
    S_list = list(S)
    n = len(S_list)
    if n == 0:
        return True
    
    for i, x in enumerate(S_list):
        target = WEIGHTS[x - 1]
        # Use precomputed subset sums for element x's "others" set
        # But we need to check if target is achievable using ONLY elements in S_list \ {x}
        # The precomputed sums include ALL other elements, not just those in S.
        # So we still need a per-check verification.
        # However, we can use a quick filter: if target > sum of weights in S\{x}, skip
        others_weights = [WEIGHTS[y-1] for y in S_list if y != x]
        if not others_weights:
            continue
        if sum(others_weights) < target:
            continue
        
        # Check via meet-in-the-middle on the actual subset
        m = len(others_weights)
        if m <= 1:
            if m == 1 and others_weights[0] == target:
                return False
            continue
        
        mid = m // 2
        left = others_weights[:mid]
        right = others_weights[mid:]
        
        left_sums = {0}
        for w in left:
            left_sums = left_sums | {s + w for s in left_sums}
        
        found = False
        right_sums = {0}
        for w in right:
            new_sums = {s + w for s in right_sums}
            for s in new_sums:
                if target - s in left_sums:
                    found = True
                    break
            if found:
                break
            right_sums |= new_sums
        
        if found:
            return False
    
    return True

def is_valid_fast(S, S_set):
    """Faster validity check using bitmask-based approach for small sets."""
    if not S:
        return True
    S_list = list(S)
    n = len(S_list)
    
    for i, x in enumerate(S_list):
        target = WEIGHTS[x - 1]
        others_weights = [WEIGHTS[y-1] for y in S_list if y != x]
        if not others_weights:
            continue
        if sum(others_weights) < target:
            continue
        
        m = len(others_weights)
        if m <= 1:
            if m == 1 and others_weights[0] == target:
                return False
            continue
        
        mid = m // 2
        left = others_weights[:mid]
        right = others_weights[mid:]
        
        left_sums = {0}
        for w in left:
            left_sums = left_sums | {s + w for s in left_sums}
        
        found = False
        right_sums = {0}
        for w in right:
            new_sums = {s + w for s in right_sums}
            for s in new_sums:
                if target - s in left_sums:
                    found = True
                    break
            if found:
                break
            right_sums |= new_sums
        
        if found:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    best = []
    best_size = 0
    
    # Fallback
    fallback = list(range(13, 25))
    if is_valid(fallback):
        best = fallback
        best_size = len(fallback)
    
    random.seed(12345)
    
    # Multi-start greedy with various orderings
    orderings = [
        sorted(UNIVERSE, reverse=True),  # Large first
        sorted(UNIVERSE),                # Small first
        list(reversed(UNIVERSE)),
    ]
    for _ in range(30):
        perm = UNIVERSE[:]
        random.shuffle(perm)
        orderings.append(perm)
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = S + [x]
            if is_valid(candidate):
                S = candidate
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local search: try adding any missing element
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in UNIVERSE:
            if time.time() > deadline:
                break
            if x not in best:
                candidate = best + [x]
                if is_valid(candidate):
                    best = candidate
                    best_size = len(best)
                    improved = True
    
    # Swap moves: remove one, try adding two different ones
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        S = best[:]
        for i in range(len(S)):
            if time.time() > deadline - 2:
                break
            S_no_i = S[:i] + S[i+1:]
            # Try adding pairs of elements not in S_no_i
            missing = [x for x in UNIVERSE if x not in S_no_i]
            for a_idx in range(len(missing)):
                if time.time() > deadline - 2:
                    break
                for b_idx in range(a_idx + 1, len(missing)):
                    if time.time() > deadline - 2:
                        break
                    candidate = S_no_i + [missing[a_idx], missing[b_idx]]
                    if len(candidate) > best_size and is_valid(candidate):
                        best = candidate
                        best_size = len(best)
                        S = best[:]
                        break
                if len(best) > len(S_no_i):
                    break
            if len(best) > len(S_no_i) + 1:
                break
    
    # Simulated annealing / random restarts with local optimization
    if time.time() < deadline - 10:
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            
            # Random initial set
            S = []
            perm = UNIVERSE[:]
            random.shuffle(perm)
            for x in perm:
                if time.time() > deadline - 5:
                    break
                candidate = S + [x]
                if is_valid(candidate):
                    S = candidate
            
            # Local optimization: add elements
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for x in UNIVERSE:
                    if x not in S:
                        candidate = S + [x]
                        if is_valid(candidate):
                            S = candidate
                            improved = True
            
            # Swap moves
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                S_copy = S[:]
                for i in range(len(S_copy)):
                    if time.time() > deadline - 5:
                        break
                    S_no_i = S_copy[:i] + S_copy[i+1:]
                    missing = [x for x in UNIVERSE if x not in S_no_i]
                    for a_idx in range(len(missing)):
                        if time.time() > deadline - 5:
                            break
                        for b_idx in range(a_idx + 1, len(missing)):
                            if time.time() > deadline - 5:
                                break
                            candidate = S_no_i + [missing[a_idx], missing[b_idx]]
                            if len(candidate) > len(S_copy) and is_valid(candidate):
                                S_copy = candidate
                                break
                        if len(S_copy) > len(S_no_i):
                            break
                    if len(S_copy) > len(S_no_i) + 1:
                        break
                if len(S_copy) > len(S):
                    S = S_copy
                else:
                    break
            
            if len(S) > best_size:
                best = S
                best_size = len(S)
    
    return best
