# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    N = 4096
    
    def check_addition(S, x, sums_set):
        """Check if adding x to S preserves unique triple sums. Returns (valid, new_sums_list)."""
        new_sums = []
        # (x, x, x)
        s3 = x * 3
        new_sums.append(s3)
        # (x, x, s) for each s in S
        for s in S:
            new_sums.append(2 * x + s)
        # (x, s, t) for s <= t in S
        for i in range(len(S)):
            si = S[i]
            for j in range(i, len(S)):
                sj = S[j]
                new_sums.append(x + si + sj)
        
        # Check all new sums against existing sums_set and each other
        for s in new_sums:
            if s in sums_set:
                return False, None
        if len(new_sums) != len(set(new_sums)):
            return False, None
        return True, new_sums
    
    def greedy_search(ordering):
        """Greedy search with given ordering of candidates."""
        S = []
        sums_set = set()
        for x in ordering:
            if time.time() > deadline:
                break
            valid, new_sums = check_addition(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
        return S
    
    def try_improve(S):
        """Try to improve set S by swapping or adding elements."""
        if time.time() > deadline:
            return S
        S = sorted(S)
        sums_set = set()
        # Rebuild sums_set
        for a, b, c in combinations_with_replacement(S, 3):
            sums_set.add(a + b + c)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            # Try adding each element not in S
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                valid, new_sums = check_addition(S, x, sums_set)
                if valid:
                    S.append(x)
                    S.sort()
                    sums_set.update(new_sums)
                    S_set.add(x)
                    improved = True
                    break
            if not improved:
                # Try swapping: remove one element, try to add two
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    S_no_i = S[:i] + S[i+1:]
                    # Rebuild sums_set for S_no_i
                    sums_set_no_i = set()
                    for a, b, c in combinations_with_replacement(S_no_i, 3):
                        sums_set_no_i.add(a + b + c)
                    S_set_no_i = set(S_no_i)
                    # Try adding two elements
                    for x1 in range(1, N + 1):
                        if x1 in S_set_no_i:
                            continue
                        valid1, new_sums1 = check_addition(S_no_i, x1, sums_set_no_i)
                        if not valid1:
                            continue
                        S_tmp = S_no_i + [x1]
                        S_tmp.sort()
                        sums_set_tmp = sums_set_no_i | set(new_sums1)
                        S_set_tmp = S_set_no_i | {x1}
                        for x2 in range(x1 + 1, N + 1):
                            if x2 in S_set_tmp:
                                continue
                            valid2, new_sums2 = check_addition(S_tmp, x2, sums_set_tmp)
                            if valid2:
                                # Accept this swap
                                S = S_tmp + [x2]
                                S.sort()
                                sums_set = sums_set_tmp | set(new_sums2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return S
    
    best_set = []
    
    # Strategy 1: Ascending greedy
    S1 = greedy_search(range(1, N + 1))
    best_set = max(best_set, S1, key=len)
    
    # Strategy 2: Descending greedy
    if time.time() < deadline:
        S2 = greedy_search(range(N, 0, -1))
        best_set = max(best_set, S2, key=len)
    
    # Strategy 3: Multiple random orderings
    if time.time() < deadline:
        for attempt in range(20):
            if time.time() > deadline:
                break
            random.seed(attempt * 12345 + 67890)
            order = list(range(1, N + 1))
            random.shuffle(order)
            S_rand = greedy_search(order)
            best_set = max(best_set, S_rand, key=len)
    
    # Local improvement on best set
    if time.time() < deadline and len(best_set) > 0:
        best_set = try_improve(best_set)
    
    # Try more random starts if time permits
    if time.time() < deadline - 10:
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            random.seed(attempt * 98765 + 43210)
            order = list(range(1, N + 1))
            random.shuffle(order)
            S_rand = greedy_search(order)
            if len(S_rand) > len(best_set):
                best_set = S_rand
            elif len(S_rand) >= len(best_set) and time.time() < deadline - 10:
                S_improved = try_improve(S_rand)
                if len(S_improved) > len(best_set):
                    best_set = S_improved
    
    return best_set
# EVOLVE-BLOCK-END
