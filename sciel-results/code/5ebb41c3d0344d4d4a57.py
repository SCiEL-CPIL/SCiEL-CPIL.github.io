"""Improved program for large sets with Property P."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin within 90s limit
    
    n = 256
    
    def is_valid(S):
        """Check if set S satisfies the property."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(x, S):
        """Check if x can be added to set S while maintaining the property."""
        S_list = sorted(S)
        m = len(S_list)
        
        # Case 1: x is the smallest (a), with b <= c from S, both > x
        for i in range(m):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, m):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: a from S < x, x is b or c
        # (a, x, c) where a < x <= c, c in S
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
            # (a, b, x) where a < b <= x, b in S
            for b in S_list:
                if b <= a:
                    continue
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
            # (a, x, x)
            if (2 * x) % a == 0:
                return False
        
        return True
    
    # Baseline: upper 1/3
    S_upper_1_3 = list(range(171, 257))
    best_set = set(S_upper_1_3)
    best_size = len(best_set)
    
    # Strategy 1: Greedy descending from empty
    S = set()
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
    
    # Strategy 2: Greedy ascending from empty
    if time.time() < deadline:
        S = set()
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Strategy 3: Start with upper 1/3, add lower elements greedily (descending)
    if time.time() < deadline:
        S = set(S_upper_1_3)
        for x in range(170, 0, -1):
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Strategy 4: Start with upper half [129,256], try to fix conflicts
    if time.time() < deadline:
        S = set(range(129, 257))
        # Remove elements that cause conflicts, then try to add back others
        # First, find all conflicts and remove the "worst" elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            S_list = sorted(S)
            m = len(S_list)
            # Find all violating triples
            conflicts = {}  # element -> count of conflicts it's involved in
            for i in range(m):
                a = S_list[i]
                for j in range(i + 1, m):
                    b = S_list[j]
                    for k in range(j, m):
                        c = S_list[k]
                        if (b + c) % a == 0:
                            # This triple violates
                            for elem in [a, b, c]:
                                conflicts[elem] = conflicts.get(elem, 0) + 1
            if conflicts:
                # Remove the element with most conflicts
                worst = max(conflicts, key=conflicts.get)
                S.discard(worst)
                changed = True
        # Now try to add back removed elements
        for x in range(128, 0, -1):
            if time.time() > deadline:
                break
            if x not in S and can_add(x, S):
                S.add(x)
        # Also try to add back from 129-256
        for x in range(256, 128, -1):
            if time.time() > deadline:
                break
            if x not in S and can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Strategy 5: Randomized greedy with multiple restarts
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Strategy 6: Local search - try swaps to increase size
    if time.time() < deadline:
        S = set(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = sorted(S)
            # Try to add any element not in S
            for x in range(n, 0, -1):
                if time.time() > deadline:
                    break
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            if not improved:
                # Try swaps: remove one, add one
                for x_remove in S_list:
                    if time.time() > deadline:
                        break
                    S_test = S - {x_remove}
                    for x_add in range(n, 0, -1):
                        if time.time() > deadline:
                            break
                        if x_add in S_test:
                            continue
                        if can_add(x_add, S_test):
                            # Check if this actually increases size (it should since we remove 1, add 1)
                            # But maybe we can add MORE after the swap
                            S_new = S_test | {x_add}
                            # Try to add more
                            for x_extra in range(n, 0, -1):
                                if time.time() > deadline:
                                    break
                                if x_extra not in S_new and can_add(x_extra, S_new):
                                    S_new.add(x_extra)
                            if len(S_new) > len(S):
                                S = S_new
                                improved = True
                                break
                    if improved:
                        break
    
    # Final validation
    if not is_valid(best_set):
        best_set = set(S_upper_1_3)
    
    return list(best_set)
# EVOLVE-BLOCK-END
