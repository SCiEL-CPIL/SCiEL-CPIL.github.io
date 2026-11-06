import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 256
    
    # Precompute: for each a in 1..n, and each x in a+1..n, 
    # find all c in x..n such that a | (x+c)
    # conflict_pairs[a][x] = sorted list of c values where a | (x+c)
    # If this list is empty, a and x never conflict
    conflict_pairs = {}
    for a in range(1, n + 1):
        conflict_pairs[a] = {}
        for x in range(a + 1, n + 1):
            # Find c in [x, n] such that (x + c) % a == 0
            # x + c = k*a for some k
            # c = k*a - x, need x <= c <= n
            # k*a - x >= x => k*a >= 2x => k >= ceil(2x/a)
            # k*a - x <= n => k*a <= n + x => k <= floor((n+x)/a)
            k_min = (2 * x + a - 1) // a
            k_max = (n + x) // a
            if k_min <= k_max:
                # c values: k*a - x for k in [k_min, k_max]
                cs = [k * a - x for k in range(k_min, k_max + 1)]
                # Filter to ensure c >= x and c <= n
                cs = [c for c in cs if x <= c <= n]
                if cs:
                    conflict_pairs[a][x] = cs
            # else: no conflict possible
    
    def is_valid_set(S):
        """Check: for all a < b <= c in S, a does not divide b+c"""
        S_list = sorted(S)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if b in conflict_pairs.get(a, {}):
                    for c in conflict_pairs[a][b]:
                        if c in S_set:
                            return False
        return True
    
    def can_add_element(S, x):
        """Check if x can be added to S without violating constraints.
        x must be smaller than all elements in S for the new a=x case.
        Also need to check if x as b or c creates issues with existing a < x."""
        S_set = set(S)
        S_list = sorted(S_set)
        
        # Case 1: x is the smallest (a=x), check all pairs (b,c) in S with b >= x
        # But if x < all elements in S, then for a=x, b,c are from S and b,c >= min(S) > x
        for i, b in enumerate(S_list):
            if b < x:
                continue
            for k in range(i, len(S_list)):
                c = S_list[k]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b or c, with some a < x already in S
        # For each a in S with a < x:
        #   - if x is b: check if any c in S with c >= x and a | (x+c)
        #   - if x is c: check if any b in S with a < b <= x and a | (b+x)
        for a in S_list:
            if a >= x:
                break
            # x as b: need c in S, c >= x, a | (x+c)
            if x in conflict_pairs.get(a, {}):
                for c in conflict_pairs[a][x]:
                    if c in S_set:
                        return False
            # x as c: need b in S, a < b <= x, a | (b+x)
            # b+x = k*a, b = k*a - x, need a < b <= x
            k_min = (a + 1 + x + a - 1) // a  # b >= a+1 => k*a - x >= a+1 => k >= (a+1+x)/a
            k_max = (x + x) // a  # b <= x => k*a - x <= x => k <= 2x/a
            for k in range(k_min, k_max + 1):
                b = k * a - x
                if a < b <= x and b in S_set:
                    return False
        
        return True
    
    def greedy_extend(S, candidates, deadline):
        """Try to add elements from candidates to S."""
        current_set = set(S)
        for x in candidates:
            if time.time() > deadline:
                break
            if x not in current_set and can_add_element(current_set, x):
                current_set.add(x)
        return current_set
    
    def local_search_polish(S, deadline):
        """Try to improve S by removing one element and re-adding multiple."""
        current_set = set(S)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for a in sorted(current_set):
                if time.time() > deadline - 2:
                    break
                test_set = set(current_set)
                test_set.discard(a)
                # Try to greedily add back elements not in test_set
                candidates = [x for x in range(1, n + 1) if x not in test_set]
                # Sort by how "easy" they are to add (fewer conflicts)
                def ease(x):
                    count = 0
                    for y in test_set:
                        if x < y:
                            if y in conflict_pairs.get(x, {}):
                                count += len(conflict_pairs[x][y])
                        elif y < x:
                            # y as a, x as b or c
                            if x in conflict_pairs.get(y, {}):
                                count += len(conflict_pairs[y][x])
                    return count
                candidates.sort(key=ease)
                extended = greedy_extend(test_set, candidates, deadline)
                if len(extended) > len(current_set):
                    current_set = extended
                    improved = True
                    break
        return current_set
    
    # 1. Find the best interval [L, 256]
    best_set = set()
    best_size = 0
    
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Quick check: for interval [L, n], does any a in [L,n] divide b+c for b,c in [L,n], a<b<=c?
        # For a = L (smallest), check if L divides any b+c where b,c in [L+1, n]
        # b+c ranges from 2*(L+1) to 2*n
        # L divides b+c iff b+c is a multiple of L
        # Multiples of L in [2*(L+1), 2*n]: 
        k_min = (2 * (L + 1) + L - 1) // L
        k_max = (2 * n) // L
        if k_min <= k_max:
            # There exist multiples, but need to check if they can be formed as b+c with b,c in [L+1,n]
            # For a specific multiple m = k*L, need b+c=m with L+1 <= b <= c <= n
            # b >= L+1, c = m - b <= n => b >= m - n
            # c >= b => b <= m/2
            # So b in [max(L+1, m-n), min(n, m//2)] and c = m-b in [b, n]
            # This is complex; just do full check for L >= 130
            if L >= 130:
                S_test = set(range(L, n + 1))
                if is_valid_set(S_test):
                    size = len(S_test)
                    if size > best_size:
                        best_size = size
                        best_set = S_test
        else:
            # No multiples of L in the range, so L doesn't conflict as 'a'
            # But other a values in [L+1, n] might conflict
            if L >= 130:
                S_test = set(range(L, n + 1))
                if is_valid_set(S_test):
                    size = len(S_test)
                    if size > best_size:
                        best_size = size
                        best_set = S_test

    if not best_set:
        return list(range(192, 257))

    # 2. Multi-start greedy with random restarts
    best_result = set(best_set)
    best_result_size = len(best_result)
    
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            current_set = set(best_set)
        elif restart == 1:
            # Try starting from a slightly different interval
            L = max(130, min(256, min(best_set) - 5))
            S_test = set(range(L, n + 1))
            if is_valid_set(S_test):
                current_set = S_test
            else:
                current_set = set(best_set)
        else:
            # Random upper-tail interval
            L = random.randint(140, 256)
            S_test = set(range(L, n + 1))
            if not is_valid_set(S_test):
                continue
            current_set = S_test
        
        # Greedy extend with elements sorted by conflict count
        candidates = [x for x in range(1, min(list(current_set))) if x not in current_set]
        
        def conflict_score(x):
            count = 0
            S_list = sorted(current_set)
            for b in S_list:
                if b < x:
                    continue
                for c in S_list:
                    if c < b:
                        continue
                    if (b + c) % x == 0:
                        count += 1
            return count
        
        candidates.sort(key=conflict_score)
        current_set = greedy_extend(current_set, candidates, deadline)
        
        # Local search polish
        if time.time() < deadline - 5:
            current_set = local_search_polish(current_set, deadline)
        
        if len(current_set) > best_result_size:
            best_result = current_set
            best_result_size = len(current_set)

    # 3. Final polish: try adding elements with lowest conflict
    if time.time() < deadline - 3:
        current_set = set(best_result)
        candidates = [x for x in range(1, n + 1) if x not in current_set]
        
        def conflict_score_final(x):
            count = 0
            S_list = sorted(current_set)
            for b in S_list:
                if b < x:
                    continue
                for c in S_list:
                    if c < b:
                        continue
                    if (b + c) % x == 0:
                        count += 1
            return count
        
        candidates.sort(key=conflict_score_final)
        
        for x in candidates:
            if time.time() > deadline - 1:
                break
            if can_add_element(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_result_size:
            best_result = current_set
            best_result_size = len(current_set)

    return list(best_result)

# EVOLVE-BLOCK-END
