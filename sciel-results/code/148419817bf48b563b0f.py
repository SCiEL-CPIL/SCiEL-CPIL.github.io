import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    n = 256
    
    # Precompute conflict tables: for each a, list of (b, c) pairs where a | (b+c), b <= c, a < b
    conflict_pairs = {}
    for a in range(1, n + 1):
        pairs = []
        for b in range(a + 1, n + 1):
            k_min = (2 * b + a - 1) // a
            k_max = (n + b) // a
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if b <= c <= n:
                    pairs.append((b, c))
        conflict_pairs[a] = pairs
    
    # Precompute for each element x, which (a, b, c) triples it participates in
    # For faster conflict detection
    element_conflicts = {x: [] for x in range(1, n + 1)}
    for a in range(1, n + 1):
        for b, c in conflict_pairs[a]:
            element_conflicts[a].append((b, c, a))  # (b, c, a) meaning a conflicts with b,c
            element_conflicts[b].append((a, c, b))  # b conflicts with a,c
            element_conflicts[c].append((a, b, c))  # c conflicts with a,b
    
    def is_valid(S):
        """
        Check if S is a valid set using precomputed conflict pairs.
        """
        if not S:
            return True
        
        S_list = sorted(S)
        
        for a in S_list:
            if a == 1:
                # 1 divides everything. Need at most one element > 1.
                count_gt_1 = sum(1 for x in S_list if x > 1)
                if count_gt_1 >= 2:
                    return False
                continue
            
            # Check all conflict pairs for this a
            for b, c in conflict_pairs[a]:
                if b in S and c in S:
                    return False
        
        return True
    
    def get_conflict(S, x):
        """
        If S U {x} is invalid, find one conflicting triple involving x.
        Returns (a, b, c) or None.
        """
        if x == 1:
            gt_1 = [y for y in S if y > 1]
            if len(gt_1) >= 2:
                return (1, gt_1[0], gt_1[1])
            return None
        
        if 1 in S:
            gt_1 = [y for y in S if y > 1]
            if len(gt_1) >= 1:
                return (1, gt_1[0], x)
        
        # Check conflicts where x is the 'a' role
        for b, c in conflict_pairs[x]:
            if b in S and c in S:
                return (x, b, c)
        
        # Check conflicts where x is the 'b' or 'c' role
        for a, other in element_conflicts[x]:
            if a in S and other in S:
                # Determine which is b and which is c
                if a < x:
                    if x <= other:
                        return (a, x, other)
                    else:
                        return (a, other, x)
                else:
                    if other <= x:
                        return (other, x, a)
                    else:
                        return (other, a, x)
        
        return None
    
    # Step 1: Find best starting interval
    best_start = None
    best_start_size = 0
    
    for L in range(1, n + 1):
        if time.time() > deadline - 20:
            break
        S_test = set(range(L, n + 1))
        if is_valid(S_test):
            # Try to add some small elements
            S_greedy = set(S_test)
            for x in range(1, L):
                if time.time() > deadline - 20:
                    break
                if x in S_greedy:
                    continue
                test = S_greedy | {x}
                if is_valid(test):
                    S_greedy.add(x)
            
            if len(S_greedy) > best_start_size:
                best_start_size = len(S_greedy)
                best_start = S_greedy
    
    if best_start is None:
        best_start = {n}
    
    best_set = best_start
    best_size = len(best_set)
    
    # Step 2: Multi-start greedy with swaps
    candidates = list(range(1, n + 1))
    
    def greedy_with_swaps(S, candidates, time_limit):
        S = set(S)
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in S:
                continue
            
            test = S | {x}
            if is_valid(test):
                S.add(x)
            else:
                if time.time() < time_limit - 0.5:
                    conflict = get_conflict(S, x)
                    if conflict:
                        a, b, c = conflict
                        for y in [a, b, c]:
                            if y == x or y not in S:
                                continue
                            test_swap = S | {x}
                            test_swap.remove(y)
                            if is_valid(test_swap):
                                S = test_swap
                                break
        return S
    
    # Run multiple attempts with different random orderings
    num_attempts = 40
    for attempt in range(num_attempts):
        if time.time() > deadline - 8:
            break
        
        random.shuffle(candidates)
        S = greedy_with_swaps(best_start, candidates, deadline - 8)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Step 3: Final local search on best_set
    current = set(best_set)
    remaining = [x for x in range(1, n + 1) if x not in current]
    random.shuffle(remaining)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for x in remaining:
            if time.time() > deadline - 2:
                break
            if x in current:
                continue
            test = current | {x}
            if is_valid(test):
                current.add(x)
                improved = True
                best_size = len(current)
                best_set = current
            else:
                conflict = get_conflict(current, x)
                if conflict:
                    a, b, c = conflict
                    for y in [a, b, c]:
                        if y == x or y not in current:
                            continue
                        test_swap = current | {x}
                        test_swap.remove(y)
                        if is_valid(test_swap):
                            current = test_swap
                            improved = True
                            best_size = len(current)
                            best_set = current
                            break

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
