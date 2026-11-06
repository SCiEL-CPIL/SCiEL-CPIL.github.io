import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Precompute conflict information: for each pair (a, b) with a < b, 
    # find all c such that a | (b+c) and b <= c <= n
    # This allows faster validity checks
    
    # conflicts[a][b] = list of c values where a | (b+c) and b <= c <= n
    conflicts = {}
    for a in range(1, n + 1):
        conflicts[a] = {}
        for b in range(a + 1, n + 1):
            c_list = []
            k_min = (2 * b + a - 1) // a
            k_max = (n + b) // a
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if b <= c <= n:
                    c_list.append(c)
            conflicts[a][b] = c_list
    
    def is_valid_fast(mask, S_list):
        """
        Check validity using precomputed conflicts.
        mask: bitmask of S
        S_list: sorted list of S
        """
        if not S_list:
            return True
        
        # Check a=1 case
        if 1 in S_list:
            count_gt_1 = sum(1 for x in S_list if x > 1)
            if count_gt_1 >= 2:
                return False
        
        for i, a in enumerate(S_list):
            if a == 1:
                continue
            
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for c in conflicts[a][b]:
                    if (mask >> c) & 1:
                        return False
        
        return True
    
    def get_conflicts_fast(S, x):
        """
        Find one conflicting triple in S U {x}.
        """
        S_new = S | {x}
        mask = 0
        for y in S_new:
            mask |= (1 << y)
        S_list = sorted(S_new)
        
        # Check a=1
        if 1 in S_list:
            if x == 1:
                gt_1 = [y for y in S if y > 1]
                if len(gt_1) >= 2:
                    return (1, gt_1[0], gt_1[1])
            else:
                if 1 in S:
                    gt_1 = [y for y in S if y > 1]
                    if len(gt_1) >= 1:
                        return (1, gt_1[0], x)
        
        for i, a in enumerate(S_list):
            if a == 1:
                continue
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for c in conflicts[a][b]:
                    if (mask >> c) & 1:
                        return (a, b, c)
        
        return None
    
    # Step 1: Find largest valid interval [L, n]
    best_L = n
    for L in range(1, n + 1):
        if time.time() > deadline - 5:
            break
        S_test = set(range(L, n + 1))
        mask = 0
        for x in S_test:
            mask |= (1 << x)
        if is_valid_fast(mask, sorted(S_test)):
            best_L = L
            break
    
    # Step 2: Multi-start Greedy + Local Search
    best_set = set(range(best_L, n + 1))
    best_size = len(best_set)
    
    # Candidates to add
    candidates = list(range(1, best_L))
    
    def greedy_add(S, candidates, time_limit):
        S = set(S)
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in S:
                continue
            
            test = S | {x}
            mask = 0
            for y in test:
                mask |= (1 << y)
            if is_valid_fast(mask, sorted(test)):
                S.add(x)
            else:
                # Try to swap
                if time.time() < time_limit - 2:
                    conflict = get_conflicts_fast(S, x)
                    if conflict:
                        a, b, c = conflict
                        for y in [b, c, a]:
                            if y == x:
                                continue
                            if y in S:
                                test_swap = S | {x}
                                test_swap.remove(y)
                                mask = 0
                                for z in test_swap:
                                    mask |= (1 << z)
                                if is_valid_fast(mask, sorted(test_swap)):
                                    S = test_swap
                                    break
        return S

    # Run multiple attempts with different random seeds
    for attempt in range(80):
        if time.time() > deadline - 12:
            break
        
        random.seed(attempt * 12345 + 67890)
        S = set(range(best_L, n + 1))
        S = greedy_add(S, candidates, deadline - 12)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Step 3: Intensive Local Search on best_set
    current = set(best_set)
    remaining = [x for x in range(1, n + 1) if x not in current]
    random.shuffle(remaining)
    
    improved = True
    while improved and time.time() < deadline - 4:
        improved = False
        for x in remaining:
            if time.time() > deadline - 4:
                break
            if x in current:
                continue
            
            test = current | {x}
            mask = 0
            for y in test:
                mask |= (1 << y)
            if is_valid_fast(mask, sorted(test)):
                current.add(x)
                improved = True
                best_size = len(current)
                best_set = current
            else:
                # Try all possible swaps: remove any y in current, add x
                # This is more thorough than just trying the conflict elements
                if time.time() < deadline - 4:
                    found_swap = False
                    for y in sorted(current, reverse=True):  # Try removing larger elements first
                        test_swap = current | {x}
                        test_swap.remove(y)
                        mask = 0
                        for z in test_swap:
                            mask |= (1 << z)
                        if is_valid_fast(mask, sorted(test_swap)):
                            current = test_swap
                            improved = True
                            best_size = len(current)
                            best_set = current
                            found_swap = True
                            break
                    if found_swap:
                        # Re-shuffle remaining after a swap
                        remaining = [x2 for x2 in range(1, n + 1) if x2 not in current]
                        random.shuffle(remaining)
    
    # Step 4: Final polish - try adding any remaining elements one more time
    if time.time() < deadline - 2:
        remaining = [x for x in range(1, n + 1) if x not in current]
        for x in remaining:
            if time.time() > deadline - 2:
                break
            test = current | {x}
            mask = 0
            for y in test:
                mask |= (1 << y)
            if is_valid_fast(mask, sorted(test)):
                current.add(x)
                best_size = len(current)
                best_set = current
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
