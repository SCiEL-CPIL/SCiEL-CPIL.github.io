import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (evaluator kills at 90s)
    
    TARGET = 200
    MAX_ELEM = 100
    
    # Bitset-based subset sum check: use Python int as bitset
    # Bit i set means sum i is reachable
    def is_valid_bitset(S):
        """Check if any subset of S sums to TARGET using bitset DP."""
        bits = 1  # bit 0 set: sum 0 reachable
        for x in S:
            if x > TARGET:
                continue
            bits |= (bits << x)
            if (bits >> TARGET) & 1:
                return False
        return True
    
    def find_conflict_fast(S):
        """Find a subset of S that sums to TARGET. Returns list of elements or None."""
        # dp[s] = bitmask of which elements (by index) are used
        # Use list of length TARGET+1, initialized to -1
        # parent[s] = (prev_sum, element_index)
        parent = [None] * (TARGET + 1)
        parent[0] = (0, -1)  # base case
        reachable = [0] * (TARGET + 1)
        reachable[0] = 1
        
        for idx, x in enumerate(S):
            if x > TARGET:
                continue
            # Process in reverse to avoid using x multiple times in one iteration
            for s in range(TARGET - x, -1, -1):
                if reachable[s] and not reachable[s + x]:
                    reachable[s + x] = 1
                    parent[s + x] = (s, idx)
            if reachable[TARGET]:
                # Reconstruct
                elems = []
                cur = TARGET
                while cur > 0:
                    prev_s, elem_idx = parent[cur]
                    elems.append(S[elem_idx])
                    cur = prev_s
                return elems
        return None
    
    def greedy_build(order):
        """Build a valid set by greedily adding elements in given order."""
        S = []
        bits = 1
        for x in order:
            if x > TARGET:
                S.append(x)
                continue
            if (bits >> (TARGET - x)) & 1:
                continue
            bits |= (bits << x)
            S.append(x)
        return S
    
    def local_search_improve(S):
        """Try to add elements from complement to S while maintaining validity."""
        S_set = set(S)
        all_elements = set(range(1, MAX_ELEM + 1))
        complement = list(all_elements - S_set)
        improved = True
        current_S = list(S)
        
        bits = 1
        for x in current_S:
            if x <= TARGET:
                bits |= (bits << x)
        
        while improved:
            improved = False
            random.shuffle(complement)
            for x in complement:
                if x > TARGET:
                    current_S.append(x)
                    S_set.add(x)
                    complement.remove(x)
                    improved = True
                else:
                    if not ((bits >> (TARGET - x)) & 1):
                        bits |= (bits << x)
                        current_S.append(x)
                        S_set.add(x)
                        complement.remove(x)
                        improved = True
        return current_S
    
    best_set = []
    best_size = -1
    
    def try_update(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Strategy 0: Try explicit known-good sets first
    # {1, ..., 50} - check if valid
    explicit_sets = [
        list(range(1, 51)),
        list(range(1, 46)) + list(range(55, 101)),  # 45 + 46 = 91 elements
        list(range(1, 41)) + list(range(61, 101)),  # 40 + 40 = 80 elements
    ]
    
    for es in explicit_sets:
        if is_valid_bitset(es):
            es = local_search_improve(es)
            try_update(es)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 1: Greedy descending (large elements first)
    order = list(range(MAX_ELEM, 0, -1))
    S = greedy_build(order)
    S = local_search_improve(S)
    try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 2: Greedy ascending
    order = list(range(1, MAX_ELEM + 1))
    S = greedy_build(order)
    S = local_search_improve(S)
    try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 3: Randomized greedy with multiple restarts
    num_restarts = 3000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = list(range(1, MAX_ELEM + 1))
        random.shuffle(order)
        S = greedy_build(order)
        S = local_search_improve(S)
        try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 4: Start from full set, remove minimal conflicts
    num_removal_tries = 800
    for i in range(num_removal_tries):
        if time.time() > deadline:
            break
        
        S = list(range(1, MAX_ELEM + 1))
        random.shuffle(S)
        
        max_removals = 50
        for _ in range(max_removals):
            conflict = find_conflict_fast(S)
            if conflict is None:
                break
            elem_to_remove = random.choice(conflict)
            S.remove(elem_to_remove)
        
        if is_valid_bitset(S):
            S = local_search_improve(S)
            try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 5: Smart removal - remove elements that appear in most conflicts
    num_smart_removal = 300
    for i in range(num_smart_removal):
        if time.time() > deadline:
            break
        
        S = list(range(1, MAX_ELEM + 1))
        
        conflict_count = {x: 0 for x in S}
        for _ in range(30):
            conflict = find_conflict_fast(S)
            if conflict is None:
                break
            for elem in conflict:
                conflict_count[elem] += 1
        
        for elem in sorted(S, key=lambda x: conflict_count[x], reverse=True):
            if not is_valid_bitset(S):
                S.remove(elem)
            else:
                break
        
        if is_valid_bitset(S):
            S = local_search_improve(S)
            try_update(S)
    
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    # Strategy 6: Hill climbing on the current best set
    if best_set:
        S = best_set.copy()
        S_set = set(S)
        complement = list(set(range(1, MAX_ELEM + 1)) - S_set)
        
        bits = 1
        for x in S:
            if x <= TARGET:
                bits |= (bits << x)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(complement)
            for x in complement:
                if x > TARGET:
                    S.append(x)
                    S_set.add(x)
                    complement.remove(x)
                    improved = True
                else:
                    if not ((bits >> (TARGET - x)) & 1):
                        bits |= (bits << x)
                        S.append(x)
                        S_set.add(x)
                        complement.remove(x)
                        improved = True
        
        try_update(S)
    
    # Final check
    if time.time() > deadline:
        return best_set if best_set else [100]
    
    if not best_set:
        best_set = [100]
    
    return best_set
# EVOLVE-BLOCK-END
