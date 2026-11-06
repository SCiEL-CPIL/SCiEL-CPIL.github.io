import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    TARGET = 200
    MAX_ELEM = 100
    
    def is_valid_bitset(S):
        bits = 1
        for x in S:
            if x > TARGET:
                continue
            bits |= (bits << x)
            if (bits >> TARGET) & 1:
                return False
        return True
        
    def find_conflict_bitset(S):
        dp = {0: []}
        for x in S:
            if x > TARGET:
                continue
            new_dp = {}
            for s, elems in dp.items():
                ns = s + x
                if ns <= TARGET and ns not in dp and ns not in new_dp:
                    new_dp[ns] = elems + [x]
            for ns, elems in new_dp.items():
                dp[ns] = elems
            if TARGET in dp:
                return dp[TARGET]
        return None

    def greedy_build(order):
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

    # Original strategies
    order = list(range(MAX_ELEM, 0, -1))
    S = greedy_build(order)
    S = local_search_improve(S)
    try_update(S)
    
    order = list(range(1, MAX_ELEM + 1))
    S = greedy_build(order)
    S = local_search_improve(S)
    try_update(S)

    # New Strategy: Specialized 3-sum-free search on [51, 100]
    def solve_3sum_free():
        # Elements in [51, 100] only form sums 200 via triples (or pairs if we included >100, but we don't)
        # Min pair sum 103, Max 199. No pairs sum to 200.
        # Min triple sum 156, Max 297. Triples can sum to 200.
        # Quad min sum 210. No quads sum to 200.
        # So we just need to avoid triples.
        
        S = list(range(51, 101))
        
        def find_triple_conflict(S):
            S_set = set(S)
            # Check all pairs (y, z) in S
            # If 200 - y - z is in S and distinct from y, z, it's a conflict.
            # To speed up, iterate y < z.
            n = len(S)
            for i in range(n):
                y = S[i]
                for j in range(i+1, n):
                    z = S[j]
                    x = TARGET - y - z
                    if x in S_set and x != y and x != z:
                        return [x, y, z]
            return None

        # Remove conflicts
        while True:
            conflict = find_triple_conflict(S)
            if conflict is None:
                break
            # Remove a random element from the conflict
            elem_to_remove = random.choice(conflict)
            S.remove(elem_to_remove)
            
        # Local Search: Try to add back removed elements
        removed = list(set(range(51, 101)) - set(S))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(removed)
            for x in removed:
                # Check if adding x creates a triple
                # We need to check if there exist y, z in S such that x+y+z=200
                # i.e., y+z = 200-x
                target_sum = TARGET - x
                # Check pairs in S
                S_set = set(S)
                valid = True
                for i in range(len(S)):
                    y = S[i]
                    z = target_sum - y
                    if z in S_set and z != y and z != x:
                        valid = False
                        break
                if valid:
                    S.append(x)
                    removed.remove(x)
                    improved = True
                    
        return S

    S_3sum = solve_3sum_free()
    if is_valid_bitset(S_3sum):
        try_update(S_3sum)
        
    # Another variant: Start with [67, 100] and add from [51, 66]
    def solve_3sum_free_v2():
        S = list(range(67, 101))
        candidates = list(range(51, 67))
        
        def find_triple_conflict(S):
            S_set = set(S)
            n = len(S)
            for i in range(n):
                y = S[i]
                for j in range(i+1, n):
                    z = S[j]
                    x = TARGET - y - z
                    if x in S_set and x != y and x != z:
                        return [x, y, z]
            return None

        for x in candidates:
            # Try to add x
            S_test = S + [x]
            if find_triple_conflict(S_test) is None:
                S = S_test
            else:
                # If conflict, we could try to remove one from the conflict
                # But for now, just skip or try to fix
                pass
        
        # Local search on this S
        removed = list(set(range(51, 101)) - set(S))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(removed)
            for x in removed:
                S_test = S + [x]
                if find_triple_conflict(S_test) is None:
                    S = S_test
                    removed.remove(x)
                    improved = True
        return S

    S_3sum_v2 = solve_3sum_free_v2()
    if is_valid_bitset(S_3sum_v2):
        try_update(S_3sum_v2)

    if time.time() > deadline:
        return best_set if best_set else [100]
    
    if not best_set:
        best_set = [100]
        
    return best_set
# EVOLVE-BLOCK-END
