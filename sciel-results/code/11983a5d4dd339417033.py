import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
WEIGHTS = {i: L // i for i in UNIVERSE}

def is_valid_subset(S):
    """
    Check if the set S is reciprocal-subsum-free.
    """
    if not S:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    for i, x in enumerate(S_list):
        target = WEIGHTS[x]
        # Others
        others = [WEIGHTS[y] for y in S_list if y != x]
        
        # Check if target is a subset sum of others
        # Pruning: if sum(others) < target, continue
        if sum(others) < target:
            continue
            
        # Use a set for subset sums, pruning values > target
        possible_sums = {0}
        for w in others:
            if w > target:
                continue
            new_sums = set()
            for s in possible_sums:
                ns = s + w
                if ns <= target:
                    new_sums.add(ns)
            possible_sums.update(new_sums)
            if target in possible_sums:
                return False
                
    return True

def find_conflicts(S):
    """
    Returns a list of conflicts. Each conflict is (x, T) where w_x == sum(T).
    We only need to know which elements are involved.
    """
    S_list = list(S)
    conflicts = []
    for i, x in enumerate(S_list):
        target = WEIGHTS[x]
        others = [WEIGHTS[y] for y in S_list if y != x]
        others_elements = [y for y in S_list if y != x]
        
        if sum(others) < target:
            continue
            
        # Find a subset T that sums to target
        # Backtracking to find one witness subset
        def find_subset(idx, current_sum, current_set):
            if current_sum == target:
                return current_set
            if idx == len(others) or current_sum > target:
                return None
            # Option 1: exclude others[idx]
            res = find_subset(idx + 1, current_sum, current_set)
            if res is not None:
                return res
            # Option 2: include others[idx]
            if current_sum + others[idx] <= target:
                res = find_subset(idx + 1, current_sum + others[idx], current_set + [others_elements[idx]])
                if res is not None:
                    return res
            return None

        witness = find_subset(0, 0, [])
        if witness:
            # Conflict involves x and all elements in witness
            involved = set([x])
            involved.update(witness)
            conflicts.append(involved)
            
    return conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    # Initialize with all elements
    S = set(UNIVERSE)
    
    # Phase 1: Remove elements to resolve conflicts
    # We iteratively find conflicts and remove the "most critical" element
    # A simple heuristic: remove the element that is the TARGET of a conflict, 
    # or if multiple, remove one involved in the most conflicts.
    
    while time.time() < deadline:
        conflicts = find_conflicts(S)
        if not conflicts:
            break
            
        # Count how many times each element appears in conflicts
        count = {e: 0 for e in S}
        for c in conflicts:
            for e in c:
                if e in count:
                    count[e] += 1
        
        # Remove the element with the highest count
        # Tie-breaker: smaller index? Or random?
        # Removing the target (x) is generally effective as it kills the equation.
        # Let's prioritize targets if possible, but count-based is robust.
        max_count = max(count.values())
        candidates = [e for e in S if count[e] == max_count]
        # Randomly pick one to escape local minima
        random.shuffle(candidates)
        S.remove(candidates[0])
        
    # Phase 2: Try to add back elements
    # Sort candidates by some heuristic (e.g., larger weight = smaller index?)
    # Adding smaller indices (larger weights) is harder to fit?
    # Actually, let's just try all elements not in S.
    remaining = [e for e in UNIVERSE if e not in S]
    # Shuffle to try different orders
    random.shuffle(remaining)
    
    for e in remaining:
        if time.time() > deadline:
            break
        # Try adding e
        new_S = S | {e}
        if is_valid_subset(new_S):
            S = new_S
            # If we added one, we might be able to add more, so restart loop?
            # Or just continue? Continuing is faster but might miss combos.
            # Given small N, let's restart the addition phase if we succeed?
            # No, simple sequential addition is okay for a first pass.
            # Let's do a full restart of Phase 2 if we add something, to be greedy?
            # Actually, just continuing the loop `remaining` is fine, but 
            # adding `e` might make previously failed `f` valid? Unlikely if f failed due to conflict with existing S.
            # But adding e changes S.
            # To be safe and fast, we'll just do one pass.
            
    # Phase 3: Local Search / Swaps
    # Try swapping elements in S with elements not in S
    # If |S| is already high, we might want to try to increase size.
    # The loop above only adds. It never removes to add two?
    # The removal phase in Phase 1 was aggressive.
    # Let's do a few iterations of:
    # 1. Try to add any remaining element.
    # 2. If none can be added, try swapping: remove x from S, add y from ~S, if valid and size increases?
    #    Wait, swap keeps size same. We want to INCREASE size.
    #    So we need to remove 1, add 2? Or remove 0, add 1.
    #    Phase 2 does remove 0, add 1.
    #    If Phase 2 fails, we are stuck at current size.
    #    Can we do better?
    #    Maybe Phase 1 removed too much.
    #    Let's restart with a different random seed or order?
    
    # Given the time budget, let's run a few restarts with different randomizations in Phase 1.
    best_S = set(S)
    
    for _ in range(5): # A few restarts
        if time.time() > deadline:
            break
        S = set(UNIVERSE)
        random.seed() # Reset random state? Or just use current random
        # Randomize removal order
        while time.time() < deadline:
            conflicts = find_conflicts(S)
            if not conflicts:
                break
            count = {e: 0 for e in S}
            for c in conflicts:
                for e in c:
                    if e in count:
                        count[e] += 1
            max_count = max(count.values())
            candidates = [e for e in S if count[e] == max_count]
            random.shuffle(candidates)
            S.remove(candidates[0])
            
        # Add back
        remaining = [e for e in UNIVERSE if e not in S]
        random.shuffle(remaining)
        for e in remaining:
            if time.time() > deadline:
                break
            new_S = S | {e}
            if is_valid_subset(new_S):
                S = new_S
                
        if len(S) > len(best_S):
            best_S = set(S)
            
    return list(best_S)

# EVOLVE-BLOCK-START
def construct():
    # Call the logic defined above
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    import time
    import random
    from math import lcm
    from functools import reduce

    UNIVERSE = list(range(1, 25))
    L = reduce(lcm, UNIVERSE)
    WEIGHTS = {i: L // i for i in UNIVERSE}

    def is_valid_subset(S):
        if not S:
            return True
        S_list = list(S)
        for i, x in enumerate(S_list):
            target = WEIGHTS[x]
            others = [WEIGHTS[y] for y in S_list if y != x]
            if sum(others) < target:
                continue
            possible_sums = {0}
            for w in others:
                if w > target:
                    continue
                new_sums = set()
                for s in possible_sums:
                    ns = s + w
                    if ns <= target:
                        new_sums.add(ns)
                possible_sums.update(new_sums)
                if target in possible_sums:
                    return False
        return True

    def find_conflicts(S):
        S_list = list(S)
        conflicts = []
        for i, x in enumerate(S_list):
            target = WEIGHTS[x]
            others = [WEIGHTS[y] for y in S_list if y != x]
            others_elements = [y for y in S_list if y != x]
            if sum(others) < target:
                continue
            
            def find_subset(idx, current_sum, current_set):
                if current_sum == target:
                    return current_set
                if idx == len(others) or current_sum > target:
                    return None
                res = find_subset(idx + 1, current_sum, current_set)
                if res is not None:
                    return res
                if current_sum + others[idx] <= target:
                    res = find_subset(idx + 1, current_sum + others[idx], current_set + [others_elements[idx]])
                    if res is not None:
                        return res
                return None

            witness = find_subset(0, 0, [])
            if witness:
                involved = set([x])
                involved.update(witness)
                conflicts.append(involved)
        return conflicts

    start_time = time.time()
    deadline = start_time + 85.0

    best_S = set(range(13, 25)) # Fallback to seed

    for restart in range(10):
        if time.time() > deadline:
            break
        S = set(UNIVERSE)
        
        while time.time() < deadline:
            conflicts = find_conflicts(S)
            if not conflicts:
                break
            count = {e: 0 for e in S}
            for c in conflicts:
                for e in c:
                    if e in count:
                        count[e] += 1
            max_count = max(count.values())
            candidates = [e for e in S if count[e] == max_count]
            random.shuffle(candidates)
            S.remove(candidates[0])
        
        remaining = [e for e in UNIVERSE if e not in S]
        random.shuffle(remaining)
        for e in remaining:
            if time.time() > deadline:
                break
            new_S = S | {e}
            if is_valid_subset(new_S):
                S = new_S
        
        if len(S) > len(best_S):
            best_S = set(S)
            
    return list(best_S)
