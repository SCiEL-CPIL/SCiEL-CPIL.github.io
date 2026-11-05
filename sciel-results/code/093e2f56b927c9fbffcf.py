import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for safety
    
    def is_valid_bitset(S):
        """Check if no subset of S sums to 200 using bitset."""
        reachable = 1  # bit i set means sum i is reachable
        for x in S:
            reachable |= (reachable << x)
            if (reachable >> 200) & 1:
                return False
        return True
    
    def get_reachable_bitset(S):
        """Get the reachable sums bitset for S."""
        reachable = 1
        for x in S:
            reachable |= (reachable << x)
        return reachable
    
    def can_add(S_reachable, x):
        """Check if adding x to a set with reachable bitset S_reachable keeps sum 200 unreachable."""
        new_reach = S_reachable | (S_reachable << x)
        return ((new_reach >> 200) & 1) == 0
    
    best_set = set()
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            if is_valid_bitset(S):
                best_set = set(S)
                best_size = len(S)
    
    # 1. Deterministic Greedies
    # Large to Small
    s = []
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    update_best(set(s))
    
    # Small to Large
    s = []
    reachable = 1
    for x in range(1, 101):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    update_best(set(s))
    
    # 2. Subtractive Greedy: Start with full set, remove elements to break sum-200 subsets
    # Find a witness (subset summing to 200), remove one element from it, repeat
    def find_witness(S):
        """Find a subset of S summing to 200, or return None."""
        elements = list(S)
        random.shuffle(elements)
        
        # DP to find if 200 is reachable, and track path
        dp = [False] * 201
        dp[0] = True
        # parent[s] = (element, prev_sum) that first made s reachable
        parent = [None] * 201
        
        for x in elements:
            for s in range(200, x - 1, -1):
                if dp[s - x] and not dp[s]:
                    dp[s] = True
                    parent[s] = (x, s - x)
        
        if not dp[200]:
            return None
        
        # Reconstruct
        witness = []
        s = 200
        while s > 0:
            x, prev_s = parent[s]
            witness.append(x)
            s = prev_s
        return witness
    
    # Run subtractive greedy multiple times with different randomizations
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        S = set(range(1, 101))
        # Remove elements until valid
        for _ in range(100):  # Max 100 removals
            if is_valid_bitset(S):
                break
            w = find_witness(S)
            if w is None:
                break
            # Remove a random element from the witness
            # Prefer removing larger elements (they're "less valuable" for cardinality? No, all count equally)
            # Actually, to maximize cardinality, we want to remove as few as possible.
            # Remove one random element from witness
            to_remove = random.choice(w)
            S.remove(to_remove)
        
        update_best(S)
    
    # 3. Local Search: Try to add excluded elements
    # Start from best_set
    current_set = set(best_set)
    current_reach = get_reachable_bitset(current_set)
    excluded = [x for x in range(1, 101) if x not in current_set]
    
    # Iteratively try to add elements
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        random.shuffle(excluded)
        for x in excluded[:]:
            if x not in current_set:
                if can_add(current_reach, x):
                    current_set.add(x)
                    current_reach |= (current_reach << x)
                    excluded.remove(x)
                    improved = True
                    update_best(current_set)
    
    # 4. Swap-based local search: Try swapping elements to enable additions
    for _ in range(500):  # Limited number of swap attempts
        if time.time() > deadline - 1:
            break
        
        if not excluded or not current_set:
            break
        
        # Pick a random excluded element
        x = random.choice(excluded)
        
        # Check if x can be added
        if can_add(current_reach, x):
            current_set.add(x)
            current_reach |= (current_reach << x)
            excluded.remove(x)
            update_best(current_set)
            continue
        
        # Try to find a swap: remove a from current_set, add x
        # We need the new set to be valid
        # Heuristic: try removing elements that are "conflicting" with x
        # Specifically, if 200-x is reachable, we need to break all paths to 200-x
        
        # Try a few random swaps
        for _ in range(5):
            if time.time() > deadline - 1:
                break
            if not current_set:
                break
            a = random.choice(list(current_set))
            new_set = set(current_set)
            new_set.remove(a)
            new_set.add(x)
            
            if is_valid_bitset(new_set):
                # Valid swap
                current_set = new_set
                current_reach = get_reachable_bitset(current_set)
                excluded.remove(x)
                excluded.append(a)
                random.shuffle(excluded)
                update_best(current_set)
                break
    
    # 5. Final polish: try all possible single additions
    if time.time() < deadline - 1:
        for x in range(1, 101):
            if x not in current_set:
                if can_add(current_reach, x):
                    current_set.add(x)
                    current_reach |= (current_reach << x)
                    update_best(current_set)
    
    return list(best_set)
