import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    # Initial valid set: {66, ..., 100}
    # Min 3-sum: 66+67+68 = 201 > 200
    # Max 2-sum: 100+99 = 199 < 200
    # Thus, no subset sums to 200.
    current_set = set(range(66, 101))
    best_set = set(current_set)
    best_size = len(current_set)

    def is_valid(s):
        """
        Check if any subset of s sums to 200 using bitset DP.
        Returns True if valid (no sum 200), False otherwise.
        """
        reachable = 1  # bit 0 is 1
        for x in s:
            reachable |= (reachable << x)
            if reachable & (1 << 200):
                return False
        return True

    def find_conflict(s, target=200):
        """
        Find one subset of s that sums to target.
        Returns a list of elements in the subset, or None if no such subset exists.
        """
        # DP to find reachability and parent pointers
        # dp[j] = True if j is reachable
        # parent[j] = the element that was added to reach j (for backtracking)
        # We need to track which element led to the state.
        # Since we process elements one by one, we can store the element used.
        
        # Optimization: We only care about sums up to target.
        dp = [False] * (target + 1)
        parent = [None] * (target + 1)
        dp[0] = True
        
        for x in s:
            # Iterate backwards to avoid using x multiple times
            for j in range(target, x - 1, -1):
                if dp[j - x] and not dp[j]:
                    dp[j] = True
                    parent[j] = x
        
        if not dp[target]:
            return None
        
        # Reconstruct the subset
        conflict = []
        j = target
        while j > 0:
            x = parent[j]
            if x is None:
                # Should not happen if dp[target] is True
                break
            conflict.append(x)
            j -= x
        return conflict

    # Greedy phase: Try to add elements from 1 to 65
    # We sort candidates to try smaller ones first, as they are more "flexible"
    candidates = list(range(1, 66))
    
    for x in candidates:
        if time.time() > deadline - 5.0:
            break
            
        if x in current_set:
            continue
            
        new_set = current_set | {x}
        
        if is_valid(new_set):
            current_set = new_set
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # Conflict exists. Find it.
            conflict = find_conflict(new_set)
            if conflict:
                # The conflict must include x because current_set was valid.
                # So conflict = {x} U C' where C' is subset of current_set.
                # We want to remove some y in C' to make it valid.
                # Try removing each y in C' and check validity.
                # To maximize efficiency, we might prefer removing larger y? 
                # Or smaller y? 
                # Let's try removing each y in C' and pick the first one that works.
                # Or better: try all and pick the one that leaves the "best" set?
                # For now, try removing each y in C' (excluding x).
                
                removable = [y for y in conflict if y != x]
                found_swap = False
                for y in removable:
                    candidate_set = (current_set - {y}) | {x}
                    if is_valid(candidate_set):
                        current_set = candidate_set
                        found_swap = True
                        if len(current_set) > best_size:
                            best_size = len_size = len(current_set) # Typo fix below
                        break
                
                # If no single swap worked, we skip x.
                # We could try double swaps, but that's expensive.
                # Let's stick to single swaps for speed.
                
    # Local Search Phase: Try to improve further
    # Iterate through all possible swaps (x not in S, y in S)
    # This is O(N^2 * Cost(Valid)). N=100, so 10000 checks.
    # Cost(Valid) is O(N * 200 / word_size) which is very fast.
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for x in range(1, 101):
            if x in current_set:
                continue
            for y in range(1, 101):
                if y not in current_set:
                    continue
                # Try swapping y for x
                # Only consider if x < y? No, any swap.
                # But to avoid infinite loops, we need a metric.
                # Since size is constant, we can't use size.
                # We can use a heuristic: prefer adding smaller numbers?
                # Or just randomize order to escape local minima.
                
                # To keep it deterministic and fast, let's only try if x < y
                # This encourages filling with smaller numbers, which might leave room for others?
                # Actually, let's just try all swaps but limit iterations.
                
                if x >= y:
                    continue
                
                candidate_set = (current_set - {y}) | {x}
                if is_valid(candidate_set):
                    # Accept the swap. 
                    # Why? Smaller numbers might be better for future additions?
                    # Or it's just a random walk.
                    # Let's accept it to explore the space.
                    current_set = candidate_set
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    break # Restart inner loop? Or just continue?
                    # Breaking inner loop and restarting outer is safer for convergence.
                    # But it's slow. Let's just continue scanning.
                    
    return list(best_set)
