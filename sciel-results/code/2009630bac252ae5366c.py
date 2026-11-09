import time

def is_mean(a, others):
    # a is the candidate mean, others is the set of other elements
    # Check if a is the mean of any subset of others of size >= 2
    # Equivalent to: sum(subset) == len(subset) * a
    # Let diff_i = other_i - a
    # We want sum(subset_diffs) == 0
    # Subset size >= 2 is guaranteed if no diff is 0 (which is true as a not in others)
    # and we don't pick empty set.
    
    diffs = [y - a for y in others]
    
    # Range of sums
    # min_sum = sum(d for d in diffs if d < 0)
    # max_sum = sum(d for d in diffs if d > 0)
    
    # To be safe and fast, use a fixed range based on max possible
    # Max diff 512. Max count 48. Max sum 25000.
    # Let's use a dictionary or a set of reachable sums.
    # Since range is small, a boolean array is best.
    
    # Calculate bounds
    pos_sum = sum(d for d in diffs if d > 0)
    neg_sum = sum(d for d in diffs if d < 0)
    
    if pos_sum == 0 or neg_sum == 0:
        return False # Can't sum to 0 with only pos or only neg (unless all 0, which is false)
        
    offset = -neg_sum
    size = pos_sum - neg_sum + 1
    
    # dp[i] is True if sum i - offset is reachable
    dp = [False] * size
    dp[offset] = True # Sum 0
    
    # We need to know if 0 is reachable by a non-empty subset.
    # If we just check dp[offset] at the end, it's True from the start.
    # We need to track if it was reached by adding elements.
    # Since no element is 0, any addition changes the sum.
    # So if dp[offset] becomes True again after being False, or if we track counts?
    # Actually, if we start with dp[offset]=True, and we do:
    # new_dp = dp copy
    # for d in diffs:
    #   for i in range(size):
    #     if dp[i]: new_dp[i+d] = True
    # This will keep dp[offset] True forever.
    
    # Better: Start with dp = [False]*size.
    # This represents "no elements picked".
    # Then we want to know if we can pick a subset that sums to 0.
    # If we start with empty set, sum is 0.
    # We want to exclude the empty set.
    # So, we can just check if there is a subset of size >= 1 that sums to 0.
    # Since no element is 0, size >= 1 implies size >= 2? No, size 1 means d=0, impossible.
    # So any subset summing to 0 has size >= 2.
    
    # So we just need to check if 0 is a subset sum of diffs, excluding the empty set.
    # We can do this by initializing dp with all False, and then... 
    # Wait, standard subset sum finds if 0 is reachable.
    # If we start with `dp = {0}`, and add elements, 0 is always reachable (empty set).
    # To exclude empty set, we can just check if there is ANY other way to reach 0.
    # Or simpler: 
    # 1. Check if any single diff is 0 (False).
    # 2. Check if any pair sums to 0.
    # 3. Check if any triple sums to 0.
    # This is $O(N^2)$ for pairs, $O(N^3)$ for triples.
    # For N=48, $48^3 \approx 100k$ ops. Very fast.
    # And we can just do this for all subset sizes? No, subset sum covers all sizes.
    
    # Let's stick to the Bitset/Array DP but handle the "empty set" issue.
    # We can initialize the DP such that the "0 sum" state is only activated by non-empty subsets.
    # This is tricky.
    
    # Alternative: Just use a set of reachable sums.
    reachable = {0}
    for d in diffs:
        new_reachable = set()
        for s in reachable:
            new_reachable.add(s + d)
        reachable = reachable.union(new_reachable)
        
    # Now, is 0 in reachable? Yes (from start).
    # Did we add 0 via a non-empty subset?
    # If 0 was in reachable BEFORE adding any element, it's the empty set.
    # If 0 is in reachable AFTER, it might be empty or non-empty.
    # We can track the size of the subset?
    # Or simply:
    # If 0 is in reachable, and there exists a d such that -d is in reachable (before adding d)?
    
    # Simpler: 
    # If the set of diffs contains 0, return True. (False here)
    # Else, if 0 is in the subset sums, it must be formed by at least 2 elements.
    # So we just need to know if 0 is in the subset sums.
    # The only "false positive" is the empty set.
    # But the empty set sums to 0.
    # So `0 in reachable` is always True.
    
    # We need to check if 0 is reachable by a subset of size >= 1.
    # Let's modify the DP to track min size? No.
    # Let's just check:
    # Is there a subset of diffs that sums to 0?
    # If we remove the "start with 0" and instead "start with nothing", 
    # and add elements one by one, we build the set of sums.
    # Initially sums = {0} (empty set).
    # After adding d1: sums = {0, d1}.
    # If d1 == 0, we have a subset of size 1 summing to 0.
    # If d1 != 0, 0 is still there (empty set).
    # After adding d2: sums = {0, d1, d2, d1+d2}.
    # If d1+d2 == 0, we have a subset of size 2 summing to 0.
    # So, if 0 is in the final set of sums, AND 0 was not ONLY from the empty set...
    # Actually, if any non-empty subset sums to 0, then 0 is in the set.
    # The empty set also sums to 0.
    # So 0 is ALWAYS in the set.
    # This logic is flawed.
    
    # Correct Logic:
    # We want to know if there is a subset $S \subseteq \text{diffs}, S \neq \emptyset$ such that $\sum_{d \in S} d = 0$.
    # Since no $d=0$, $|S| \ge 2$.
    # We can compute the set of all subset sums.
    # Let $R$ be the set of all subset sums.
    # $0 \in R$ is always true.
    # We need to know if there is *another* way to form 0.
    # This is equivalent to: Is the number of subsets summing to 0 greater than 1?
    # Or: If we remove the empty set from consideration, is 0 reachable?
    
    # Efficient check:
    # If we sort diffs, we can check pairs, triples...
    # But DP is better.
    # Let's use a boolean array `dp` where `dp[s]` is True if sum `s` is reachable.
    # Initialize `dp[0] = True`.
    # For each `d` in diffs:
    #    `new_dp = dp.copy()`
    #    For each `s` where `dp[s]` is True:
    #        `new_dp[s+d] = True`
    #    `dp = new_dp`
    # At the end, `dp[0]` is True.
    # This doesn't tell us if it's non-empty.
    
    # Trick:
    # Initialize `dp` with all False.
    # This means "no subset picked yet".
    # But we need to pick at least one.
    # Let's pick the first element `d1`.
    # `reachable = {d1}`.
    # For `d2` in `diffs[1:]`:
    #    `new_reachable = set(reachable)`
    #    For `s` in `reachable`:
    #        `new_reachable.add(s + d2)`
    #    `reachable = new_reachable`
    # At the end, check if `0 in reachable`.
    # This excludes the empty set!
    # This is $O(N \cdot |R|)$.
    # $|R|$ is bounded by the range of sums.
    # This is perfect.

def can_add(a, others):
    diffs = [y - a for y in others]
    if not diffs:
        return True
        
    # Start with the first diff
    # If the first diff is 0, we have a violation (size 1). But y!=a, so diff!=0.
    reachable = {diffs[0]}
    
    for d in diffs[1:]:
        # Optimization: if 0 is already in reachable, we can return True early?
        # No, we need to process all to be sure? 
        # Actually, if 0 is in reachable at any step, it means a subset of the processed elements sums to 0.
        # So we can return True immediately.
        if 0 in reachable:
            return True
            
        new_sums = set()
        for s in reachable:
            new_sums.add(s + d)
        reachable = reachable.union(new_sums)
        
        # Early termination if 0 is found
        if 0 in reachable:
            return True
            
    return 0 in reachable

def construct():
    start_time = time.time()
    deadline = start_time + 50 # 50s safety margin
    
    # Greedy construction
    selected = []
    # Try to add elements in a way that maximizes count.
    # Randomized restarts?
    
    # Strategy: 
    # 1. Try to build a large set greedily.
    # 2. If stuck, try local search.
    
    # Let's try a simple greedy first.
    # Order: 1 to 512.
    # This might be biased.
    # Better: Random order, multiple attempts.
    
    best_set = []
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Random permutation of 1..512
        candidates = list(range(1, 513))
        import random
        random.shuffle(candidates)
        
        current_set = []
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Check if x can be added
            # 1. x is not a mean of current_set
            if can_add(x, current_set):
                # 2. No y in current_set is a mean of current_set + {x}
                # This requires checking each y in current_set
                # This is expensive: O(N) checks.
                # Each check is O(N * Range).
                # Total O(N^2 * Range).
                # For N=48, this is fine.
                
                valid = True
                for y in current_set:
                    # Check if y is a mean of (current_set \ {y}) U {x}
                    others = [z for z in current_set if z != y] + [x]
                    if can_add(y, others):
                        valid = False
                        break
                        
                if valid:
                    current_set.append(x)
                    
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
