"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _check_valid(S, target=200):
    """Check if no subset of S sums to target using bitset DP."""
    dp = 1  # bit i set means sum i is achievable
    for x in S:
        dp |= (dp << x)
        if (dp >> target) & 1:
            return False
    return True

def construct():
    """
    Returns a nonempty subset of [1, 100] with no subset summing to 200,
    maximizing cardinality.
    
    Strategy: 
    1. Start with a large valid set (e.g., all numbers > 100/2).
    2. Greedily try to add elements from the lower half.
    3. If adding an element causes a conflict, try to remove a minimal set of 
       elements to resolve the conflict, preferring to remove elements that 
       "block" many other additions or are less "valuable".
    4. Perform local search with swaps to try to increase size.
    """
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Initial set: {66, ..., 100} has size 35.
    # Any two elements sum to at least 66+67=133 and at most 199.
    # No single element is 200.
    # So no subset sums to 200.
    best_set = list(range(66, 101))
    best_size = len(best_set)
    
    # Candidate elements to add: [1, 65]
    candidates = list(range(1, 66))
    
    # Pre-compute conflicts? Or just check on the fly.
    # Checking on the fly with bitset DP is fast (O(N * Target / word_size)).
    # Target 200 is small.
    
    # Greedy Add Phase
    # We try to add candidates in a specific order.
    # Smaller numbers are "more dangerous" because they can participate in more sums?
    # Or larger numbers in the lower half might be safer?
    # Let's try adding in descending order of "safety" or just iterate.
    # Actually, let's try to add elements that have the fewest conflicts with the current set.
    
    # To speed up conflict detection, we can check if adding x creates a subset sum 200.
    # This is equivalent to: does S contain a subset summing to 200-x?
    # We can maintain the bitset of S.
    
    # Let's implement a helper to check if adding x is valid given current bitset dp_S
    def can_add(x, dp_S, target=200):
        # Check if (dp_S | (dp_S << x)) has bit target set
        # This is equivalent to checking if dp_S has bit (target-x) set, 
        # OR if x itself is target (not possible here), 
        # OR if some subset of S sums to target-x.
        # Wait, the new subset could be just {x} (if x==target, no) or {x} U {subset of S}.
        # So we need to check if (target - x) is achievable by S.
        # Also, we must ensure that x is not already in S (it isn't, candidates are disjoint).
        if x >= target:
            return False # x alone would be target (if x==200) or > target (not possible here)
        # Check if target-x is in dp_S
        return ((dp_S >> (target - x)) & 1) == 0
    
    # Initialize dp for best_set
    dp_best = 1
    for x in best_set:
        dp_best |= (dp_best << x)
        
    # We want to add as many candidates as possible.
    # If we can't add x directly, we might remove some elements from best_set to make room.
    # This is a hitting set problem / conflict resolution.
    
    # Let's try a simple local search / hill climbing approach.
    # State: current set S.
    # Move: Add x (if valid). If invalid, try to remove a minimal set R from S such that S \ R U {x} is valid.
    # To keep it simple and fast, let's try removing 1 element first.
    
    # We will iterate through candidates.
    # To make it more effective, we can shuffle candidates or order them by "impact".
    # Let's try adding candidates in random order or sorted order.
    # Sorted descending might be better? Large numbers in [1,65] are "closer" to 100, so maybe they conflict less?
    # Actually, small numbers conflict with more pairs in S.
    # Let's try adding in ascending order first, as it's a common heuristic for subset sum (fill small gaps).
    
    # However, the current seed program's greedy approach failed to add anything.
    # This suggests that for S={66..100}, every x in [1,65] conflicts.
    # Let's verify:
    # x=1: need 199 in S. 99+100=199. Yes.
    # x=2: need 198 in S. 98+100=198. Yes.
    # ...
    # x=34: need 166 in S. 66+100=166. Yes.
    # x=35: need 165 in S. 66+99=165. Yes.
    # x=65: need 135 in S. 66+69=135. Yes.
    
    # So we MUST remove elements to add any new ones.
    
    # Strategy:
    # 1. Try to add x.
    # 2. If fails, identify a conflicting subset C in S that sums to 200-x.
    # 3. Try to remove one element from C. If S \ {y} U {x} is valid, accept.
    # 4. If that works, we maintain size. But we want to INCREASE size.
    # 5. So, after a swap (size constant), we try to add another element.
    # 6. If we can add two elements while removing one, size increases.
    
    # Let's implement a loop that attempts to add candidates, allowing swaps.
    
    # We'll keep a set of "blocked" candidates or just iterate repeatedly.
    # To avoid infinite loops, we'll track the best size and only accept moves that increase size,
    # or moves that keep size but "improve" the set (e.g., allow more future additions).
    # Simplest: Accept any valid set with size > best_size.
    # For same size, we can accept if it opens up more options.
    
    # Let's try a different approach:
    # Start with S = {66..100}.
    # Try all x in [1,65].
    # For each x, find if there is a y in S such that (S - {y}) U {x} is valid.
    # If yes, perform the swap.
    # Then try to add another element z.
    
    # This is essentially a local search.
    
    # Let's implement a more robust local search:
    # 1. Current set S.
    # 2. For each x not in S:
    #    a. Try to add x. If valid, S = S U {x}, size++.
    #    b. If invalid, try to remove y from S such that (S - {y}) U {x} is valid.
    #       If found, S = (S - {y}) U {x}. (Size unchanged, but structure changed)
    # 3. Repeat until no improvements or time limit.
    
    # To make step 2b efficient, we can pre-compute or quickly find conflicts.
    # Finding a single y to remove:
    # If adding x fails, it means there is a subset T of S summing to 200-x.
    # If we remove any y in T, the sum 200-x is no longer achievable using T.
    # But other subsets might sum to 200-x.
    # So we need to check if (S - {y}) has a subset summing to 200-x.
    
    # This check is O(|S| * Target).
    # We can do this for each y in S.
    # |S| is ~35. Target is 200. 35 * 200 = 7000 ops. Very fast.
    
    # Let's implement this local search.
    
    S = set(best_set)
    size = len(S)
    
    # Pre-compute dp for S? No, S changes.
    # We'll recompute dp for each candidate move? That might be slow if we do many swaps.
    # But 60s is plenty.
    
    # Helper to get dp for a set
    def get_dp(S_set):
        dp = 1
        for x in S_set:
            dp |= (dp << x)
        return dp
    
    # Initial dp
    dp_S = get_dp(S)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any element from [1,65] not in S
        for x in range(1, 66):
            if time.time() > deadline:
                break
            if x in S:
                continue
            
            # Check if adding x directly is valid
            # Check if 200-x is in dp_S
            if ((dp_S >> (200 - x)) & 1) == 0:
                # Can add directly
                S.add(x)
                size += 1
                dp_S = get_dp(S) # Update dp
                improved = True
                # Restart loop? Or continue?
                # Continuing might miss other additions, but next outer loop will catch them.
                # Breaking inner loop and restarting outer is safer to pick up new opportunities.
                # But for speed, let's just mark improved and continue scanning.
                # Actually, adding x changes dp_S, so subsequent checks in this iteration are invalid.
                # We should break and restart the for-loop over x.
                break
            
            # If direct add fails, try to swap: remove y, add x
            # Find y in S such that (S - {y}) U {x} is valid.
            # This is equivalent to: (S - {y}) has no subset summing to 200-x.
            # Let's iterate y in S.
            found_swap = False
            for y in list(S):
                if time.time() > deadline:
                    break
                # Check if S - {y} has subset sum 200-x
                # We can compute dp for S - {y} quickly?
                # Or check if 200-x is achievable by S - {y}.
                # This is a subset sum query.
                # We can do a quick DP check for S - {y} targeting 200-x.
                # Since 200-x < 200, and elements are positive, this is fast.
                
                # Optimization: If y is not part of any subset summing to 200-x in S,
                # then removing y doesn't help.
                # How to check if y is in a subset summing to 200-x?
                # This is the "subset sum with required element" problem.
                # Can be solved by checking if 200-x-y is achievable by S - {y}.
                # If yes, then y is in a subset summing to 200-x.
                # If no, then y is NOT in any subset summing to 200-x.
                # So we only need to try y if y IS in a subset summing to 200-x.
                
                # Let's check if y is in a subset summing to 200-x.
                # This requires a DP check on S - {y} for target 200-x-y.
                # This is O(|S| * (200-x-y)).
                
                # To save time, we can just try the swap and verify the full set.
                # Full verification is O(|S| * 200).
                # With |S| ~ 35, this is ~7000 ops.
                # Trying 35 swaps is 35 * 7000 = 245,000 ops.
                # Trying 65 candidates is 65 * 245,000 = 15,925,000 ops.
                # This is ~16 million ops per outer loop iteration.
                # Python can do ~10-50 million simple ops per second.
                # So one full pass might take 0.3 - 1.5 seconds.
                # We have 55 seconds. So we can do ~30-100 passes.
                # This should be sufficient to find a better solution.
                
                new_S = S - {y}
                new_S.add(x)
                # Verify new_S
                if _check_valid(new_S, 200):
                    S = new_S
                    # Size is unchanged
                    dp_S = get_dp(S)
                    improved = True
                    found_swap = True
                    break # Found a valid swap, break y-loop
            
            if found_swap:
                # Break x-loop to restart with new S
                break
    
    return list(S)

# EVOLVE-BLOCK-END
