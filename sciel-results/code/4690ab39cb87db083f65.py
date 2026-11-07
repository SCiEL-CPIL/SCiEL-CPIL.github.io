import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Fallback known valid set (powers of 2)
    fallback = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Bitset-based DP for fast validity checking and subset sum computation
    # We represent sets of sums as Python integers (bitsets).
    # dp[c] is a bitset where bit s is 1 if sum s is achievable with exactly c elements.
    
    def compute_bitset_sums(elements):
        n = len(elements)
        max_sum = sum(elements)
        # dp[c] is an integer representing the set of sums for cardinality c
        dp = [0] * (n + 1)
        dp[0] = 1  # Sum 0 is achievable with 0 elements
        
        for x in elements:
            # Iterate backwards to avoid using x multiple times
            for c in range(n - 1, -1, -1):
                # New sums for c+1 are old sums for c shifted by x
                # We need to mask to ensure we don't exceed max_sum or interfere incorrectly
                # Actually, since we're just adding, we can just shift.
                # But we must ensure we don't pick up bits that were already set in dp[c+1] from previous iterations?
                # No, standard 0/1 knapsack bitset logic:
                # dp_new[c+1] = dp_old[c+1] | (dp_old[c] << x)
                # Since we iterate backwards, dp[c] is still "old" (before x was added to lower cardinalities).
                # Wait, if we update dp[c+1] in place, we must be careful.
                # dp[c+1] |= (dp[c] << x)
                # This is correct because dp[c] has not been modified in this iteration of x (since we go down from n-1 to 0).
                dp[c + 1] |= (dp[c] << x)
        
        return dp

    def is_valid(elements):
        """
        Check if the set of elements satisfies the cardinality-separated subset sum condition.
        Returns True if valid, False otherwise.
        """
        if not elements:
            return True
        
        # Sort to ensure consistent behavior, though not strictly required for validity
        # elements = sorted(elements)
        
        dp = compute_bitset_sums(elements)
        
        # Check for overlap between different cardinalities
        # We want to ensure that for any c1 != c2, (dp[c1] & dp[c2]) == 0
        # To do this efficiently, we can accumulate a "used" bitset.
        used = 0
        for c in range(len(elements) + 1):
            if (dp[c] & used) != 0:
                return False
            used |= dp[c]
        
        return True

    # Strategy:
    # 1. Try to find a valid set of size K, starting from K=9 up to a reasonable limit (e.g., 15).
    # 2. For each K, use backtracking search with randomized candidate selection.
    # 3. To speed up backtracking, we use a "pruning" check: if the current partial set
    #    already violates the condition, backtrack.
    #    Additionally, we can check if the remaining candidates can possibly fill the set.
    
    # Pre-compute all candidates 1..256
    all_candidates = list(range(1, 257))
    
    # We will try to construct sets of size K.
    # We start with the best known size (8) and try to extend.
    # However, simple extension might fail. So we restart the search for size K from scratch
    # but with a bias towards small numbers (which are more likely to fit).
    
    best_set = fallback
    best_size = len(fallback)
    
    # Helper for backtracking
    def backtrack_search(current_set, remaining_candidates, target_size, rng, time_limit):
        """
        Attempt to find a valid set of size target_size starting from current_set.
        remaining_candidates is a list of available integers.
        Returns the found set or None.
        """
        if len(current_set) == target_size:
            return current_set[:]
        
        if time.time() > time_limit:
            return None
        
        # If we can't reach the target size with remaining candidates, fail
        if len(current_set) + len(remaining_candidates) < target_size:
            return None
            
        # Shuffle remaining candidates to introduce randomness
        # We want to try smaller numbers first generally, but randomize to escape local minima
        # Let's sort by value, then randomize slightly? Or just shuffle.
        # Shuffling completely might pick large numbers early, leading to large sums and quick failure.
        # Let's use a weighted shuffle or just try sorted order with random restarts.
        
        # For efficiency, we'll iterate through candidates.
        # We can prune candidates that are "too large" if they cause immediate conflicts?
        # The is_valid check is O(N^2) in bit operations, which is fast.
        
        candidates = remaining_candidates[:]
        # Sort candidates. Smaller numbers are generally "safer" for packing.
        # But to find diverse solutions, we can randomize the order.
        # Let's try a mix: 50% sorted, 50% shuffled.
        if rng.random() < 0.5:
            candidates.sort()
        else:
            rng.shuffle(candidates)
        
        for i, x in enumerate(candidates):
            new_set = current_set + [x]
            
            # Quick check: is new_set valid?
            if is_valid(new_set):
                # Recurse
                result = backtrack_search(new_set, candidates[i+1:], target_size, rng, time_limit)
                if result is not None:
                    return result
            
            # Optional: Pruning based on sum range?
            # If the minimum possible sum for the next cardinality overlaps with existing sums, 
            # it's hard to check without full DP. The is_valid call is fast enough.
            
        return None

    # Main loop: Try to find sets of increasing size
    # We start from size 9. If we find a size 9, we try size 10, etc.
    # We give each size a time budget.
    
    # To improve success rate, we can use the "best" set found so far as a seed for extension?
    # No, backtracking from scratch with randomization is more robust for finding *any* valid set of size K.
    # However, we can bias the candidate pool.
    
    # Let's allocate time.
    # If we have 80 seconds, we can try multiple restarts for each K.
    
    current_target = best_size + 1
    max_target = 15  # Upper bound guess. 2^15 is manageable for DP, but search space is huge.
    
    while current_target <= max_target:
        # Time budget for this target size
        # If we've found a better set, we can spend less time on larger sizes?
        # Let's give each target size a fixed number of restarts or time slice.
        
        time_budget_for_k = 5.0  # 5 seconds per cardinality attempt
        restarts = 0
        found = False
        
        while time.time() < start_time + deadline - 5.0: # Keep 5s reserve
            if time.time() > start_time + deadline - time_budget_for_k:
                break
                
            rng = random.Random(int(time.time() * 1000) + current_target + restarts)
            
            # To make search faster, we can restrict candidates to a subset?
            # If we pick numbers > 128, sums grow very fast.
            # Let's try full range first.
            
            result = backtrack_search([], all_candidates, current_target, rng, start_time + deadline - 2.0)
            
            if result is not None:
                # Verify one last time
                if is_valid(result):
                    best_set = result
                    best_size = len(result)
                    found = True
                    break
                else:
                    # Should not happen if is_valid is correct, but safety
                    pass
            
            restarts += 1
            # If we've done enough restarts, move to next K
            if restarts > 50: # 50 restarts of 5s is too long, backtrack_search has internal time check
                 break
        
        if found:
            current_target += 1
        else:
            # If we couldn't find size K, it's unlikely we'll find size K+1 easily?
            # Not necessarily, but we can try one more time with a different strategy or stop.
            # Given the difficulty, if we fail K=9, K=10 is very hard.
            # Let's try K+1 anyway if we have time, but reduce budget.
            if current_target > 10:
                break
            current_target += 1
            time_budget_for_k = 2.0 # Reduce budget for larger K
    
    # Final verification
    if not is_valid(best_set):
        best_set = fallback
    
    return [int(x) for x in best_set]

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Result: {s}")
    print(f"Size: {len(s)}")
