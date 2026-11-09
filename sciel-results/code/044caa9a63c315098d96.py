import time
import random
import sys

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 58  # Keep 2s margin for evaluator overhead

    best_set = [31, 32, 34]
    best_size = len(best_set)

    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        return sums

    def check_divisibility_antichain(non_empty_sums):
        """
        Check if non_empty_sums forms an antichain under divisibility.
        No two distinct values u, v should satisfy u|v or v|u.
        """
        sl = sorted(non_empty_sums)
        n = len(sl)
        for i in range(n):
            u = sl[i]
            for j in range(i + 1, n):
                v = sl[j]
                if v % u == 0:
                    return False
        return True

    def is_valid_set(S):
        """
        Checks if set S is valid.
        1. All subset sums are distinct.
        2. No two distinct subset sums divide one another.
        """
        if not S:
            return True
        
        sums = compute_sums(S)
        non_empty_sums = {s for s in sums if s != 0}
        
        # Check distinctness: should have exactly 2^k - 1 non-empty sums
        if len(non_empty_sums) != (1 << len(S)) - 1:
            return False
            
        # Check divisibility antichain
        if not check_divisibility_antichain(non_empty_sums):
            return False
            
        return True

    def check_validity_incremental(S, x, sums_S):
        """
        Checks if S U {x} is valid, given S is valid.
        S: list of elements
        x: new element
        sums_S: precomputed subset sums of S (including 0)
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        if x in S:
            return False, None
            
        # Generate new sums: x + s for s in sums_S
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness: sums_S and new_sums must be disjoint
        if sums_S & new_sums:
            return False, None
        
        all_sums = sums_S | new_sums
        non_empty_all = {s for s in all_sums if s != 0}
        
        if not check_divisibility_antichain(non_empty_all):
            return False, None
        
        return True, all_sums

    def dfs_superincreasing(current_sum, last_val, current_sums, depth, path):
        """
        Depth-first search for superincreasing sequences.
        Superincreasing: next_val > current_sum.
        This guarantees distinct subset sums.
        """
        nonlocal best_set, best_size
        
        if len(path) > best_size:
            best_size = len(path)
            best_set = path[:]
            
        if time.time() - start_time > time_limit:
            return

        # Try to add the next element
        # next_val must be > current_sum
        # And next_val <= 256
        start_val = current_sum + 1
        if start_val > 256:
            return
            
        # We can try a range of values. 
        # Smaller values keep the sum lower, allowing more elements.
        # Larger values might avoid divisibility conflicts.
        # Let's iterate from start_val to 256.
        # To speed up, we can stop if we've tried enough or found a good one?
        # No, we need to explore.
        
        for next_val in range(start_val, 257):
            if time.time() - start_time > time_limit:
                return
            
            # Check if adding next_val is valid
            # Since it's superincreasing, distinctness is guaranteed.
            # We only need to check divisibility.
            
            new_sums = {next_val + s for s in current_sums}
            all_sums = current_sums | new_sums
            non_empty_all = {s for s in all_sums if s != 0}
            
            if check_divisibility_antichain(non_empty_all):
                # Valid, recurse
                dfs_superincreasing(current_sum + next_val, next_val, all_sums, depth + 1, path + [next_val])
                
                # Optional: Pruning. If we found a solution of size X, 
                # and we know max possible size is Y, we can prune.
                # But max possible is hard to estimate.
                
                # Heuristic: If we found a valid set, do we need to try larger next_vals?
                # Larger next_vals increase the sum, making it harder to add more elements.
                # So if we found a valid path with next_val, trying larger next_vals 
                # is unlikely to yield a *longer* sequence, but might be necessary 
                # if the smaller next_val leads to a dead end later.
                # However, for maximizing length, smaller next_val is generally better 
                # for keeping the sum low.
                # Let's continue searching to be safe, but maybe break if we've found 
                # a very deep solution?
                
    # Search Strategy
    # 1. DFS for superincreasing sequences
    # Start with 0 sum, 0 last_val
    # We can start with any first element 1..256
    # But 1 is always bad (divides all).
    # Let's start from 2.
    
    # To avoid redundant work, we can fix the first element?
    # No, different first elements lead to different branches.
    
    # We'll run DFS for a few starting points.
    # Starting with small numbers allows more elements.
    # Start with 2, 3, 4, 5, 6, 7, 8.
    
    for start_val in range(2, 10):
        if time.time() - start_time > time_limit:
            break
        initial_sums = {0, start_val}
        dfs_superincreasing(start_val, start_val, initial_sums, 1, [start_val])
        
    # 2. Fallback: Original Greedy + Local Search
    random.seed(42)
    num_restarts = 1000
    for attempt in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        S = []
        sums_S = {0}
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            valid, new_sums = check_validity_incremental(S, x, sums_S)
            if valid:
                S.append(x)
                sums_S = new_sums
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]

    # Final verification of best_set
    if best_set and len(best_set) <= 14:
        if not is_valid_set(best_set):
            # Fallback to seed
            best_set = [31, 32, 34]
            best_size = 3
    elif len(best_set) > 14:
        pass

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
