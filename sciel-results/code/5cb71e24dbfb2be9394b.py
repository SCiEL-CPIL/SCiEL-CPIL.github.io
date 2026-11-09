import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    S = set(range(1, 101))
    
    # Helper to find a subset of S summing to 200
    # Returns the subset as a list, or None
    def find_subset_sum_200(S):
        # DP: dp[s] stores the subset (as a frozenset or tuple) that sums to s
        # To save memory/time, we can store just the last element and backtrack
        # Or since 200 is small, we can store the actual sets if we are careful
        # Better: store parent pointer and element
        
        max_sum = 200
        parent = [-1] * (max_sum + 1)
        val = [0] * (max_sum + 1)
        parent[0] = -2 # Base case marker
        
        # We need to iterate through elements of S
        # To ensure we find a valid subset, order doesn't matter for existence
        # But for path reconstruction, we need to know which element was added
        
        # Standard 0/1 Knapsack style for subset sum
        # dp[s] = True/False
        # To reconstruct, we can just run a backtracking search on the found sum?
        # No, simpler:
        
        reachable = {0}
        # This is too slow if we do it from scratch every time?
        # |S| is up to 100. 200 sums.
        # 100 * 200 = 20k ops.
        # 100 iterations = 2M ops.
        # Python sets are fast.
        
        # Let's use the set of reachable sums and store the subset?
        # Storing subsets is memory heavy.
        
        # Alternative:
        # Just check if 200 is reachable.
        # If yes, find one such subset using a recursive helper or stored paths.
        
        # Let's store the path.
        # path[s] = (prev_sum, element)
        # Initialize path[0] = None
        
        path = [None] * (max_sum + 1)
        # path[s] is a tuple (prev_sum, elem)
        
        # We need to process elements one by one to allow 0/1 selection
        # But we just need ONE subset.
        # We can use a BFS/DFS on the state space of sums?
        # No, just simple DP with backtracking.
        
        # dp[s] = True if s is reachable
        dp = [False] * (max_sum + 1)
        dp[0] = True
        
        # To allow backtracking, we need to know which element led to dp[s]
        # This is tricky in 1D DP because we overwrite.
        # We can use a list of dicts or just re-run a search.
        
        # Simplest robust method:
        # 1. Check if 200 is reachable using boolean DP.
        # 2. If yes, find the subset using a recursive helper on the original S.
        
        # Step 1: Boolean DP
        for x in S:
            # Update dp in reverse to avoid using x twice
            for s in range(max_sum, x - 1, -1):
                if dp[s - x]:
                    dp[s] = True
            if dp[200]:
                break # Early exit if found? No, we need the subset.
        
        if not dp[200]:
            return None
            
        # Step 2: Find the subset
        # Use a helper function to find a subset summing to 200
        def find_sub(S, target):
            if target == 0:
                return []
            if not S:
                return None
            # Try each element
            # Optimization: sort S?
            for i, x in enumerate(S):
                if x <= target:
                    res = find_sub(S[i+1:], target - x)
                    if res is not None:
                        return [x] + res
            return None
        
        # This recursive call might be slow if S is large and target is large.
        # But target is 200, and we break early.
        # Worst case depth 200. Branching factor 100.
        # But we prune if x > target.
        # And we stop at first solution.
        
        # Better: Use the DP table to guide the search?
        # Or just use the set-based reachable method which is fast for small sums.
        
        # Let's stick to the set-based method for finding the subset, as it's O(|S| * 200)
        # and we can store the subset in a dict: sum -> subset (tuple)
        # Memory: 200 entries, each tuple size <= 200. Small.
        
        subsets = {0: ()}
        for x in S:
            new_subsets = {}
            for s, sub in subsets.items():
                ns = s + x
                if ns <= 200:
                    if ns not in new_subsets:
                        new_subsets[ns] = sub + (x,)
            # Update subsets
            for ns, sub in new_subsets.items():
                if ns not in subsets:
                    subsets[ns] = sub
            if 200 in subsets:
                return list(subsets[200])
        
        return None

    while True:
        if time.time() > deadline:
            break
            
        T = find_subset_sum_200(S)
        if T is None:
            break
            
        # Remove the smallest element from T
        x_to_remove = min(T)
        S.remove(x_to_remove)
        
    return list(S)
