def construct():
    """
    Finds a maximum cardinality subset of [1, 100] with no subset summing to 200.
    
    Approach:
    Models the problem as a constraint satisfaction where we iteratively remove
    elements that participate in forbidden sums. We use a greedy heuristic that
    removes the "most critical" element (smallest value, proxy for highest
    degree in the conflict hypergraph) from any found forbidden subset.
    """
    TARGET = 200
    # Current set of available vertices
    current_set = set(range(1, 101))
    
    def find_forbidden_subset(S):
        """
        Finds any subset of S that sums to TARGET.
        Returns the subset as a list, or None if no such subset exists.
        Uses dynamic programming with path reconstruction.
        """
        # dp[s] = True if sum s is achievable
        # parent[s] = (prev_sum, item) to reconstruct
        dp = [False] * (TARGET + 1)
        dp[0] = True
        parent = [None] * (TARGET + 1)
        
        # We need to be careful with reconstruction.
        # Standard 0/1 knapsack DP.
        # To reconstruct, we iterate through items and update dp.
        # We store the last item that made a state reachable.
        
        # Optimization: Only consider items in S
        items = list(S)
        
        # To allow reconstruction, we can store the item that led to each state.
        # However, since we remove items one by one, the set S changes.
        # We just need *one* valid subset.
        
        # We'll use a dictionary or list to track the last item added for each sum.
        # last_item[s] = item i such that dp[s] became True when adding i.
        last_item = [None] * (TARGET + 1)
        
        # Iterate through items
        for item in items:
            # Update dp from high to low to ensure each item used at most once
            # We only care about sums up to TARGET
            for s in range(TARGET, item - 1, -1):
                if dp[s - item] and not dp[s]:
                    dp[s] = True
                    last_item[s] = item
                    # Note: This simple backtracking doesn't work perfectly for multiple items
                    # because dp[s-item] might have been updated in the same iteration?
                    # No, we go high to low, so dp[s-item] is from previous iteration (or earlier items).
                    # Wait, if we just set dp[s]=True, we don't know the *full* path yet.
                    # We can reconstruct by walking back:
                    # current_sum = TARGET
                    # item = last_item[current_sum]
                    # prev_sum = current_sum - item
                    # item = last_item[prev_sum] ...
                    # This works if last_item[prev_sum] was set by an *earlier* item.
                    # Since we process items one by one, and update s from high to low,
                    # dp[s-item] is either from previous items or 0.
                    # So the chain of last_items will refer to distinct items.
        
        if not dp[TARGET]:
            return None
            
        # Reconstruct the subset
        subset = []
        current_sum = TARGET
        while current_sum > 0:
            item = last_item[current_sum]
            if item is None:
                # Should not happen if dp[TARGET] is True
                break
            subset.append(item)
            current_sum -= item
            
        return subset

    # Greedy removal loop
    while True:
        forbidden = find_forbidden_subset(current_set)
        if forbidden is None:
            break
        
        # Heuristic: Remove the smallest element from the forbidden subset.
        # Small elements are more "versatile" and appear in more sum combinations,
        # effectively having a higher degree in the conflict hypergraph.
        # Removing them clears many potential conflicts.
        min_elem = min(forbidden)
        current_set.remove(min_elem)
        
    return list(current_set)
