"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute which elements are in any subset summing to 200
    # Elements that appear in at least one subset of [1,100] summing to 200
    # Use a greedy minimum hitting set approach:
    # Iteratively find a subset summing to 200 from remaining candidates,
    # remove the element with highest "criticality" from that subset.
    
    candidates = list(range(1, 101))
    excluded = set()
    
    # Helper: find a subset of 'pool' that sums to 200
    def find_subset_summing_to_200(pool):
        # Try to find a subset summing to 200
        # Use a simple DP/backtracking for small sets
        pool = sorted(pool, reverse=True)
        n = len(pool)
        # Try greedy: pick large elements
        # Use a recursive search with pruning
        def dfs(start, remaining, current):
            if remaining == 0:
                return current
            if start >= n or remaining < 0:
                return None
            # Pruning: if sum of all remaining from start < remaining, fail
            suffix_sum = sum(pool[start:])
            if suffix_sum < remaining:
                return None
            # Try including pool[start]
            res = dfs(start + 1, remaining - pool[start], current + [pool[start]])
            if res is not None:
                return res
            # Try excluding
            res = dfs(start + 1, remaining, current)
            return res
        return dfs(0, 200, [])
    
    # Iteratively find forbidden subsets and exclude one element
    # To minimize exclusions, exclude the element that appears in the most remaining forbidden subsets
    # Approximate: exclude the largest element in the found subset (heuristic)
    
    # Better: use a greedy set cover / hitting set
    # Start with all elements, repeatedly find a 200-sum subset, remove one element
    # To maximize final size, we want to remove elements that "cover" many potential 200-sums
    
    # Precompute degree: for each element, how many subsets of [1,100] summing to 200 contain it
    # This is expensive; instead use iterative approach
    
    removed = set()
    remaining = set(range(1, 101))
    
    # Iteratively find a subset summing to 200 from remaining, remove one element
    # Stop when no such subset exists
    while True:
        subset = find_subset_summing_to_200(list(remaining))
        if subset is None:
            break
        # Choose which element to remove: pick the one with highest "degree"
        # Approximate degree by how many other elements in remaining can pair with it
        # in some 200-sum. Use a heuristic: remove the median element of the subset
        # to balance coverage.
        # Actually, remove the element that appears in the most potential 200-sums.
        # Approximate: for each candidate in subset, count how many pairs (a,b) in remaining
        # such that a+b+c=200 for c=candidate.
        best_to_remove = None
        best_score = -1
        for c in subset:
            # Count pairs (a,b) in remaining, a<b, a!=c, b!=c, a+b=200-c
            target = 200 - c
            count = 0
            rem_list = [x for x in remaining if x != c]
            rem_set = set(rem_list)
            for a in rem_list:
                b = target - a
                if b > a and b in rem_set:
                    count += 1
            # Also count triples, etc. - just use pair count as proxy
            score = count
            if score > best_score:
                best_score = score
                best_to_remove = c
        if best_to_remove is None:
            best_to_remove = max(subset)
        remaining.discard(best_to_remove)
    
    return list(remaining)
# EVOLVE-BLOCK-END
