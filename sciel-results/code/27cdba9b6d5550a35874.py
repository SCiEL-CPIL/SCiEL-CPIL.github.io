import time

def construct():
    start_time = time.time()
    
    def find_subset_sum(target, S):
        """Find one subset of S summing to target. Return list or None."""
        items = sorted(S)
        dp = [False] * (target + 1)
        dp[0] = True
        back = [None] * (target + 1)
        
        for item in items:
            for s in range(target, item - 1, -1):
                if dp[s - item] and not dp[s]:
                    dp[s] = True
                    back[s] = (s - item, item)
        
        if not dp[target]:
            return None
        
        subset = []
        cur = target
        while cur > 0:
            prev, item = back[cur]
            subset.append(item)
            cur = prev
        return subset

    def solve_with_heuristic(S, strategy):
        S = set(S)
        while True:
            if time.time() - start_time > 85:
                break
            bad = find_subset_sum(200, S)
            if bad is None:
                break
            if strategy == 'max':
                x = max(bad)
            elif strategy == 'min':
                x = min(bad)
            elif strategy == 'median':
                bad_sorted = sorted(bad)
                x = bad_sorted[len(bad_sorted)//2]
            else:
                x = max(bad)
            S.remove(x)
        return list(S)

    # Try different heuristics
    base_set = set(range(1, 101))
    
    candidates = []
    
    # Heuristic 1: Remove Largest
    c1 = solve_with_heuristic(base_set, 'max')
    candidates.append(c1)
    
    # Heuristic 2: Remove Smallest
    c2 = solve_with_heuristic(base_set, 'min')
    candidates.append(c2)
    
    # Heuristic 3: Remove Median
    c3 = solve_with_heuristic(base_set, 'median')
    candidates.append(c3)
    
    # Heuristic 4: Greedy Descending (Seed)
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    candidates.append(selected)
    
    # Heuristic 5: Greedy Ascending
    selected, reachable = [], {0}
    for x in range(1, 101):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    candidates.append(selected)
    
    # Pick the best
    best = max(candidates, key=len)
    
    return best
