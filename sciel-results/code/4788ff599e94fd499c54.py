import numpy as np

def construct():
    n = 512
    max_size = 48
    
    # We'll use a greedy approach: try to add elements in a strategic order
    # Key insight: larger numbers are "safer" as means because they require larger sums
    # Also, avoid numbers that are easily expressible as averages
    
    # Strategy: Use numbers that are "hard to average" - e.g., numbers with certain properties
    # Let's try a greedy approach with subset-sum verification using bitsets
    
    def is_valid_set(selected, candidate):
        """Check if adding candidate to selected keeps the set valid."""
        new_set = selected + [candidate]
        new_set_set = set(new_set)
        
        for a in new_set:
            others = [x for x in new_set if x != a]
            # Check if a is the mean of any subset of others with size >= 2
            # sum(subset) = len(subset) * a
            # We need to check for k = 2 to len(others)
            # Use subset sum: can we form sum = k*a using k elements from others?
            
            # For efficiency, we only need to check subsets that include the candidate
            # if candidate is new, or check all if a is the candidate
            if a == candidate:
                # Check if candidate is mean of any subset of selected with size >= 2
                others = selected
            else:
                # Check if a is mean of any subset of (selected \ {a}) + [candidate]
                # that includes candidate (subsets without candidate were already valid)
                others = [x for x in selected if x != a] + [candidate]
                # But we only need to check subsets that include candidate
                # Let's just check all subsets for correctness, but optimize later
            
            # Use subset-sum DP to check if any k >= 2 exists such that
            # we can pick k elements from 'others' summing to k*a
            # This is equivalent to: can we pick elements from 'others' 
            # such that sum(x_i - a) = 0 with count >= 2?
            # Or: sum of selected elements from others = k * a for some k >= 2
            
            # Let's do a direct check using bitset subset sum for each k
            # But that might be slow. Alternative: check all subset sums
            
            # For small sets, brute force with optimization
            # Since we're building incrementally, selected is at most 47 elements
            # Let's use a smarter approach
            
            # Check: is there a subset T of others, |T| >= 2, with sum(T) = |T| * a?
            # Equivalent: sum(T \ elements) ... let's just iterate over possible k
            
            others_sorted = sorted(others)
            m = len(others_sorted)
            if m < 2:
                continue
                
            # For each k from 2 to m, check if we can pick k elements summing to k*a
            # Use DP: dp[i][j] = can we pick j elements from first i summing to target?
            # But target depends on k, so we need separate checks
            
            # Alternative: generate all subset sums with their sizes
            # Since m <= 48, 2^48 is too large, but we can use meet-in-the-middle or DP
            
            # Let's use a DP that tracks (count, sum) pairs
            # dp[count] = set of possible sums using 'count' elements
            # But sums can be large (up to 48*512 = 24576)
            
            # For efficiency, let's use a different approach:
            # For a fixed a, we need sum(T) - |T|*a = 0, i.e., sum(x_i - a) = 0
            # Let y_i = x_i - a for x_i in others
            # We need a subset of y's (size >= 2) that sums to 0
            
            # This is the zero-sum subset problem
            # We can use DP: dp[s] = min/max count to achieve sum s, or just track achievable sums
            
            # Given the constraints, let's use a bitset approach for subset sums
            # with a twist: we track the count
            
            # Actually, for the greedy approach, let's just do a reasonable check
            # Since we're adding one element at a time, the set is small during construction
            
            # Simple approach: for each k from 2 to min(m, 10) or so, check subset sum
            # But we need to check all k up to m
            
            # Let's use a DP where dp[j] is a bitset of achievable sums using j elements
            # Sums range from min(others)*j to max(others)*j, but we can bound it
            
            # For now, let's do a simpler check: only check k=2,3,4,5 and hope that's sufficient
            # This is NOT correct but might work in practice for a greedy heuristic
            
            # Actually, let me implement a proper but optimized check
            # For each k, target = k * a
            # Check if target can be formed by summing k distinct elements from others
            
            # Use DP: reachable[j] = set of sums achievable with j elements
            # But storing sets is expensive. Use bitset if sums are bounded.
            
            # Max sum with k elements: k * max(others) <= k * 512
            # Min sum: k * min(others) >= k * 1
            # For k up to 48, max sum ~ 24576, which is manageable with bitset
            
            max_sum_bound = k * 512 if False else m * 512  # use m*512 as global bound
            # Actually, let's use a dictionary-based DP for clarity and correctness
            
            # dp[count] = set of sums
            dp = [set() for _ in range(m + 1)]
            dp[0].add(0)
            
            for x in others_sorted:
                for j in range(m - 1, -1, -1):
                    if dp[j]:
                        for s in list(dp[j]):
                            dp[j + 1].add(s + x)
            
            for k in range(2, m + 1):
                target = k * a
                if target in dp[k]:
                    return False
        
        return True
    
    # Greedy: try to build a large set
    # Order: try numbers that are "hard to be means" first
    # Numbers that are prime or have few divisors might be safer
    # Or just try in descending order (larger numbers are harder to be averages)
    
    # Let's try descending order first
    candidates = list(range(n, 0, -1))
    
    selected = []
    for x in candidates:
        if len(selected) >= max_size:
            break
        if is_valid_set(selected, x):
            selected.append(x)
    
    if len(selected) >= max_size:
        return selected[:max_size]
    
    # If we didn't reach max_size, try a different order
    # Try ascending order
    selected = []
    for x in range(1, n + 1):
        if len(selected) >= max_size:
            break
        if is_valid_set(selected, x):
            selected.append(x)
    
    if len(selected) >= max_size:
        return selected[:max_size]
    
    # Try a mixed approach: interleave
    selected = []
    # Start with some large numbers, then fill in
    for x in list(range(n, 0, -2)) + list(range(n - 1, 0, -2)):
        if len(selected) >= max_size:
            break
        if is_valid_set(selected, x):
            selected.append(x)
    
    return selected[:max_size]
