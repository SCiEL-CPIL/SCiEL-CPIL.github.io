"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    
    N = 512
    nums = list(range(1, N + 1))
    
    # Build conflict graph: edge between a,b if lcm(a,b) > 512
    # lcm(a,b) = a*b // gcd(a,b)
    n = len(nums)
    # Use bitmasks for adjacency
    adj = [0] * n  # adj[i] is bitmask of nodes conflicting with i
    
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Branch and bound for maximum independent set
    # Use greedy coloring for upper bound
    import sys
    sys.setrecursionlimit(100000)
    
    best_set = [0]  # at least {1}
    best_size = 1
    
    # Order nodes by degree (descending) for better pruning
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    # Map from original index to position in order
    pos_in_order = [0] * n
    for idx, orig in enumerate(order):
        pos_in_order[orig] = idx
    
    # Rebuild adjacency in new order
    new_adj = [0] * n
    for new_i in range(n):
        orig_i = order[new_i]
        mask = 0
        for new_j in range(new_i + 1, n):
            orig_j = order[new_j]
            if adj[orig_i] & (1 << orig_j):
                mask |= (1 << new_j)
        new_adj[new_i] = mask
    
    # Greedy coloring for upper bound on remaining nodes
    def greedy_color_bound(candidates):
        """Return upper bound on independent set size from candidates (bitmask)."""
        if candidates == 0:
            return 0
        # Greedy coloring: assign colors greedily
        color_classes = []
        remaining = candidates
        while remaining:
            # Pick a maximal independent set greedily
            color_class = 0
            rem = remaining
            while rem:
                # Pick lowest set bit
                lb = rem & (-rem)
                idx = lb.bit_length() - 1
                color_class |= lb
                # Remove idx and all its neighbors from rem
                rem &= ~new_adj[idx]
                rem &= ~lb
            color_classes.append(color_class)
            remaining &= ~color_class
        return len(color_classes)
    
    # Branch and bound
    def bbs(candidates, current_set, current_size):
        nonlocal best_set, best_size
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound
        ub = current_size + greedy_color_bound(candidates)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on (first in order)
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        
        # Branch 1: include v
        new_candidates = candidates & ~lb & ~new_adj[v]
        current_set.append(v)
        bbs(new_candidates, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: exclude v
        bbs(candidates & ~lb, current_set, current_size)
    
    # Start with all nodes
    all_candidates = (1 << n) - 1
    bbs(all_candidates, [], 0)
    
    # Convert back to original values
    result = [nums[order[i]] for i in best_set]
    return result
# EVOLVE-BLOCK-END
