"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations

def construct():
    # Build conflict graph: edge between i and j if i + j == 200
    # (pairwise conflict as a first approximation)
    vertices = list(range(1, 101))
    adj = {v: set() for v in vertices}
    for i in range(1, 101):
        for j in range(i + 1, 101):
            if i + j == 200:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy maximum independent set: pick lowest degree vertex first
    remaining = set(vertices)
    independent_set = []
    
    while remaining:
        # Find vertex with lowest degree in the induced subgraph on remaining
        min_deg = float('inf')
        best_v = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_v = v
        independent_set.append(best_v)
        # Remove best_v and its neighbors from remaining
        to_remove = {best_v} | (adj[best_v] & remaining)
        remaining -= to_remove
    
    # Verify: no subset of independent_set sums to 200
    def has_subset_sum_200(nums):
        # Use DP: reachable sums up to 200
        reachable = {0}
        for x in nums:
            new = {s + x for s in reachable if s + x <= 200}
            if 200 in new:
                return True
            reachable |= new
        return 200 in reachable
    
    if has_subset_sum_200(independent_set):
        # Fallback: use the original greedy from high to low
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
    
    return independent_set
# EVOLVE-BLOCK-END
