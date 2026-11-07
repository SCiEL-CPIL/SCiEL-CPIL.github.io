import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = {tuple(e) for e in P['edges']}
    edges.add((e[1], e[0]) for e in edges) # Just in case, though edges are undirected
    
    # Precompute adjacency matrix for fast clique/independent set checks
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    def is_clique(subset):
        s = list(subset)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if not adj[s[i]][s[j]]:
                    return False
        return True

    def is_independent(subset):
        s = list(subset)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if adj[s[i]][s[j]]:
                    return False
        return True

    # Generate all valid subsets (cliques and independent sets)
    valid_subsets = []
    for mask in range(1, 1 << n):
        subset = [i for i in range(n) if mask & (1 << i)]
        if is_clique(subset) or is_independent(subset):
            valid_subsets.append(tuple(subset))
    
    # Sort by size descending to prioritize larger parts
    valid_subsets.sort(key=lambda x: -len(x))

    # Precompute for each vertex, which valid subsets contain it
    # This helps in pruning when we pick a specific vertex
    subsets_by_vertex = [[] for _ in range(n)]
    for s in valid_subsets:
        for v in s:
            subsets_by_vertex[v].append(s)

    # Memoization for search
    memo = {}
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s out of 60s limit

    def can_cover(remaining, k):
        if time.time() > deadline:
            return False
        if k == 0:
            return len(remaining) == 0
        if len(remaining) == 0:
            return True
        if k == 1:
            rem_tuple = tuple(sorted(remaining))
            return rem_tuple in valid_subsets_set # Need to create this set
        key = (frozenset(remaining), k)
        if key in memo:
            return memo[key]
        
        # Pick a vertex from remaining to anchor the search
        # Heuristic: pick vertex with fewest valid subsets covering it
        v = min(remaining, key=lambda x: len([s for s in subsets_by_vertex[x] if set(s).issubset(remaining)]))
        
        found = False
        # Try all valid subsets containing v and subset of remaining
        for s in subsets_by_vertex[v]:
            if set(s).issubset(remaining):
                new_remaining = frozenset(remaining - set(s))
                if can_cover(new_remaining, k - 1):
                    memo[key] = True
                    return True
        memo[key] = False
        return False

    # Create a set of valid subsets for O(1) lookup in base case
    valid_subsets_set = set(valid_subsets)

    # Find minimum k
    min_k = n
    for k in range(1, n + 1):
        if can_cover(frozenset(range(n)), k):
            min_k = k
            break

    # Reconstruct the partition
    # We need to store the choices. Let's modify can_cover to store the path or do a second pass.
    # Easier: run a similar search that records the chosen subset.
    reconstruction_memo = {}
    
    def get_partition(remaining, k):
        if time.time() > deadline:
            return None
        if k == 0:
            if len(remaining) == 0:
                return []
            return None
        if len(remaining) == 0:
            return []
        if k == 1:
            rem_tuple = tuple(sorted(remaining))
            if rem_tuple in valid_subsets_set:
                return [list(rem_tuple)]
            return None
            
        key = (frozenset(remaining), k)
        if key in reconstruction_memo:
            return reconstruction_memo[key]
        
        v = min(remaining, key=lambda x: len([s for s in subsets_by_vertex[x] if set(s).issubset(remaining)]))
        
        for s in subsets_by_vertex[v]:
            if set(s).issubset(remaining):
                new_remaining = frozenset(remaining - set(s))
                sub_partition = get_partition(new_remaining, k - 1)
                if sub_partition is not None:
                    partition = [list(s)] + sub_partition
                    reconstruction_memo[key] = partition
                    return partition
        reconstruction_memo[key] = None
        return None

    partition = get_partition(frozenset(range(n)), min_k)
    
    # Fallback if reconstruction failed or deadline exceeded
    if partition is None:
        # Fallback to greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in edges for i, u in enumerate(trial) for w in trial[i + 1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        partition = parts

    # Ensure types are correct: list of lists of ints
    final_partition = []
    for part in partition:
        final_partition.append([int(v) for v in part])
        
    return final_partition

# EVOLVE-BLOCK-END
