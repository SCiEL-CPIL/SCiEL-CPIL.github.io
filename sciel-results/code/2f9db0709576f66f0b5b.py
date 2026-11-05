<<<<<<< SEARCH
"""Construct C4-free graph with no independent 11-set; maximize n."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _verify_c4_free(n, edges):
    """Return True if graph has no C4 (no two vertices with >=2 common neighbors)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return False
    return True

def _has_independent_set_of_size_k(n, adj, k):
    """Backtracking search for independent set of size k. Returns True if found."""
    # Use greedy coloring bound for pruning (Bron–Kerbosch style)
    # Simple backtracking with ordering
    best = [0]  # current max independent set size found so far (for early termination)
    
    def backtrack(remaining, current_size, current_set):
        if current_size + len(remaining) <= current_size:  # can't beat
            return False
        if current_size == k:
            return True
        # If even taking all remaining can't reach k
        if current_size + len(remaining) < k:
            return False
        # Try greedy: pick vertex with min degree in remaining
        # For speed, just iterate
        for idx in range(len(remaining)):
            v = remaining[idx]
            # Check if v is independent of current_set
            if all(v not in adj[u] for u in current_set):
                new_remaining = [x for x in remaining[idx+1:] if x not in adj[v]]
                if backtrack(new_remaining, current_size + 1, current_set + [v]):
                    return True
        return False
    
    return backtrack(list(range(n)), 0, [])
=======
"""Construct C4-free graph with no independent 11-set; maximize n."""
import time
from itertools import combinations
import random

# EVOLVE-BLOCK-START

def _verify_c4_free(n, edges):
    """Return True if graph has no C4 (no two vertices with >=2 common neighbors)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return False
    return True

def _has_independent_set_of_size_k(n, adj_list, k, deadline=None):
    """
    Fast bitset-based branch-and-bound for independent set of size k.
    adj_list: list of sets of neighbors for each vertex.
    Returns True if an independent set of size k exists.
    """
    # Build bitmasks for adjacency
    adj_masks = [0] * n
    for v in range(n):
        mask = 0
        for u in adj_list[v]:
            mask |= (1 << u)
        adj_masks[v] = mask
    
    # Precompute: for each vertex, the set of non-neighbors (including itself)
    # non_adj[v] = bitmask of vertices not adjacent to v (excluding v itself)
    all_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        # Non-neighbors of v: all vertices except v and its neighbors
        non_adj[v] = all_mask & ~(adj_masks[v] | (1 << v))
    
    def branch(candidates, depth, current_size):
        """
        candidates: bitmask of candidate vertices
        depth: recursion depth
        current_size: size of current independent set
        Returns True if we can find independent set of size k.
        """
        if current_size == k:
            return True
        
        if deadline is not None and time.time() > deadline:
            return False
        
        # Count candidates
        if candidates == 0:
            return False
        
        # Upper bound: current_size + popcount(candidates)
        if current_size + bin(candidates).count('1') < k:
            return False
        
        # Greedy coloring upper bound for pruning
        # Color the candidate subgraph to get an upper bound on independent set size
        # Simple greedy: order by degree, assign colors
        cand_list = []
        temp = candidates
        while temp:
            v = (temp & -temp).bit_length() - 1
            cand_list.append(v)
            temp &= temp - 1
        
        # Greedy coloring of the induced subgraph on cand_list
        # Color by ordering vertices by decreasing degree within candidates
        deg_in_cand = {}
        for v in cand_list:
            deg_in_cand[v] = bin(candidates & adj_masks[v]).count('1')
        
        cand_sorted = sorted(cand_list, key=lambda x: -deg_in_cand[x])
        
        # Greedy coloring
        color = {}
        color_count = 0
        for v in cand_sorted:
            # Find smallest color not used by neighbors
            used_colors = set()
            for u in cand_list:
                if u != v and (adj_masks[v] >> u) & 1:
                    if u in color:
                        used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            color_count = max(color_count, c + 1)
        
        # Upper bound: current_size + color_count
        if current_size + color_count < k:
            return False
        
        # Choose pivot vertex: one with max degree in candidates
        pivot = max(cand_list, key=lambda x: bin(candidates & adj_masks[x]).count('1'))
        
        # Branch 1: include pivot
        new_candidates = candidates & non_adj[pivot]
        if branch(new_candidates, depth + 1, current_size + 1):
            return True
        
        # Branch 2: exclude pivot
        new_candidates = candidates & ~(1 << pivot)
        if branch(new_candidates, depth + 1, current_size):
            return True
        
        return False
    
    return branch(all_mask, 0, 0)
>>>>>>> REPLACE

