"""Rewritten program using Maximum Independent Set via branch-and-bound with coloring bound."""
import sys

def construct():
    # We need a nonempty subset S of [1, 256] such that for every distinct a, b in S,
    # (a * b) % (a + b) != 0. Maximize |S|.
    
    # Build conflict graph: nodes are 1..256. Edge between a and b if (a*b) % (a+b) == 0.
    # Then we want a Maximum Independent Set (MIS) in this graph.
    
    n = 256
    # adjacency as list of sets for fast lookup
    adj = [set() for _ in range(n)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a - 1].add(b - 1)
                adj[b - 1].add(a - 1)
    
    # Convert to sorted lists for deterministic ordering
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    # Use a branch-and-bound algorithm with the Tomita coloring bound.
    # We'll implement a recursive search that picks vertices in a fixed order,
    # and uses a greedy coloring of the remaining candidate set to get an upper bound.
    
    # For efficiency, we represent the current candidate set as a bitset (Python int).
    # Bit i (0-indexed) corresponds to vertex i (which is value i+1).
    
    # Precompute bit masks for each vertex
    vertex_bit = [1 << i for i in range(n)]
    
    # Precompute for each vertex, the mask of its neighbors
    neighbor_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj_lists[i]:
            mask |= vertex_bit[j]
        neighbor_mask[i] = mask
    
    # Greedy coloring function: given a set of vertices (as a bitmask),
    # assign colors greedily and return the number of colors used.
    # The coloring is an upper bound on the size of an independent set in that subgraph.
    
    def color_bound(candidates_mask):
        """Return the number of colors in a greedy coloring of the induced subgraph on candidates_mask."""
        if candidates_mask == 0:
            return 0
        
        colors = 0
        remaining = candidates_mask
        
        while remaining != 0:
            colors += 1
            # Pick one vertex from remaining, remove it and all its neighbors from remaining
            # For a valid upper bound, we just need any coloring; greedy works.
            # We pick the lowest-set bit for determinism.
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            remaining &= ~v_bit
            remaining &= ~neighbor_mask[v]
        
        return colors
    
    # Branch and bound
    best_set = [0]  # list to hold best independent set size
    best_vertices = [0]  # list to hold best independent set vertices
    
    def search(candidates_mask, current_set, current_size):
        """
        candidates_mask: bitmask of vertices still available to add
        current_set: list of vertices in current independent set
        current_size: len(current_set)
        """
        if candidates_mask == 0:
            if current_size > best_set[0]:
                best_set[0] = current_size
                best_vertices[0] = list(current_set)
            return
        
        # Upper bound: current_size + color_bound(candidates_mask)
        ub = current_size + color_bound(candidates_mask)
        if ub <= best_set[0]:
            return
        
        # Pick a vertex to branch on. Strategy: pick a vertex with high degree in the candidate subgraph
        # to reduce branching factor. For simplicity, pick the lowest-indexed vertex.
        # A better heuristic: pick vertex with max degree in candidate subgraph.
        
        # Get list of candidate vertices
        candidates = []
        m = candidates_mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            candidates.append(v)
            m &= ~v_bit
        
        if not candidates:
            return
        
        # Heuristic: pick vertex with maximum degree within candidates
        def degree_in_candidates(v):
            # Count neighbors of v that are in candidates_mask
            common = neighbor_mask[v] & candidates_mask
            return common.bit_count()
        
        # Sort candidates by degree descending, then by index ascending for determinism
        candidates.sort(key=lambda v: (-degree_in_candidates(v), v))
        
        # Branch on the first vertex (highest degree)
        v = candidates[0]
        v_bit = vertex_bit[v]
        
        # Branch 1: include v
        # New candidates: candidates_mask minus v and all neighbors of v
        new_candidates = candidates_mask & ~v_bit & ~neighbor_mask[v]
        current_set.append(v)
        search(new_candidates, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: exclude v
        new_candidates = candidates_mask & ~v_bit
        search(new_candidates, current_set, current_size)
    
    # Initial call
    all_mask = (1 << n) - 1
    search(all_mask, [], 0)
    
    # Convert best vertices to values (vertex i -> value i+1)
    result = [v + 1 for v in best_vertices[0]]
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
