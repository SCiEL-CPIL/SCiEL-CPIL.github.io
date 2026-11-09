import time
import math

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    n = 256
    # Perfect squares up to 512
    squares = set()
    for k in range(1, 23):
        squares.add(k * k)
    
    # Build conflict graph: conflict[i] is a bitmask of vertices conflicting with i
    # Include self-conflicts (2*i is a square)
    conflict = [0] * n
    
    for i in range(n):
        val = i + 1
        # Self-conflict: 2*val is a perfect square
        if 2 * val in squares:
            conflict[i] |= (1 << i)
        # Conflicts with other vertices j > i
        for j in range(i + 1, n):
            if val + (j + 1) in squares:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Vertices with self-conflict can never be in the solution
    available = 0
    for i in range(n):
        if not (conflict[i] & (1 << i)):
            available |= (1 << i)
    
    # For branch-and-bound MIS, we work on the available vertices
    # Re-index to compact representation
    idx_map = {}
    compact = []
    for i in range(n):
        if available & (1 << i):
            idx_map[i] = len(compact)
            compact.append(i)
    
    m = len(compact)
    
    if m == 0:
        return [1]
    
    # Build compact conflict bitmasks
    cconflict = [0] * m
    for i in range(m):
        orig_i = compact[i]
        mask = conflict[orig_i] & available
        # Convert to compact indices
        cmask = 0
        for j in range(m):
            if mask & (1 << compact[j]):
                cmask |= (1 << j)
        cconflict[i] = cmask
    
    # Maximum Independent Set via Branch and Bound
    # Using Tomita's algorithm for maximum clique on the complement graph
    # MIS in G = Max Clique in complement of G
    
    # Complement adjacency: comp_adj[i] has bits set for vertices NOT conflicting with i (excluding i)
    comp_adj = [0] * m
    all_mask = (1 << m) - 1
    for i in range(m):
        # In complement, i is adjacent to j if i and j are NOT in conflict (and i != j)
        # comp_adj[i] = all_mask ^ (1<<i) ^ cconflict[i]
        comp_adj[i] = all_mask & ~(1 << i) & ~cconflict[i]
    
    # Tomita's maximum clique algorithm with coloring
    best_clique = []
    best_size = 0
    
    # Precompute coloring order for initial coloring
    # We'll use a greedy coloring on the remaining candidates
    
    def color_sort(candidates, comp_adj):
        """Color the candidates and return them sorted by color (descending)."""
        nonlocal m
        uncolored = candidates
        colors = [0] * m
        color_idx = 0
        ordered = []
        
        while uncolored:
            color_idx += 1
            # Greedily pick a vertex from uncolored
            # Pick the one with highest degree in uncolored subgraph for better pruning
            best_v = -1
            best_deg = -1
            for v in range(m):
                if uncolored & (1 << v):
                    deg = bin(comp_adj[v] & uncolored).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            if best_v < 0:
                break
            uncolored &= ~(1 << best_v)
            colors[best_v] = color_idx
            ordered.append(best_v)
        
        # Sort by color descending
        ordered.sort(key=lambda v: colors[v], reverse=True)
        return ordered, colors
    
    def max_clique_recursive(candidates, current_clique, depth):
        nonlocal best_clique, best_size, deadline, m, comp_adj
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
        
        # Color sort
        ordered, colors = color_sort(candidates, comp_adj)
        
        # Process in order
        remaining = candidates
        for v in ordered:
            if not (remaining & (1 << v)):
                continue
            if time.time() > deadline:
                return
            
            # Pruning: if color[v] + len(current_clique) <= best_size, skip
            if colors[v] + len(current_clique) <= best_size:
                # No need to explore further (all subsequent have lower or equal color)
                break
            
            current_clique.append(v)
            # New candidates: intersection of remaining and neighbors of v
            new_candidates = remaining & comp_adj[v]
            max_clique_recursive(new_candidates, current_clique, depth + 1)
            current_clique.pop()
            remaining &= ~(1 << v)
    
    # Initial coloring for the full set
    initial_ordered, initial_colors = color_sort(all_mask, comp_adj)
    
    # Start recursion
    max_clique_recursive(all_mask, [], 0)
    
    # Convert back to original values
    if best_clique:
        result = [compact[v] + 1 for v in best_clique]
    else:
        # Fallback: just pick one valid vertex
        result = [compact[0] + 1]
    
    return result
