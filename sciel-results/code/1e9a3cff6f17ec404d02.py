# EVOLVE-BLOCK-START
import numpy as np
import time


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum independent set in the Cayley graph on Z/205Z where edges
    connect pairs with a forbidden difference (nonzero square mod 205).
    Uses a bitset-based branch-and-bound with greedy coloring upper bound.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill

    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Build adjacency: for each vertex, which other vertices are adjacent
    # We use bitmasks for efficient set operations
    # adjacency[i] is a bitmask of vertices adjacent to i
    adj = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if (i - j) % m in forbidden:
                adj[i] |= (1 << j)

    # Greedy independent set to get initial lower bound
    def greedy_is():
        remaining = list(range(m))
        result = []
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best = None
            best_deg = m + 1
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and (adj[v] >> u) & 1)
                if deg < best_deg:
                    best_deg = deg
                    best = v
            result.append(best)
            remaining.remove(best)
            # Remove neighbors of best from remaining
            remaining = [u for u in remaining if u != best and not ((adj[best] >> u) & 1)]
        return result

    initial_is = greedy_is()
    best_size = len(initial_is)
    best_set = set(initial_is)

    # Branch and bound using bitmasks
    # We'll use a recursive search with pruning
    
    def branch_and_bound(candidates, current_set, depth=0):
        nonlocal best_size, best_set, start_time
        
        if time.time() > deadline:
            return
        
        if len(candidates) == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Upper bound: len(current_set) + len(candidates)
        if len(current_set) + len(candidates) <= best_size:
            return
        
        # Greedy coloring upper bound on the candidate subgraph
        # Color the candidates graph, number of colors gives upper bound on independent set
        candidates_list = list(candidates)
        
        # Greedy coloring: assign colors to vertices in the candidate subgraph
        colors = {}
        color_count = 0
        for v in candidates_list:
            # Find smallest color not used by neighbors
            neighbor_colors = set()
            for u in candidates_list:
                if u != v and ((adj[v] >> u) & 1):
                    if u in colors:
                        neighbor_colors.add(colors[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            color_count = max(color_count, c + 1)
        
        # Upper bound is color_count (max independent set in graph <= number of colors)
        if len(current_set) + color_count <= best_size:
            return
        
        # Choose vertex to branch on
        # Heuristic: pick vertex with highest degree in candidates subgraph
        best_v = None
        best_deg = -1
        for v in candidates_list:
            deg = sum(1 for u in candidates_list if u != v and ((adj[v] >> u) & 1))
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        # Branch 1: include best_v
        new_current = current_set | {best_v}
        new_candidates = set(u for u in candidates if u != best_v and not ((adj[best_v] >> u) & 1))
        branch_and_bound(new_candidates, new_current, depth + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude best_v
        new_candidates2 = candidates - {best_v}
        branch_and_bound(new_candidates2, current_set, depth + 1)
    
    # Start with all vertices
    all_vertices = set(range(m))
    branch_and_bound(all_vertices, set())
    
    # If we didn't improve, return the greedy solution
    if len(best_set) <= best_size:
        result = sorted(best_set)
    else:
        result = sorted(best_set)
    
    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
