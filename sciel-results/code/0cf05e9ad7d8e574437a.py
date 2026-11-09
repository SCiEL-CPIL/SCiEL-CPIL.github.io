"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_c6(color_subgraph, n):
    """Check if a color subgraph (list of edges) contains a C6 (cycle of length 6)."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for (u, v) in color_subgraph:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all simple paths of length 5 (6 vertices), check if first and last are adjacent
    # Use backtracking with pruning
    def find_c6(start):
        # DFS to find paths of length 5 starting from start
        def dfs(current, path, visited):
            if len(path) == 6:
                # Check if path[0] is adjacent to path[5]
                if path[0] in adj[path[5]]:
                    return True
                return False
            for nxt in adj[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    if dfs(nxt, path, visited):
                        return True
                    path.pop()
                    visited.remove(nxt)
            return False
        
        visited = {start}
        return dfs(start, [start], visited)
    
    for v in range(n):
        if find_c6(v):
            return True
    return False

def _get_color_subgraphs(colors, n):
    """Return list of 4 subgraphs, each a list of (u,v) edges for that color."""
    subs = [[] for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            subs[c].append((i, j))
            idx += 1
    return subs

def _has_monochromatic_c6(colors, n):
    """Check if any color class contains a C6."""
    subs = _get_color_subgraphs(colors, n)
    for c in range(4):
        if _has_c6(subs[c], n):
            return True
    return False

def _count_c6s(colors, n):
    """Count total number of C6s across all color classes."""
    subs = _get_color_subgraphs(colors, n)
    total = 0
    for c in range(4):
        adj = [set() for _ in range(n)]
        for (u, v) in subs[c]:
            adj[u].add(v)
            adj[v].add(u)
        
        def count_c6s_in_subgraph(adj, n):
            count = 0
            # Count each C6 exactly once by requiring the smallest vertex to be first
            for start in range(n):
                def dfs(current, path, visited):
                    nonlocal count
                    if len(path) == 6:
                        if path[0] in adj[path[5]]:
                            # Ensure start is the smallest vertex in the cycle
                            if all(start < v for v in path[1:]):
                                count += 1
                        return
                    for nxt in adj[current]:
                        if nxt not in visited and nxt > start:  # only go to larger vertices to avoid duplicates
                            visited.add(nxt)
                            path.append(nxt)
                            dfs(nxt, path, visited)
                            path.pop()
                            visited.remove(nxt)
                
                visited = {start}
                dfs(start, [start], visited)
            return count
        
        total += count_c6s_in_subgraph(adj, n)
    return total

def _random_local_search(initial_colors, n, max_steps=50000, deadline=None):
    """Perform local search to minimize C6 count."""
    colors = list(initial_colors)
    best_colors = list(colors)
    best_c6_count = _count_c6s(colors, n)
    
    if best_c6_count == 0:
        return colors
    
    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge to flip
        idx = random.randint(0, len(colors) - 1)
        old_color = colors[idx]
        new_color = random.choice([c for c in range(4) if c != old_color])
        colors[idx] = new_color
        
        new_c6_count = _count_c6s(colors, n)
        
        if new_c6_count < best_c6_count:
            best_c6_count = new_c6_count
            best_colors = list(colors)
            if best_c6_count == 0:
                return colors
        else:
            # Revert
            colors[idx] = old_color
        
        step += 1
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with the known n=9 solution
    n0 = 9
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    best_n = n0
    best_colors = list(colors0)
    
    # Try to extend from n=9 up to n=17
    for target_n in range(n0 + 1, 18):
        if time.time() > deadline - 10:
            break
        
        # Initialize new colors: construct list in lexicographic order
        # For pairs (i,j) where j < n0, use the color from colors0.
        # Otherwise, use a random color.
        colors = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if j < n0:
                    # Calculate index of (i,j) in the lexicographic list for K_n0
                    # Index = sum_{k=0}^{i-1} (n0-1-k) + (j-i-1)
                    # Sum = i*n0 - i - (i*(i-1))/2
                    idx0 = i * (n0 - 1) - i * (i - 1) // 2 + (j - i - 1)
                    colors.append(colors0[idx0])
                else:
                    colors.append(random.randint(0, 3))
        
        # Run local search
        # Reduce max_steps if time is tight, but keep it reasonable
        result = _random_local_search(colors, target_n, max_steps=5000, deadline=deadline - 5)
        
        if _count_c6s(result, target_n) == 0:
            best_n = target_n
            best_colors = result
        else:
            # If we can't find a valid coloring for this n, stop trying larger n
            # But we might still try with more iterations... for now, break
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
