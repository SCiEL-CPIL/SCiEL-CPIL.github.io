"""Rewritten program: construct a 4-colouring of K_n with no monochromatic C6."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Precompute edge index mapping for n up to 20
def edge_index(n, i, j):
    """Return index of edge (i,j) in the lexicographic edge list for K_n."""
    # Edges are ordered: (0,1),(0,2),...,(0,n-1),(1,2),...,(n-2,n-1)
    # Index = i*(2*n - i - 1)//2 + (j - i - 1)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def build_adj(n, colors, color):
    """Build adjacency list for a given color class."""
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = edge_index(n, i, j)
            if colors[idx] == color:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def has_c6(n, colors, color):
    """Check if a color class contains a C6. Returns True if found."""
    adj = build_adj(n, colors, color)
    
    # For each edge (u,v), check if there's a path of length 5 from v to u
    # that doesn't use edge (u,v). This path plus edge (u,v) forms a C6.
    # Path of length 5: v -> a -> b -> c -> d -> e -> u (6 edges? No, 5 edges)
    # Wait: C6 has 6 edges. If we fix edge (u,v), the remaining 5 edges form a path
    # of length 5 from v to u: v-a-b-c-d-u. That's 5 edges, 6 vertices total.
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # Temporarily remove edge (u,v)
            adj[u].discard(v)
            adj[v].discard(u)
            
            # DFS for simple path of length 5 from v to u
            # State: (current_node, depth, visited_set)
            # We want to reach u at depth 5
            
            found = False
            visited = {v}
            
            def dfs(node, depth):
                nonlocal found
                if found:
                    return
                if depth == 5:
                    if node == u:
                        found = True
                    return
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        dfs(nb, depth + 1)
                        visited.discard(nb)
                        if found:
                            return
            
            dfs(v, 0)
            
            # Restore edge
            adj[u].add(v)
            adj[v].add(u)
            
            if found:
                return True
    
    return False

def is_valid(n, colors):
    """Check if the coloring is valid (no monochromatic C6)."""
    for color in range(4):
        if has_c6(n, colors, color):
            return False
    return True

def count_c6s(n, colors):
    """Count total number of monochromatic C6s across all colors."""
    total = 0
    for color in range(4):
        adj = build_adj(n, colors, color)
        count = 0
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                adj[u].discard(v)
                adj[v].discard(u)
                
                # Count paths of length 5 from v to u
                # Use BFS with depth tracking, but only count simple paths
                # This is expensive, so we'll just check existence for now
                # For counting, we'd need to enumerate all paths, which is too slow
                # Let's just use has_c6 logic but count instead of early exit
                
                visited = {v}
                def count_paths(node, depth):
                    nonlocal count
                    if depth == 5:
                        if node == u:
                            count += 1
                        return
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            count_paths(nb, depth + 1)
                            visited.discard(nb)
                
                count_paths(v, 0)
                
                adj[u].add(v)
                adj[v].add(u)
        total += count
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with a known small valid coloring
    # For n=4, any coloring is valid (no C6 possible)
    # Let's start with n=8 and try to grow
    
    best_n = 4
    best_colors = [0] * (4 * 3 // 2)  # All edges color 0, no C6 in K4
    
    # Try to build up from small n
    current_n = 4
    current_colors = [0] * (current_n * (current_n - 1) // 2)
    
    # Try to extend to larger n
    random.seed(42)
    
    for target_n in range(5, 20):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random extensions
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            
            # Start from current valid coloring and add new vertex
            n = target_n
            num_edges = n * (n - 1) // 2
            
            # Initialize new edges randomly
            colors = current_colors.copy()
            for i in range(current_n, n):
                for j in range(i + 1, n):
                    idx = edge_index(n, i, j)
                    colors[idx] = random.randint(0, 3)
            # Edges from new vertices to old vertices
            for i in range(current_n, n):
                for j in range(i):
                    idx = edge_index(n, j, i)
                    colors[idx] = random.randint(0, 3)
            
            # Local search to eliminate C6s
            for step in range(200):
                if time.time() > deadline - 1:
                    break
                
                # Check validity
                if is_valid(n, colors):
                    current_n = n
                    current_colors = colors.copy()
                    best_n = max(best_n, n)
                    best_colors = colors.copy()
                    break
                
                # Find a C6 and try to fix it
                # Try random edge flips
                fixed = False
                for _ in range(10):
                    if time.time() > deadline - 1:
                        break
                    
                    # Pick a random edge and flip its color
                    e_idx = random.randint(0, num_edges - 1)
                    old_color = colors[e_idx]
                    new_color = random.randint(0, 3)
                    if new_color == old_color:
                        new_color = (new_color + 1) % 4
                    
                    colors[e_idx] = new_color
                    
                    # Check if this improves things (fewer C6s)
                    # This is expensive, so just accept with some probability
                    if random.random() < 0.3:
                        fixed = True
                        break
                    else:
                        colors[e_idx] = old_color
                
                if fixed:
                    continue
                else:
                    # No improvement, break and try new initialization
                    break
        
        if current_n == target_n:
            continue
        else:
            # Couldn't extend to target_n, try next
            continue
    
    # Final verification
    if best_n >= 1:
        if not is_valid(best_n, best_colors):
            # Fallback to trivial valid coloring
            best_n = 4
            best_colors = [0] * 6
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
