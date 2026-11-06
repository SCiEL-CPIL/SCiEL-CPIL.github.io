import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer for safety

    # We aim for n=17, then 16, etc.
    # The best known is 17.
    for n in range(17, 9, -1):
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on n
        # n=17 gets most time
        time_limit = 100.0 if n == 17 else 50.0
        target_end = time.time() + time_limit
        
        result = _solve(n, target_end)
        if result:
            return result
            
    # Fallback
    if time.time() < deadline - 1:
        res = _solve(10, time.time() + 1.0)
        if res:
            return res
            
    return {'n': 1, 'colors': []}

def _solve(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Adjacency lists for each color
    # adj[c][u] is a set of neighbors
    adj = [ [set() for _ in range(n)] for _ in range(4) ]
    colors = [0] * num_edges
    
    # Helper to update adj
    def add_edge(c, u, v):
        adj[c][u].add(v)
        adj[c][v].add(u)
        
    def remove_edge(c, u, v):
        adj[c][u].discard(v)
        adj[c][v].discard(u)

    def count_paths_c6(u, v, c, exclude_u=None, exclude_v=None):
        """
        Count simple paths of length 5 from u to v in G_c.
        If exclude_u/exclude_v is provided, the edge (exclude_u, exclude_v) is treated as non-existent.
        """
        count = 0
        visited = {u}
        
        def dfs(current, depth):
            nonlocal count
            if depth == 5:
                if current == v:
                    count += 1
                return
                
            neighbors = adj[c][current]
            # Filter out visited and excluded edge
            for nb in neighbors:
                if nb in visited:
                    continue
                if exclude_u is not None:
                    if (current == exclude_u and nb == exclude_v) or (current == exclude_v and nb == exclude_u):
                        continue
                
                visited.add(nb)
                dfs(nb, depth + 1)
                visited.remove(nb)
                
        dfs(u, 0)
        return count

    def get_total_c6_count():
        total = 0
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            # Count C6s containing this edge
            # This is paths of length 5 between u and v in G_c \ {(u,v)}
            total += count_paths_c6(u, v, c, exclude_u=u, exclude_v=v)
        # Note: Each C6 is counted 6 times (once for each edge).
        # So actual number of C6s is total // 6.
        # For cost minimization, we can just use total.
        return total

    # Initialization: Geometric
    # Place vertices on a circle.
    # Color based on distance
    for idx, (u, v) in enumerate(edges):
        dist = abs(u - v)
        dist = min(dist, n - dist)
        # Map distance to color
        # n=17, max dist=8
        # 0-1 -> 0, 2-3 -> 1, 4-5 -> 2, 6-7 -> 3, 8 -> 0?
        # Let's try a spread
        colors[idx] = (dist * 4) // max(1, n)
        # Ensure 0..3
        colors[idx] = colors[idx] % 4
        add_edge(colors[idx], u, v)
        
    current_cost = get_total_c6_count()
    best_cost = current_cost
    best_colors = colors.copy()
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    # Simulated Annealing
    while time.time() < deadline:
        # Pick random edge
        idx = random.randint(0, num_edges - 1)
        u, v = edges[idx]
        old_c = colors[idx]
        
        # Pick new color
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        # Calculate delta
        # Remove (u,v) from old_c: destroys C6s containing (u,v) in old_c
        # Add (u,v) to new_c: creates C6s containing (u,v) in new_c
        
        # Cost of C6s destroyed:
        # Paths of length 5 between u and v in G_old_c \ {(u,v)}
        cost_destroyed = count_paths_c6(u, v, old_c, exclude_u=u, exclude_v=v)
        
        # Cost of C6s created:
        # Paths of length 5 between u and v in G_new_c (which does not have (u,v) yet)
        cost_created = count_paths_c6(u, v, new_c)
        
        delta = cost_created - cost_destroyed
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            # Accept
            # Update state
            colors[idx] = new_c
            remove_edge(old_c, u, v)
            add_edge(new_c, u, v)
            
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
            if best_cost == 0:
                # Found a valid coloring
                return {'n': n, 'colors': best_colors}
                
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with higher T?
            T = 10.0
            
    if best_cost == 0:
        return {'n': n, 'colors': best_colors}
        
    return None

# Verification function to be safe
def _verify(n, colors):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(colors) != len(edges):
        return False
    if any(c not in [0,1,2,3] for c in colors):
        return False
        
    adj = [ [set() for _ in range(n)] for _ in range(4) ]
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        adj[c][u].add(v)
        adj[c][v].add(u)
        
    for c in range(4):
        # Check for C6 in G_c
        # Simple DFS for cycles of length 6
        # To be fast, we can just check all 6-tuples? No, too slow.
        # We can use the same path counting logic.
        # If sum of paths for all edges is 0, no C6.
        for u in range(n):
            for v in adj[c][u]:
                if u < v:
                    # Count paths of length 5 from u to v in G_c \ {(u,v)}
                    # We need to exclude (u,v)
                    # The DFS in count_paths_c6 handles this if we pass exclude
                    # But here we are in a different context.
                    # Let's just do a quick check:
                    # If there is a C6, count_paths_c6(u, v, c, u, v) > 0
                    if count_paths_c6(u, v, c, exclude_u=u, exclude_v=v) > 0:
                        return False
    return True
