import time
import sys
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute neighbors for fast access
    # We will use a static order but try MRV if static fails or is slow?
    # For N=50, a well-chosen static order or simple MRV is fine.
    # Let's use a simple static order first, optimized by trying colors in a specific order.
    
    start_time = time.time()
    timeout = 50.0 # 50 seconds limit to be safe
    
    def is_connected_in_subgraph(start, target, colors, c1, c2, exclude_v):
        """
        Check if start and target are connected in the subgraph induced by colors c1 and c2,
        excluding vertex exclude_v.
        """
        if start == target:
            return True
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor == exclude_v:
                    continue
                if colors[neighbor] in (c1, c2) and neighbor not in visited:
                    if neighbor == target:
                        return True
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def check_assignment(v, c, colors):
        """
        Check if assigning color c to vertex v is valid given current colors.
        1. Proper coloring: neighbors must not have color c.
        2. Bicolored forest: for each other color c', adding v to {c, c'} subgraph must not create a cycle.
        """
        # 1. Proper coloring
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        # 2. Bicolored forest check
        # For each color c' that is currently used (or potentially used, i.e., 0..k-1)
        # We only need to check c' that are actually present in the graph so far?
        # Actually, if c' is not present, the subgraph is just {v} or empty, which is a forest.
        # So we only need to check c' where there is at least one vertex with color c'.
        # But to be safe and simple, we check all c' in 0..k-1.
        
        # Optimization: Only check c' if there are neighbors of v with color c' or c.
        # If no neighbor has color c', then v is isolated in the {c, c'} subgraph, so no cycle.
        # If v has neighbors with color c, they are in the subgraph.
        
        for c_prime in range(k):
            if c_prime == c:
                continue
            
            # Find neighbors of v that have color c or c_prime
            # If there are fewer than 2 such neighbors, no cycle can be formed through v
            # because a cycle needs at least 2 edges incident to v in the subgraph.
            # Actually, if v has 1 neighbor in the subgraph, it's a leaf.
            # If v has 0 neighbors, it's isolated.
            # So we only need to check if v has >= 2 neighbors in the subgraph {c, c'}.
            
            neigh_in_subgraph = []
            for u in adj[v]:
                if colors[u] == c or colors[u] == c_prime:
                    neigh_in_subgraph.append(u)
            
            if len(neigh_in_subgraph) < 2:
                continue
            
            # Check if any pair of neighbors in neigh_in_subgraph is connected
            # in the subgraph induced by {c, c'} excluding v.
            # If they are connected, adding v creates a cycle.
            
            # To speed this up, we can just check connectivity from the first neighbor
            # to the second, third, etc.
            # But if we find one pair connected, it's invalid.
            # Actually, if ANY two are connected, it's a cycle.
            # So we can pick one neighbor, say neigh_in_subgraph[0], and check if it's connected
            # to any other neighbor in neigh_in_subgraph[1:].
            
            ref_neigh = neigh_in_subgraph[0]
            for other_neigh in neigh_in_subgraph[1:]:
                if is_connected_in_subgraph(ref_neigh, other_neigh, colors, c, c_prime, v):
                    return False
                    
        return True

    def backtrack(idx, colors, k, start_time):
        if time.time() - start_time > timeout:
            return None
        
        if idx == n:
            return colors[:]
        
        v = idx
        # Try colors 0 to k-1
        # Heuristic: try colors that are already used first? Or random?
        # Let's try 0, 1, 2...
        
        for c in range(k):
            if check_assignment(v, c, colors):
                colors[v] = c
                res = backtrack(idx + 1, colors, k, start_time)
                if res is not None:
                    return res
                colors[v] = -1
        return None

    # Try k=3
    colors = [-1] * n
    res = backtrack(0, colors, 3, start_time)
    if res is not None:
        return res

    # Try k=4
    colors = [-1] * n
    res = backtrack(0, colors, 4, start_time)
    if res is not None:
        return res

    # Fallback: If backtracking fails (e.g., due to time or no solution),
    # we need to return a valid coloring.
    # A valid coloring with many colors is always possible (e.g., all distinct).
    # But we want to minimize.
    # If 3 and 4 failed, maybe 5?
    # Given the constraints, 3 or 4 should work.
    # If we run out of time, we can try a greedy approach with 5 colors.
    
    # Greedy with 5 colors
    colors = [-1] * n
    for v in range(n):
        used_colors = set(colors[u] for u in adj[v] if colors[u] != -1)
        for c in range(5):
            if c not in used_colors:
                # Check bicolored constraint
                if check_assignment(v, c, colors):
                    colors[v] = c
                    break
        if colors[v] == -1:
            # Should not happen with 5 colors on degree 4
            colors[v] = 0 
            # Force a valid color? This might break constraints.
            # Better to just assign a unique color if possible?
            # But we are limited to 0..49.
            # Let's just pick the first valid one.
            for c in range(50):
                if check_assignment(v, c, colors):
                    colors[v] = c
                    break
    
    # Verify the result
    # If invalid, return list(range(n)) as a last resort (50 colors, always valid)
    # But list(range(n)) uses 50 colors.
    # Is it valid?
    # Proper coloring: yes.
    # Bicolored forest: any pair of colors induces at most 2 vertices (since all colors are distinct).
    # A graph with <= 2 vertices is a forest.
    # So list(range(n)) is always a valid solution.
    
    # However, we want to minimize.
    # If backtracking didn't find 3 or 4, we return the greedy result or fallback.
    # To be safe against timeout, we return the best we have.
    # If res is None, we return a valid coloring.
    
    # Let's do a quick check if the greedy result is valid.
    # If not, return list(range(n)).
    
    def is_valid_final(colors):
        # Proper
        for u in range(n):
            for v in adj[u]:
                if colors[u] == colors[v]:
                    return False
        # Bicolored
        colors_used = set(colors)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                uf = list(range(n))
                def find(x):
                    while uf[x] != x:
                        uf[x] = uf[uf[x]]
                        x = uf[x]
                    return x
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    uf[rx] = ry
                    return True
                for u, v in P['edges']:
                    if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                        if not union(u, v):
                            return False
        return True

    if res is None:
        # Greedy fallback
        colors = [-1] * n
        for v in range(n):
            used_colors = set(colors[u] for u in adj[v] if colors[u] != -1)
            for c in range(5):
                if c not in used_colors:
                    if check_assignment(v, c, colors):
                        colors[v] = c
                        break
            if colors[v] == -1:
                colors[v] = 0
                # If 0 is invalid, try others
                for c in range(1, 5):
                    if check_assignment(v, c, colors):
                        colors[v] = c
                        break
                if colors[v] == 0 and not check_assignment(v, 0, colors):
                     # Just pick a unique color to be safe?
                     # No, we need to minimize.
                     # But if we can't find a valid one with 5, we use 50.
                     pass
        
        if is_valid_final(colors):
            return colors
        else:
            return list(range(n))
    else:
        return res

