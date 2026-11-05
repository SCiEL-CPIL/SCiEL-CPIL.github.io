import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def check_c6(color_graph, u, v):
        # Check if adding edge (u,v) to color_graph creates a C6.
        # This happens if there is a path of length 5 between u and v in color_graph.
        # We do a BFS/DFS from u up to depth 5.
        # Since we are adding (u,v), we look for path u -> ... -> v of length 5.
        # Actually, if there is a path of length 5, then u-v plus that path is a C6.
        # Note: The path must be simple (no repeated vertices).
        
        # BFS from u
        # State: (node, depth, path_set) - path_set is expensive.
        # Better: Just check distances? No, distance 5 doesn't guarantee simple path of length 5?
        # In a simple graph, if dist(u,v)=5, there is a simple path of length 5.
        # But there could be a path of length 5 even if dist(u,v) < 5?
        # No, if there is a path of length 5, dist(u,v) <= 5.
        # If dist(u,v) == 5, we have a C6.
        # What if dist(u,v) < 5? Could there still be a simple path of length 5?
        # Yes. Example: triangle u-a-b-v and another path u-c-d-v.
        # So we must explicitly search for a simple path of length 5.
        
        # Given small n (<=20) and low density, DFS with backtracking for path of length 5 is fine.
        # Or BFS with state (node, visited_mask).
        # n=17, mask fits in int.
        
        def has_path_len_5(start, end, adj, n):
            # adj: list of lists of neighbors
            # Returns True if there is a simple path of length 5 from start to end.
            # DFS
            def dfs(node, depth, visited):
                if depth == 5:
                    return node == end
                if depth > 5:
                    return False
                for nei in adj[node]:
                    if not (visited & (1 << nei)):
                        if dfs(nei, depth + 1, visited | (1 << nei)):
                            return True
                return False
            
            return dfs(start, 0, 1 << start)
        
        adj = color_graph
        return has_path_len_5(u, v, adj, len(adj))

    def solve(n):
        # Backtracking search for n vertices
        # Edges are (i, j) for i < j
        # colors: list of -1 for unassigned
        # color_graphs: list of 4 adjacency lists (or bitmasks?)
        # Bitmasks are faster for adjacency checks.
        
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        colors = [-1] * m
        
        # Precompute edge indices
        edge_idx = {}
        for idx, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = idx
            edge_idx[(v, u)] = idx
            
        # Adjacency lists for each color
        # Using list of sets or bitmasks?
        # Bitmasks are good for "is neighbor" check.
        # For DFS path finding, we need to iterate neighbors.
        # Let's use list of lists, and update them.
        
        adj = [[set() for _ in range(n)] for _ in range(4)]
        
        # Order edges? 
        # Standard lexicographic order might be slow.
        # We can just use the given order.
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == m:
                return colors[:]
            
            u, v = edges[idx]
            
            # Try each color
            # Heuristic: try colors that are least constrained? 
            # Or just 0,1,2,3.
            # To improve, we can check if adding (u,v) to color c is valid.
            
            for c in range(4):
                # Check validity
                # We need to check if u and v are connected by a path of length 5 in adj[c].
                # Optimization: Only check if u and v are not already connected by shorter paths?
                # No, path of length 5 is specific.
                
                # Quick check: if u and v are already in adj[c], it's a multigraph? No, simple graph.
                # They are not adjacent yet.
                
                # Perform the C6 check
                # We need to temporarily add v to adj[c][u] and u to adj[c][v] for the check?
                # No, the check is "does a path of length 5 exist between u and v in the CURRENT graph?".
                # If yes, adding (u,v) creates a C6.
                # So we check on the current state.
                
                if check_c6(adj[c], u, v):
                    continue
                
                # Assign
                colors[idx] = c
                adj[c][u].add(v)
                adj[c][v].add(u)
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                # Unassign
                colors[idx] = -1
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                
            return None
        
        return backtrack(0)

    # Try n from 17 down to 9
    # We know 9 is valid.
    # We want the max n.
    
    # To increase chance of finding 17, we can randomize the order of colors or edges?
    # Randomizing edge order is hard with precomputed indices.
    # Randomizing color order is easy.
    
    # Let's try a few random restarts with shuffled color orders if the first fails quickly?
    # Or just one strong backtracking run.
    # For n=17, backtracking might be slow if not well-guided.
    # Let's add a simple heuristic: try colors in order of least usage?
    # Or just rely on the structure.
    
    # Actually, for n=17, a known construction exists.
    # If backtracking is too slow, we might time out.
    # Let's try to implement a faster check or a different search strategy.
    
    # Alternative: Use the known cyclic construction for n=17 if possible?
    # If I can't hardcode it, I must search.
    
    # Let's optimize the check_c6.
    # Instead of DFS every time, we can precompute something?
    # No, the graph changes.
    
    # Let's try to run the search for n=17.
    # If it takes too long, we fall back.
    
    # To make it faster, we can use NumPy or bitmasks for the path check?
    # Path of length 5 check:
    # We can compute the number of walks of length 5? No, simple paths.
    
    # Let's stick to the DFS but optimize it.
    # The DFS is called for each edge and each color.
    # Total calls: 136 * 4 = 544 checks per level? No, only for the current edge.
    # Depth 17.
    # 136 levels.
    # At each level, 4 checks.
    # Each check is DFS depth 5.
    # This should be fast enough for n=17 if the search tree isn't huge.
    # The search tree is the main risk.
    
    # Let's try to solve n=17.
    result = solve(17)
    
    if result is None:
        # Try n=16
        result = solve(16)
        if result is None:
            result = solve(15)
            if result is None:
                # Fallback to known 9
                n = 9
                return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    n = 17 if result is not None else (16 if len(result) == 16*15//2 else 15)
    # Wait, solve returns None or colors list.
    # If solve(17) returns list, n=17.
    # If solve(16) returns list, n=16.
    
    # Let's structure the fallback properly.
    
    # Re-implementing the loop to be cleaner
    
    # Actually, the code above inside construct() has a local solve().
    # I will rewrite the whole function cleanly.

    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def check_c6(adj, u, v, n):
        # Check if there is a simple path of length 5 from u to v in adj
        # adj: list of sets
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            if depth > 5:
                return False
            for nei in adj[node]:
                if not (visited & (1 << nei)):
                    if dfs(nei, depth + 1, visited | (1 << nei)):
                        return True
            return False
        return dfs(u, 0, 1 << u)

    def solve(n):
        if time.time() > deadline:
            return None
            
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        colors = [-1] * m
        adj = [[set() for _ in range(n)] for _ in range(4)]
        
        # Heuristic: Order edges? 
        # Let's just use natural order.
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == m:
                return colors[:]
            
            u, v = edges[idx]
            
            # Try colors 0,1,2,3
            # Randomize? No, keep deterministic for reproducibility if possible, 
            # but randomization might help escape bad local minima in backtracking?
            # Backtracking is complete, so it will find a solution if one exists, 
            # unless it's too slow.
            
            for c in range(4):
                if check_c6(adj[c], u, v, n):
                    continue
                
                colors[idx] = c
                adj[c][u].add(v)
                adj[c][v].add(u)
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                colors[idx] = -1
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                
            return None
        
        return backtrack(0)

    # Try to find the largest n
    # Start from 17 down
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        res = solve(n)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback to n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
