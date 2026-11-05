import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def has_path_len_5(u, v, c, adj, n):
        """
        Check if there is a simple path of length 5 from u to v in graph adj[c].
        adj[c] is a list of bitmasks.
        """
        if u == v:
            return False
        
        # DFS
        # visited_mask tracks visited vertices
        # curr is current vertex
        # depth is current path length
        # We want to reach v at depth 5
        
        def dfs(curr, depth, visited):
            if depth == 5:
                return curr == v
            
            # Get neighbors of curr in color c
            mask = adj[c][curr]
            
            # Iterate through neighbors
            while mask:
                lsb = mask & -mask
                nxt = lsb.bit_length() - 1
                mask -= lsb
                
                if (visited >> nxt) & 1:
                    continue
                
                # We cannot visit v before depth 5
                if nxt == v and depth < 4:
                    continue
                
                # If we reach v at depth 4, we need one more step to depth 5?
                # No, path length 5 means 5 edges: u-x1-x2-x3-x4-v.
                # Depth 0: u
                # Depth 1: x1
                # ...
                # Depth 5: v
                # So if depth == 4 and nxt == v, we return True?
                # Wait.
                # Path: u (0) -> x1 (1) -> x2 (2) -> x3 (3) -> x4 (4) -> v (5)
                # If curr is x4 (depth 4), and nxt is v, then we found a path of length 5.
                # So:
                if nxt == v:
                    if depth == 4:
                        return True
                    else:
                        continue # Cannot go to v too early
                
                if dfs(nxt, depth + 1, visited | (1 << nxt)):
                    return True
            
            return False
        
        return dfs(u, 0, 1 << u)

    def solve_n(target_n):
        if time.time() > deadline:
            return None
            
        # adj[c][i] is bitmask of neighbors of i in color c
        adj = [[0] * target_n for _ in range(4)]
        
        # Edges in lexicographic order
        edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
        
        # Symmetry breaking
        # We can fix the first few edges.
        # (0,1) = 0
        # (0,2) in {0, 1}
        # (0,3) in {0, 1, 2}
        # This reduces the initial branching factor.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            
            # Determine possible colors based on symmetry breaking
            possible_colors = [0, 1, 2, 3]
            if idx == 0: # (0,1)
                possible_colors = [0]
            elif idx == 1: # (0,2)
                possible_colors = [0, 1]
            elif idx == 2: # (0,3)
                possible_colors = [0, 1, 2]
            
            for c in possible_colors:
                # Check constraint: Does adding (u,v) in color c create a C6?
                # This happens if there is a path of length 5 from u to v in G_c.
                # Note: G_c does not contain (u,v) yet.
                if has_path_len_5(u, v, c, adj, target_n):
                    continue
                
                # Assign color c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
            
            return False
        
        if backtrack(0):
            # Extract colors
            colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    for c in range(4):
                        if (adj[c][i] >> j) & 1:
                            colors.append(c)
                            break
            return {'n': target_n, 'colors': colors}
        return None

    # Try to solve for n=18
    res = solve_n(18)
    if res:
        return res

    # Try to solve for n=17
    res = solve_n(17)
    if res:
        return res

    # Fallback to a known small solution if time is critical
    # K5 is always valid
    return {'n': 5, 'colors': [0] * 10}
