import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 118.0

    # We will attempt to find a valid 4-coloring for n in descending order.
    # The best known is 17. We try 18 first (new result), then 17, 16, 15.
    for n in [18, 17, 16, 15]:
        if time.time() > start_time + deadline - 5:
            break

        # Precompute edge list in lexicographic order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)

        # For each edge index, store (u, v)
        edge_uv = edges

        # adj[c][u] is a bitmask of neighbors of u in color c
        # Use Python ints for bitmasks (fast bitwise ops)
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges

        # Helper to map (u,v) with u<v to edge index
        # For n<=20, we can use a dict for clarity and speed
        edge_index = {}
        for idx, (u, v) in enumerate(edges):
            edge_index[(u, v)] = idx

        def set_edge(u, v, c, idx):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[idx] = c

        def unset_edge(u, v, c, idx):
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1

        # Symmetry breaking:
        # Break color permutation symmetry by fixing the first few edges.
        # Also break some vertex symmetry by fixing edges from vertex 0 to 1,2,3,4
        # to distinct colors if possible.
        # We'll fix:
        # (0,1) -> 0
        # (0,2) -> 1
        # (0,3) -> 2
        # (0,4) -> 3
        # This is valid for n >= 5.
        start_idx = 0
        if n >= 5:
            set_edge(0, 1, 0, edge_index[(0,1)])
            set_edge(0, 2, 1, edge_index[(0,2)])
            set_edge(0, 3, 2, edge_index[(0,3)])
            set_edge(0, 4, 3, edge_index[(0,4)])
            start_idx = 4  # We'll start from edge index 4, but we need to skip the fixed ones.
            # Actually, it's easier to just set a flag or start from 0 and skip fixed edges.
            # Let's create a list of fixed edge indices.
            fixed_edges = {edge_index[(0,1)], edge_index[(0,2)], edge_index[(0,3)], edge_index[(0,4)]}
        else:
            fixed_edges = set()
            if n >= 4:
                set_edge(0, 1, 0, edge_index[(0,1)])
                set_edge(0, 2, 1, edge_index[(0,2)])
                set_edge(0, 3, 2, edge_index[(0,3)])
                fixed_edges = {edge_index[(0,1)], edge_index[(0,2)], edge_index[(0,3)]}
            elif n >= 3:
                set_edge(0, 1, 0, edge_index[(0,1)])
                set_edge(0, 2, 1, edge_index[(0,2)])
                fixed_edges = {edge_index[(0,1)], edge_index[(0,2)]}
            elif n >= 2:
                set_edge(0, 1, 0, edge_index[(0,1)])
                fixed_edges = {edge_index[(0,1)]}

        # Fast check: is there a path of length 5 between u and v in color c?
        # We can use BFS/DFS with bitmasks.
        # Since n <= 20, we can use a simple DFS with a visited bitmask.
        # To speed it up, we can precompute for each color c and each node u, the set of nodes at distance d.
        # But dynamic updates make that hard.
        # Instead, we'll use a recursive DFS with memoization? No, state is too complex.
        # Let's use an iterative DFS with a stack of (node, depth, visited_mask).
        # We can prune if depth > 5.

        def has_path_len_5(u, v, c):
            if u == v:
                return False
            # If u and v are not connected in any path of length 5, return False.
            # Quick check: if v is not in the union of neighbors of neighbors...
            # Let's just do the DFS.
            # To make it faster, we can use a bidirectional search?
            # For now, simple DFS.
            # We can also check if the degree of u or v is 0, then no path.
            if adj[c][u] == 0 or adj[c][v] == 0:
                return False
            
            # We want a simple path of length 5: u = v0 - v1 - v2 - v3 - v4 - v5 = v
            # So we need to find v1, v2, v3, v4 such that:
            # (u, v1) in E_c, (v1, v2) in E_c, (v2, v3) in E_c, (v3, v4) in E_c, (v4, v) in E_c
            # and all vertices are distinct.
            
            # We can iterate over possible v1 (neighbors of u)
            # then v2 (neighbors of v1, not u)
            # then v3 (neighbors of v2, not u, v1)
            # then v4 (neighbors of v3, not u, v1, v2, and v4 must be a neighbor of v)
            
            # This is 4 levels of iteration.
            # Let's implement it with bitmasks for speed.
            
            # v1 candidates: neighbors of u
            v1_mask = adj[c][u]
            # v1 cannot be v (if path length 1, but we need length 5, so v1 != v)
            v1_mask &= ~(1 << v)
            
            while v1_mask:
                lsb1 = v1_mask & -v1_mask
                v1 = lsb1.bit_length() - 1
                v1_mask ^= lsb1
                
                # v2 candidates: neighbors of v1, not u, not v
                v2_mask = adj[c][v1]
                v2_mask &= ~(1 << u)
                v2_mask &= ~(1 << v)
                
                while v2_mask:
                    lsb2 = v2_mask & -v2_mask
                    v2 = lsb2.bit_length() - 1
                    v2_mask ^= lsb2
                    
                    # v3 candidates: neighbors of v2, not u, v1, v
                    v3_mask = adj[c][v2]
                    v3_mask &= ~(1 << u)
                    v3_mask &= ~(1 << v1)
                    v3_mask &= ~(1 << v)
                    
                    while v3_mask:
                        lsb3 = v3_mask & -v3_mask
                        v3 = lsb3.bit_length() - 1
                        v3_mask ^= lsb3
                        
                        # v4 candidates: neighbors of v3, not u, v1, v2, v
                        # and v4 must be a neighbor of v
                        v4_mask = adj[c][v3]
                        v4_mask &= adj[c][v]  # v4 must be neighbor of v
                        v4_mask &= ~(1 << u)
                        v4_mask &= ~(1 << v1)
                        v4_mask &= ~(1 << v2)
                        v4_mask &= ~(1 << v)
                        
                        if v4_mask:
                            return True
            
            return False

        # Backtracking search
        # We'll use a recursive function, but to avoid recursion depth issues,
        # we can use an iterative approach or increase recursion limit.
        # For n=18, num_edges = 153, so recursion depth is 153, which is fine.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(idx):
            if time.time() > start_time + deadline:
                return False
            if idx == num_edges:
                return True
            
            # Skip fixed edges
            if idx in fixed_edges:
                return backtrack(idx + 1)
            
            u, v = edge_uv[idx]
            
            # Try colors 0, 1, 2, 3
            # Heuristic: Try colors in order of increasing constraint?
            # For now, 0, 1, 2, 3
            for c in range(4):
                # Check if adding (u,v) to color c creates a C6
                if has_path_len_5(u, v, c):
                    continue
                
                set_edge(u, v, c, idx)
                
                if backtrack(idx + 1):
                    return True
                
                unset_edge(u, v, c, idx)
            
            return False

        if backtrack(0):
            return {'n': n, 'colors': colors}
            
    # Fallback: Return a valid small coloring if all else fails
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
