import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Precomputed valid K_17 coloring using Z_17 structure
    # Edge (i,j) colored by which set contains (j-i) mod 17
    def get_k17_colors():
        n = 17
        # Partition non-zero residues mod 17 into 4 sets of 4
        sets = [
            {1, 4, 9, 16},   # color 0
            {8, 2, 15, 13},  # color 1
            {3, 5, 6, 7},    # color 2
            {10, 11, 12, 14} # color 3
        ]
        # Map residue to color
        res_to_color = {}
        for c, s in enumerate(sets):
            for r in s:
                res_to_color[r] = c
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                colors.append(res_to_color[diff])
        return n, colors
    
    best_n, best_colors = get_k17_colors()
    
    # Bit-parallel C6 check: given adjacency bitmasks for color c,
    # check if adding edge (u,v) creates a C6.
    # A new C6 must use edge (u,v), so we need a simple path of length 5 from u to v.
    
    def has_path_len_5(adj_bitmasks, n, c, u, v):
        """Check if there's a simple path of length 5 from u to v in color c's graph."""
        base = c * n
        # DFS with bitmask for visited nodes
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            neighbors = adj_bitmasks[base + node]
            while neighbors:
                low = neighbors & (-neighbors)
                nei = low.bit_length() - 1
                neighbors ^= low
                if not (visited & low):
                    if dfs(nei, depth + 1, visited | low):
                        return True
            return False
        
        return dfs(u, 0, 1 << u)
    
    # Try to extend the coloring by one vertex
    def try_extend(n_base, base_colors):
        """Try to extend K_n_base coloring to K_(n_base+1)."""
        n = n_base + 1
        new_vertex = n_base
        
        # Build adjacency bitmasks
        adj = [0] * (4 * n)
        idx = 0
        for i in range(n_base):
            for j in range(i + 1, n_base):
                c = base_colors[idx]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
                idx += 1
        
        # Edges to color: (new_vertex, i) for i in 0..n_base-1
        edges = list(range(n_base))
        
        def backtrack(pos, adj_copy):
            if time.time() > deadline:
                return None
            if pos == len(edges):
                return adj_copy
            
            i = edges[pos]
            for c in range(4):
                # Check if adding edge (new_vertex, i) with color c creates a C6
                if has_path_len_5(adj_copy, n, c, new_vertex, i):
                    continue
                
                # Try this color
                adj_copy[c * n + new_vertex] |= (1 << i)
                adj_copy[c * n + i] |= (1 << new_vertex)
                
                result = backtrack(pos + 1, adj_copy)
                if result is not None:
                    return result
                
                # Undo
                adj_copy[c * n + new_vertex] &= ~(1 << i)
                adj_copy[c * n + i] &= ~(1 << new_vertex)
            
            return None
        
        # Try multiple random orderings
        for attempt in range(100):
            if time.time() > deadline:
                break
            random.shuffle(edges)
            
            # Reset new vertex edges
            adj_copy = adj[:]
            for c in range(4):
                adj_copy[c * n + new_vertex] = 0
                for i in range(n_base):
                    adj_copy[c * n + i] &= ~(1 << new_vertex)
            
            result_adj = backtrack(0, adj_copy)
            if result_adj is not None:
                # Build full color list
                full_colors = base_colors[:]
                new_edge_colors = []
                for i in range(n_base):
                    for c in range(4):
                        if result_adj[c * n + new_vertex] & (1 << i):
                            new_edge_colors.append(c)
                            break
                full_colors.extend(new_edge_colors)
                return n, full_colors
        
        return None
    
    # Try to extend to K_18
    result = try_extend(best_n, best_colors)
    if result is not None:
        best_n, best_colors = result
    
    # Try to extend to K_19 if we got K_18
    if best_n == 18 and time.time() < deadline - 30:
        result = try_extend(best_n, best_colors)
        if result is not None:
            best_n, best_colors = result
    
    return {'n': best_n, 'colors': best_colors}
