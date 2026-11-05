import time
import random

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin for 120s limit
    
    n = 16
    # adj[u] is a bitmask of neighbors of u
    # Initial state: Complete graph K16
    adj = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v:
                adj[u] |= (1 << v)
    
    def is_q3_free(adj):
        """
        Returns True if graph is Q3-free.
        Otherwise, returns a list of edges forming a Q3.
        """
        for u in range(n):
            # Get neighbors of u
            neighbors = []
            for v in range(n):
                if v != u and (adj[u] >> v) & 1:
                    neighbors.append(v)
            
            # Iterate over triples of neighbors
            len_n = len(neighbors)
            for i in range(len_n):
                a = neighbors[i]
                for j in range(i+1, len_n):
                    b = neighbors[j]
                    # a and b must not be adjacent
                    if (adj[a] >> b) & 1:
                        continue
                    for k in range(j+1, len_n):
                        c = neighbors[k]
                        # b and c must not be adjacent
                        if (adj[b] >> c) & 1:
                            continue
                        # a and c must not be adjacent
                        if (adj[a] >> c) & 1:
                            continue
                        
                        # Find common neighbors of a and b, excluding u and c
                        common_ab = adj[a] & adj[b]
                        common_ab &= ~(1 << u)
                        common_ab &= ~(1 << c)
                        
                        common_ac = adj[a] & adj[c]
                        common_ac &= ~(1 << u)
                        common_ac &= ~(1 << b)
                        
                        common_bc = adj[b] & adj[c]
                        common_bc &= ~(1 << u)
                        common_bc &= ~(1 << a)
                        
                        if common_ab == 0 or common_ac == 0 or common_bc == 0:
                            continue
                        
                        # Iterate over potential x, y, z
                        # x in common_ab, y in common_ac, z in common_bc
                        # We need to check if x,y,z,w form the rest of the cube
                        # To reduce complexity, we can just pick one candidate from each
                        # and check if a valid w exists.
                        
                        # Optimization: Just check if there exists ANY x,y,z
                        # that allows a w.
                        
                        # Extract lists
                        list_ab = [v for v in range(n) if (common_ab >> v) & 1]
                        list_ac = [v for v in range(n) if (common_ac >> v) & 1]
                        list_bc = [v for v in range(n) if (common_bc >> v) & 1]
                        
                        for x in list_ab:
                            if x in [a, b, c, u]: continue
                            for y in list_ac:
                                if y in [a, b, c, u, x]: continue
                                # x and y must not be adjacent? In Q3, x and y are adjacent to a, but x-y edge?
                                # In Q3 (binary 000, 001, 010, 011, 100, 101, 110, 111)
                                # u=000, a=001, b=010, c=100
                                # x (neighbor of a,b) = 011
                                # y (neighbor of a,c) = 101
                                # z (neighbor of b,c) = 110
                                # w (neighbor of x,y,z) = 111
                                # Edges: (000,001),(000,010),(000,100)
                                # (001,011),(010,011) -> x=011 adj to a,b
                                # (001,101),(100,101) -> y=101 adj to a,c
                                # (010,110),(100,110) -> z=110 adj to b,c
                                # (011,111),(101,111),(110,111) -> w=111 adj to x,y,z
                                # Also (011,101)? No, 011 and 101 differ in 2 bits. Not adjacent.
                                # So x and y should NOT be adjacent.
                                if (adj[x] >> y) & 1:
                                    continue
                                for z in list_bc:
                                    if z in [a, b, c, u, x, y]: continue
                                    if (adj[x] >> z) & 1:
                                        continue
                                    if (adj[y] >> z) & 1:
                                        continue
                                    
                                    # Find w
                                    # w must be neighbor of x, y, z
                                    common_xyz = adj[x] & adj[y] & adj[z]
                                    # w must not be in {u,a,b,c,x,y,z}
                                    mask_exclude = (1 << u) | (1 << a) | (1 << b) | (1 << c) | (1 << x) | (1 << y) | (1 << z)
                                    common_xyz &= ~mask_exclude
                                    
                                    if common_xyz != 0:
                                        # Found a Q3
                                        # Return the edges
                                        w = (common_xyz & -common_xyz).bit_length() - 1
                                        edges = [
                                            [u, a], [u, b], [u, c],
                                            [a, x], [b, x],
                                            [a, y], [c, y],
                                            [b, z], [c, z],
                                            [x, w], [y, w], [z, w]
                                        ]
                                        return edges
        return None

    best_edges_count = -1
    best_adj = None

    # Run multiple random restarts
    while time.time() - start_time < deadline:
        # Start with K16
        adj = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v:
                    adj[u] |= (1 << v)
        
        # Phase 1: Remove edges to make Q3-free
        # Use a heuristic: find Q3, remove random edge
        while time.time() - start_time < deadline:
            q3_edges = is_q3_free(adj)
            if q3_edges is None:
                break
            
            # Pick a random edge from the Q3 to remove
            # To be smarter, we could pick an edge that appears in many Q3s, 
            # but random is fast and often works well with restarts
            e = random.choice(q3_edges)
            u, v = e[0], e[1]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        if time.time() - start_time > deadline:
            break
            
        # Phase 2: Try to add edges back
        # Iterate through all possible missing edges
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    continue
                
                # Try adding edge (u,v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if Q3-free
                if is_q3_free(adj) is not None:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Keep edge
                    pass
        
        # Count edges
        edge_count = sum(bin(adj[u]).count('1') for u in range(n)) // 2
        
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_adj = adj[:]

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
