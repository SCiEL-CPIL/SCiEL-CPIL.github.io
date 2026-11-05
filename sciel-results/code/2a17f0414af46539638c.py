import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try to construct a graph on n vertices
    for n in [35, 34, 33, 32, 31, 30, 29, 28]:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        max_restarts = 50
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize empty graph
            adj = [[False] * n for _ in range(n)]
            edges = []
            
            # Greedy edge addition
            success = True
            while True:
                if time.time() > deadline:
                    success = False
                    break
                
                # Find a maximal independent set
                # Greedy: pick a vertex, remove it and neighbors
                remaining = list(range(n))
                independent_set = []
                while remaining:
                    # Pick random vertex from remaining
                    idx = random.randint(0, len(remaining) - 1)
                    v = remaining[idx]
                    independent_set.append(v)
                    # Remove v and its neighbors from remaining
                    new_remaining = []
                    for u in remaining:
                        if u != v and not adj[v][u]:
                            new_remaining.append(u)
                    remaining = new_remaining
                
                if len(independent_set) < 9:
                    # Success: no independent set of size 9
                    found = True
                    break
                
                # We have an independent set of size >= 9
                # We need to add an edge within it
                S = independent_set
                # Try random pairs in S
                random.shuffle(S)
                added_edge = False
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        u, v = S[i], S[j]
                        if adj[u][v]:
                            continue
                        
                        # Check if adding (u, v) creates a C5
                        # This happens if there is a path of length 4 between u and v in current graph
                        # Path: u - x - y - z - v
                        # We need to check if there exists x, y, z such that
                        # (u,x), (x,y), (y,z), (z,v) are edges in current graph.
                        # And vertices are distinct.
                        
                        creates_c5 = False
                        # Iterate over neighbors of u
                        neighbors_u = [x for x in range(n) if adj[u][x]]
                        for x in neighbors_u:
                            # x != v (since u,v not connected yet, and path length 4 means x != v)
                            # Actually, if x == v, path length 1.
                            if x == v:
                                continue
                            # Iterate over neighbors of x (excluding u)
                            neighbors_x = [y for y in range(n) if adj[x][y] and y != u]
                            for y in neighbors_x:
                                if y == v:
                                    continue
                                # Iterate over neighbors of y (excluding x)
                                neighbors_y = [z for z in range(n) if adj[y][z] and z != x]
                                for z in neighbors_y:
                                    if z == u:
                                        continue
                                    if z == v:
                                        # Path u-x-y-z-v (length 4)
                                        # Check distinctness: u, x, y, z, v
                                        # x != u, y != u/x, z != u/x/y, v != u/x/y/z
                                        # We already checked x!=v, y!=v, z!=v.
                                        # We need to ensure x, y, z are distinct and not u, v.
                                        # x != u (neighbor), y != u (checked), z != u (checked)
                                        # y != x (checked)
                                        # z != x (checked)
                                        # z != y (neighbor)
                                        # So distinctness is mostly handled by checks.
                                        creates_c5 = True
                                        break
                                    # If z is not v, we haven't completed the path to v.
                                    # Wait, the path must end at v.
                                    # So we only care if z is connected to v.
                                    # The loop structure above checks if z is connected to v?
                                    # No, I just checked if z == v.
                                    # I need to check if adj[z][v] is True.
                                    if adj[z][v]:
                                        # Path u-x-y-z-v
                                        # Distinctness:
                                        # u, x, y, z, v
                                        # x != u, v
                                        # y != u, x, v
                                        # z != u, x, y, v
                                        # My checks:
                                        # x != v
                                        # y != u, x
                                        # z != u, x, y
                                        # And I am inside "if adj[z][v]", so z is connected to v.
                                        # I need to ensure z != v.
                                        if z != v:
                                            creates_c5 = True
                                            break
                                    if creates_c5:
                                        break
                                if creates_c5:
                                    break
                            if creates_c5:
                                break
                        if creates_c5:
                            continue
                        
                        # If no C5 created, add the edge
                        adj[u][v] = True
                        adj[v][u] = True
                        edges.append([u, v])
                        added_edge = True
                        break
                    if added_edge:
                        break
                
                if not added_edge:
                    # Stuck: no edge in S can be added without creating C5
                    success = False
                    break
            
            if found:
                return {'n': n, 'edges': edges}
        
        # If we exhausted restarts for this n, try smaller n
    
    # Fallback to seed program (n=28)
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}

# EVOLVE-BLOCK-END
