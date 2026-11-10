import random
import time
import itertools

def is_Q3(vertices, edges_set):
    # vertices: frozenset or list of 8 vertices
    # edges_set: set of tuples (u, v) with u < v
    
    # 1. Count edges in induced subgraph
    induced_edges = []
    for u, v in itertools.combinations(vertices, 2):
        if (u, v) in edges_set:
            induced_edges.append((u, v))
            
    if len(induced_edges) != 12:
        return False
        
    # 2. Check degrees
    degrees = {v: 0 for v in vertices}
    for u, v in induced_edges:
        degrees[u] += 1
        degrees[v] += 1
        
    if any(d != 3 for d in degrees.values()):
        return False
        
    # 3. Check Bipartite
    adj = {v: [] for v in vertices}
    for u, v in induced_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    color = {}
    for start in vertices:
        if start not in color:
            color[start] = 0
            queue = [start]
            while queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False
                        
    # 4. Connectedness is implied by 3-regular bipartite on 8 vertices
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_edges = set()
    best_count = 0
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Try multiple random restarts
    while time.time() < deadline:
        current_edges = set()
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            
            # Tentatively add
            current_edges.add((u, v))
            
            # Check if this creates a Q3 where (u, v) is an edge
            # u and v must have at least 2 common neighbors in the Q3
            neighbors_u = set()
            for edge in current_edges:
                if edge[0] == u: neighbors_u.add(edge[1])
                elif edge[1] == u: neighbors_u.add(edge[0])
                
            neighbors_v = set()
            for edge in current_edges:
                if edge[0] == v: neighbors_v.add(edge[1])
                elif edge[1] == v: neighbors_v.add(edge[0])
                
            common = neighbors_u.intersection(neighbors_v)
            
            found_cube = False
            if len(common) >= 2:
                # Iterate over pairs of common neighbors
                common_list = list(common)
                for c1, c2 in itertools.combinations(common_list, 2):
                    # The Q3 must contain u, v, c1, c2.
                    # The other 4 vertices are chosen from the remaining 12.
                    # Remaining vertices: all except u, v, c1, c2
                    candidates = [x for x in range(n) if x not in (u, v, c1, c2)]
                    
                    # Pick 4 from 12: C(12, 4) = 495
                    for w_combo in itertools.combinations(candidates, 4):
                        vertices = frozenset([u, v, c1, c2] + list(w_combo))
                        if is_Q3(vertices, current_edges):
                            found_cube = True
                            break
                    if found_cube:
                        break
                            
            if found_cube:
                current_edges.remove((u, v))
            # else keep
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges

    return [[u, v] for u, v in best_edges]
