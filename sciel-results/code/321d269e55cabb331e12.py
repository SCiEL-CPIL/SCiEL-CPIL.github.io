import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 15
    
    # Helper: check if adding edge (u,v) to current graph violates constraint
    # Given adjacency matrix and degrees, check all paths s-u-v-t and s-v-u-t
    def is_valid_with_edge(adj, deg, u, v):
        # After adding (u,v), deg[u] and deg[v] increase by 1
        new_deg = list(deg)
        new_deg[u] += 1
        new_deg[v] += 1
        
        # Check paths where u and v are the middle edge
        # Path s-u-v-t: s adjacent to u (s != v), t adjacent to v (t != u), s != t
        # Need new_deg[s] != new_deg[t] for all such s,t
        for s in range(n):
            if s == u or s == v:
                continue
            if not adj[s][u]:
                continue
            for t in range(n):
                if t == u or t == v:
                    continue
                if not adj[t][v]:
                    continue
                if s == t:
                    continue
                if new_deg[s] == new_deg[t]:
                    return False
        # Path s-v-u-t
        for s in range(n):
            if s == u or s == v:
                continue
            if not adj[s][v]:
                continue
            for t in range(n):
                if t == u or t == v:
                    continue
                if not adj[t][u]:
                    continue
                if s == t:
                    continue
                if new_deg[s] == new_deg[t]:
                    return False
        
        return True
    
    # Build initial complete multipartite graph with 5 parts of 3
    # Parts: [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]
    parts = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]
    
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            # Connect if in different parts
            part_i = None
            part_j = None
            for p_idx, part in enumerate(parts):
                if i in part:
                    part_i = p_idx
                if j in part:
                    part_j = p_idx
            if part_i != part_j:
                adj[i][j] = True
                adj[j][i] = True
    
    deg = [sum(adj[i]) for i in range(n)]
    
    # Greedily try to add missing edges
    missing = [(i,j) for i in range(n) for j in range(i+1,n) if not adj[i][j]]
    random.shuffle(missing)
    
    best_edges = sum(deg)//2
    best_adj = [row[:] for row in adj]
    best_deg = list(deg)
    
    for (u,v) in missing:
        if time.time() > deadline:
            break
        if is_valid_with_edge(adj, deg, u, v):
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
            current_edges = sum(deg)//2
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [row[:] for row in adj]
                best_deg = list(deg)
    
    # Also try a few random restarts with different initial structures
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random partition into parts
        verts = list(range(n))
        random.shuffle(verts)
        # Try different numbers of parts
        num_parts = random.choice([3,4,5,6])
        sizes = [0]*num_parts
        for i, v in enumerate(verts):
            # Assign to smallest part
            min_idx = min(range(num_parts), key=lambda k: sizes[k])
            parts_r = [[] for _ in range(num_parts)]
            for v2 in verts[:i]:
                min_idx2 = min(range(num_parts), key=lambda k: len(parts_r[k]))
                parts_r[min_idx2].append(v2)
            parts_r[min_idx].append(v)
            parts_r = [p for p in parts_r if p]
            break
        # Redo properly
        parts_r = [[] for _ in range(num_parts)]
        for v in verts:
            min_idx = min(range(num_parts), key=lambda k: len(parts_r[k]))
            parts_r[min_idx].append(v)
        parts_r = [p for p in parts_r if p]
        
        adj_r = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                part_i = None
                part_j = None
                for p_idx, part in enumerate(parts_r):
                    if i in part:
                        part_i = p_idx
                    if j in part:
                        part_j = p_idx
                if part_i != part_j:
                    adj_r[i][j] = True
                    adj_r[j][i] = True
        
        deg_r = [sum(adj_r[i]) for i in range(n)]
        
        missing_r = [(i,j) for i in range(n) for j in range(i+1,n) if not adj_r[i][j]]
        random.shuffle(missing_r)
        
        for (u,v) in missing_r:
            if time.time() > deadline:
                break
            if is_valid_with_edge(adj_r, deg_r, u, v):
                adj_r[u][v] = True
                adj_r[v][u] = True
                deg_r[u] += 1
                deg_r[v] += 1
        
        current_edges = sum(deg_r)//2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj_r]
            best_deg = list(deg_r)
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    # Ensure types are correct
    return [[int(u), int(v)] for u, v in edges]
