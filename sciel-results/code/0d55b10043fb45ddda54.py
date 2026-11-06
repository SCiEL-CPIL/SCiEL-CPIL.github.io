import random
import time

def construct():
    n = 31
    target_edges = 90
    deadline = time.time() + 110 # 10s margin
    
    best_edges = None
    best_c4_count = float('inf')
    
    for attempt in range(50): # Try a few random bases
        if time.time() > deadline:
            break
            
        # Generate a random C4-free graph with 90 edges
        adj = [[False] * n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        edges = []
        
        # Shuffle all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        edge_count = 0
        for u, v in all_edges:
            if edge_count >= target_edges:
                break
            if time.time() > deadline:
                break
                
            # Check if adding (u, v) creates any C4
            # New C4s = number of edges between N(u) and N(v)
            new_c4s = 0
            for w1 in neighbors[u]:
                for w2 in neighbors[v]:
                    if w1 != w2 and adj[w1][w2]:
                        new_c4s += 1
                        
            if new_c4s == 0:
                # Add edge
                adj[u][v] = True
                adj[v][u] = True
                neighbors[u].append(v)
                neighbors[v].append(u)
                edges.append([u, v])
                edge_count += 1
                
        if edge_count < target_edges:
            continue
            
        # Now find the best edge to add to reach 91 edges
        # We want to minimize new C4s
        min_new_c4s = float('inf')
        best_edge_to_add = None
        
        for u, v in all_edges:
            if adj[u][v]:
                continue
            if time.time() > deadline:
                break
                
            new_c4s = 0
            for w1 in neighbors[u]:
                for w2 in neighbors[v]:
                    if w1 != w2 and adj[w1][w2]:
                        new_c4s += 1
                            
            if new_c4s < min_new_c4s:
                min_new_c4s = new_c4s
                best_edge_to_add = [u, v]
                
        if best_edge_to_add:
            edges.append(best_edge_to_add)
            
            # Calculate total C4s
            # It is just min_new_c4s since base was 0
            if min_new_c4s < best_c4_count:
                best_c4_count = min_new_c4s
                best_edges = edges
                
        if best_c4_count == 1:
            break # Can't do better than 1
            
    if best_edges:
        return best_edges
    else:
        # Fallback to polarity graph if random search fails
        # (Re-insert polarity graph logic or return empty/invalid?)
        # The task requires valid output.
        # We should ensure at least one valid graph is returned.
        # The polarity graph logic from the original code is safe.
        # Let's keep the original polarity graph construction as a fallback.
        
        q = 5
        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = pow(x, q-2, q)
                        rep = (1, (y * inv_x) % q, (z * inv_x) % q)
                    elif y != 0:
                        inv_y = pow(y, q-2, q)
                        rep = (0, 1, (z * inv_y) % q)
                    else:
                        rep = (0, 0, 1)
                    
                    if rep not in seen:
                        seen.add(rep)
                        points.append(rep)
        
        n = len(points)
        point_to_idx = {p: i for i, p in enumerate(points)}
        
        adj = [[False] * n for _ in range(n)]
        edges = []
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                val = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
                if val == 0:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.append([i, j])
                    
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    neighbors[i].append(j)
                    
        min_paths = float('inf')
        best_edge = None
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    count = 0
                    for w1 in neighbors[i]:
                        for w2 in neighbors[j]:
                            if w1 != w2 and adj[w1][w2]:
                                count += 1
                    if count < min_paths:
                        min_paths = count
                        best_edge = [i, j]
                        
        if best_edge:
            edges.append(best_edge)
            
        return edges
