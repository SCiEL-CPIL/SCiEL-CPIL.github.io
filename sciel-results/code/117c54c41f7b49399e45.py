import random
import time
import sys

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    max_time = 110.0  # Safety margin

    def random_graph():
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        return set(all_edges[:target_edges])
    
    edges = random_graph()
    
    def get_score(edge_set):
        adj = [[] for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    for w in range(v+1, n):
                        if w in adj[u] and w in adj[v]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    best_score = get_score(edges)
    best_edges = set(edges)
    
    # Local search
    while time.time() - start_time < max_time:
        current_score = get_score(edges)
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
            
        if best_score <= 24:
            break # Found a very good solution
            
        # Find the worst triangle
        adj = [[] for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            
        worst_tri = None
        max_s = 0
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    for w in range(v+1, n):
                        if w in adj[u] and w in adj[v]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_s:
                                max_s = s
                                worst_tri = (u, v, w)
        
        if not worst_tri:
            break
            
        u, v, w = worst_tri
        
        # Try to swap an edge incident to u, v, or w
        # with a non-edge
        improved = False
        
        # Collect edges to remove
        remove_candidates = []
        for vert in (u, v, w):
            for neighbor in adj[vert]:
                if vert < neighbor:
                    remove_candidates.append((vert, neighbor))
                else:
                    remove_candidates.append((neighbor, vert))
        
        # Collect non-edges to add
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edges:
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        non_edges_to_check = non_edges[:50]
        
        for rem_edge in remove_candidates:
            for add_edge in non_edges_to_check:
                if set(rem_edge) == set(add_edge):
                    continue
                
                # Apply swap
                new_edges = set(edges)
                new_edges.remove(rem_edge)
                new_edges.add(add_edge)
                
                new_score = get_score(new_edges)
                
                if new_score < current_score:
                    edges = new_edges
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try a random swap to escape local minimum
            rem_edge = random.choice(list(edges))
            if non_edges:
                add_edge = random.choice(non_edges)
                if set(rem_edge) != set(add_edge):
                    new_edges = set(edges)
                    new_edges.remove(rem_edge)
                    new_edges.add(add_edge)
                    new_score = get_score(new_edges)
                    if new_score <= best_score + 2:
                        edges = new_edges

    # Final polish: exact local search on best_edges
    if time.time() - start_time < max_time - 10:
        edges = set(best_edges)
        current_score = get_score(edges)
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
        
        # More aggressive local search
        while time.time() - start_time < max_time - 5:
            adj = [[] for _ in range(n)]
            deg = [0] * n
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
            
            # Find all triangles and their sums
            triangles = []
            for u in range(n):
                for v in range(u+1, n):
                    if v in adj[u]:
                        for w in range(v+1, n):
                            if w in adj[u] and w in adj[v]:
                                s = deg[u] + deg[v] + deg[w]
                                triangles.append((s, (u, v, w)))
            
            if not triangles:
                break
            
            # Sort by sum descending
            triangles.sort(reverse=True)
            
            improved = False
            for s, tri in triangles:
                if s <= best_score:
                    break  # No need to check lower-sum triangles
                
                u, v, w = tri
                # Try removing edges from this triangle
                tri_edges = [(u,v), (u,w), (v,w)]
                for rem_edge in tri_edges:
                    for add_edge in non_edges[:30]:
                        if set(rem_edge) == set(add_edge):
                            continue
                        new_edges = set(edges)
                        new_edges.remove(rem_edge)
                        new_edges.add(add_edge)
                        new_score = get_score(new_edges)
                        if new_score < current_score:
                            edges = new_edges
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break

    return {'n': n, 'edges': [list(e) for e in best_edges]}
