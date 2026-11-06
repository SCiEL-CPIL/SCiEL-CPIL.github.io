import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns a set of valid colors for edge (u, v).
        If the set is empty, the edge cannot be added.
        """
        forbidden = set()
        
        for w in neighbors[v]:
            if w == u:
                continue
            c_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v or x == w:
                    continue
                if not adj[w][x]:
                    continue
                
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
        
        return set(range(num_colors)) - forbidden
    
    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            valid_colors = get_valid_colors(u, v, adj, color_mat, neighbors)
            
            if valid_colors:
                c = random.choice(list(valid_colors))
                
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            
        return edges
    
    def local_search_color(edges, seed=None, max_iter=100):
        """
        Local search via color reassignment.
        Try to recolor edges to allow adding more edges.
        """
        if seed is not None:
            random.seed(seed)
        
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        
        edge_list = []
        for u, v, c in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edge_list.append((u, v))
        
        current_edges = [e[:] for e in edges]
        current_count = len(current_edges)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Try to add a missing edge
            added = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            if not missing:
                break
            
            random.shuffle(missing)
            
            for u, v in missing[:50]:  # Limit attempts per iteration
                valid_colors = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid_colors:
                    c = random.choice(list(valid_colors))
                    adj[u][v] = 1
                    adj[v][u] = 1
                    color_mat[u][v] = c
                    color_mat[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    current_edges.append([u, v, c])
                    current_count += 1
                    added = True
                    break
            
            if not added:
                # Try recoloring existing edges to free up colors
                improved = False
                random.shuffle(edge_list)
                
                for eu, ev in edge_list[:30]:
                    old_c = color_mat[eu][ev]
                    # Try a different color for this edge
                    other_colors = [c for c in range(num_colors) if c != old_c]
                    random.shuffle(other_colors)
                    
                    for new_c in other_colors[:5]:
                        # Check if recoloring is valid
                        valid = True
                        for w in neighbors[ev]:
                            if w == eu:
                                continue
                            c_ev_w = color_mat[ev][w]
                            for x in neighbors[eu]:
                                if x == ev or x == w:
                                    continue
                                if not adj[w][x]:
                                    continue
                                c_wx = color_mat[w][x]
                                c_xu = color_mat[x][eu]
                                if new_c == c_ev_w or new_c == c_wx or new_c == c_xu:
                                    valid = False
                                    break
                                if c_ev_w == c_wx or c_ev_w == c_xu or c_wx == c_xu:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        # Apply recoloring
                        color_mat[eu][ev] = new_c
                        color_mat[ev][eu] = new_c
                        for i, e in enumerate(current_edges):
                            if e[0] == eu and e[1] == ev:
                                current_edges[i][2] = new_c
                                break
                        improved = True
                        break
                    if improved:
                        break
                
                if not improved:
                    break
        
        return current_edges
    
    restarts = 0
    while time.time() < deadline:
        edges = run_greedy(seed=restarts)
        
        if time.time() < deadline - 15:
            improved_edges = local_search_color(edges, seed=restarts, max_iter=50)
            if improved_edges and len(improved_edges) > len(edges):
                edges = improved_edges
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        if best_count == 190:
            break
            
        restarts += 1
    
    return best_edges
