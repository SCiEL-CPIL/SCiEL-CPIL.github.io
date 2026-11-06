import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible_edges = len(all_edges)
    
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
                
                # The 3 existing edges must have distinct colors among themselves
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
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
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
        
        return edges, adj, color_mat, neighbors
    
    def local_search_color(edges, adj, color_mat, neighbors, seed=None, max_iter=200):
        """
        Local search via color reassignment to add more edges.
        Alternates between adding missing edges and recoloring existing ones.
        """
        if seed is not None:
            random.seed(seed)
        
        current_adj = [row[:] for row in adj]
        current_color_mat = [row[:] for row in color_mat]
        current_neighbors = [set(s) for s in neighbors]
        current_edges = [e[:] for e in edges]
        current_count = len(current_edges)
        
        # Build edge lookup for fast recoloring
        edge_lookup = {}
        for i, (u, v, c) in enumerate(current_edges):
            edge_lookup[(u, v)] = i
            edge_lookup[(v, u)] = i
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Try to add a missing edge
            added = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not current_adj[i][j]]
            if not missing:
                break
            
            random.shuffle(missing)
            
            for u, v in missing[:80]:  # Limit attempts per iteration
                valid_colors = get_valid_colors(u, v, current_adj, current_color_mat, current_neighbors)
                if valid_colors:
                    c = random.choice(list(valid_colors))
                    current_adj[u][v] = 1
                    current_adj[v][u] = 1
                    current_color_mat[u][v] = c
                    current_color_mat[v][u] = c
                    current_neighbors[u].add(v)
                    current_neighbors[v].add(u)
                    current_edges.append([u, v, c])
                    edge_lookup[(u, v)] = len(current_edges) - 1
                    edge_lookup[(v, u)] = len(current_edges) - 1
                    current_count += 1
                    added = True
                    break
            
            if not added:
                # Try recoloring existing edges to free up colors
                improved = False
                edge_list = [(u, v) for u, v, _ in current_edges]
                random.shuffle(edge_list)
                
                for eu, ev in edge_list[:40]:
                    old_c = current_color_mat[eu][ev]
                    # Try a different color for this edge
                    other_colors = [c for c in range(num_colors) if c != old_c]
                    random.shuffle(other_colors)
                    
                    for new_c in other_colors[:8]:
                        # Check if recoloring is valid: verify all 4-cycles remain rainbow
                        valid = True
                        for w in current_neighbors[ev]:
                            if w == eu:
                                continue
                            c_ev_w = current_color_mat[ev][w]
                            if new_c == c_ev_w:
                                valid = False
                                break
                            for x in current_neighbors[eu]:
                                if x == ev or x == w:
                                    continue
                                if not current_adj[w][x]:
                                    continue
                                c_wx = current_color_mat[w][x]
                                c_xu = current_color_mat[x][eu]
                                if new_c == c_wx or new_c == c_xu:
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
                        current_color_mat[eu][ev] = new_c
                        current_color_mat[ev][eu] = new_c
                        idx = edge_lookup[(eu, ev)]
                        current_edges[idx][2] = new_c
                        improved = True
                        break
                    if improved:
                        break
                
                if not improved:
                    # Stuck - try a few random recolors as perturbation
                    for _ in range(5):
                        if time.time() > deadline - 10:
                            break
                        eu, ev = random.choice(edge_list)
                        old_c = current_color_mat[eu][ev]
                        new_c = random.choice([c for c in range(num_colors) if c != old_c])
                        current_color_mat[eu][ev] = new_c
                        current_color_mat[ev][eu] = new_c
                        idx = edge_lookup[(eu, ev)]
                        current_edges[idx][2] = new_c
        
        # Rebuild edge list from adj/color_mat to be sure
        final_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if current_adj[u][v]:
                    final_edges.append([u, v, current_color_mat[u][v]])
        
        return final_edges
    
    restarts = 0
    while time.time() < deadline:
        edges, adj, color_mat, neighbors = run_greedy(seed=restarts)
        greedy_count = len(edges)
        
        # Try local search to improve
        if time.time() < deadline - 15:
            improved_edges = local_search_color(edges, adj, color_mat, neighbors, seed=restarts, max_iter=150)
            if improved_edges and len(improved_edges) > greedy_count:
                edges = improved_edges
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        if best_count == total_possible_edges:
            break
        
        restarts += 1
    
    return best_edges
