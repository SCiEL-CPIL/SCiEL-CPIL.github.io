import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 20
    K = 20  # number of available colours
    
    def time_ok():
        return time.time() < deadline - 5
    
    def check_all_cycles(adj, color_mat):
        """Check if all 4-cycles are rainbow."""
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    continue
                for k in range(j + 1, n):
                    if not adj[i][k]:
                        continue
                    for l in range(k + 1, n):
                        if not adj[j][l] or not adj[i][l] or not adj[k][l]:
                            continue
                        # 4-cycle: i-j-l-k-i or i-k-l-j-i etc.
                        # Check cycle i-j-l-k-i
                        c1 = color_mat[i][j]
                        c2 = color_mat[j][l]
                        c3 = color_mat[l][k]
                        c4 = color_mat[k][i]
                        if len({c1, c2, c3, c4}) < 4:
                            return False
        return True
    
    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns the set of colors that can be assigned to edge (u, v)
        without violating the rainbow 4-cycle property.
        Assumes (u, v) is NOT currently in the graph.
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
                
                # Check if existing path colors are distinct
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
        
        return set(range(K)) - forbidden
    
    def run_greedy(seed, prefer_formula=True):
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not time_ok():
                break
            valid = get_valid_colors(u, v, adj, color_mat, neighbors)
            if valid:
                if prefer_formula:
                    pref = (u + v) % K
                    if pref in valid:
                        c = pref
                    else:
                        c = random.choice(list(valid))
                else:
                    c = random.choice(list(valid))
                adj[u][v] = True
                adj[v][u] = True
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
        
        return edges
    
    def local_search_with_recolor(edges, seed):
        """Try to add more edges via recoloring and edge addition."""
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edge_list = []
        
        for u, v, c in edges:
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edge_list.append((u, v))
        
        cur_edges = [e[:] for e in edges]
        cur_count = len(cur_edges)
        
        for iteration in range(300):
            if not time_ok():
                break
            
            # Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
            if not missing:
                break
            random.shuffle(missing)
            
            added = False
            for u, v in missing:
                valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid:
                    c = random.choice(list(valid))
                    adj[u][v] = True
                    adj[v][u] = True
                    color_mat[u][v] = c
                    color_mat[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    cur_edges.append([u, v, c])
                    cur_count += 1
                    edge_list.append((u, v))
                    added = True
                    break
            
            if not added:
                # Try recoloring to enable new edges
                improved = False
                random.shuffle(edge_list)
                for eu, ev in edge_list[:50]:
                    old_c = color_mat[eu][ev]
                    other_colors = [c for c in range(K) if c != old_c]
                    random.shuffle(other_colors)
                    
                    for new_c in other_colors[:10]:
                        # Check if recoloring (eu, ev) to new_c is valid
                        valid_recolor = True
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
                                    valid_recolor = False
                                    break
                                if c_ev_w == c_wx or c_ev_w == c_xu or c_wx == c_xu:
                                    valid_recolor = False
                                    break
                            if not valid_recolor:
                                break
                        if valid_recolor:
                            color_mat[eu][ev] = new_c
                            color_mat[ev][eu] = new_c
                            # Update cur_edges
                            for e in cur_edges:
                                if (e[0] == eu and e[1] == ev) or (e[0] == ev and e[1] == eu):
                                    e[2] = new_c
                                    break
                            improved = True
                            break
                    if improved:
                        break
        
        return cur_edges
    
    # Strategy 1: Try to find valid K_{10,10} coloring via linear forms
    m = 10
    left = list(range(m))
    right = list(range(m, 2 * m))
    
    best_edges = None
    best_count = 0
    
    # Try linear forms C[l][r] = (a*l + b*r) % K
    found_valid_bipartite = False
    for a in range(1, K):
        for b in range(1, K):
            if not time_ok():
                break
            colour_matrix = [[(a * l + b * r) % K for r in range(m)] for l in range(m)]
            
            # Check all 4-cycles in K_{m,m}
            valid = True
            for l1 in range(m):
                for l2 in range(l1 + 1, m):
                    for r1 in range(m):
                        for r2 in range(r1 + 1, m):
                            c1 = colour_matrix[l1][r1]
                            c2 = colour_matrix[l2][r1]
                            c3 = colour_matrix[l2][r2]
                            c4 = colour_matrix[l1][r2]
                            if len({c1, c2, c3, c4}) < 4:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                found_valid_bipartite = True
                best_count = m * m
                best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                             for l in range(m) for r in range(m)]
                break
        if found_valid_bipartite:
            break
    
    # Strategy 2: If bipartite coloring not found, use general greedy + local search
    if best_edges is None or best_count < 100:
        # Run multiple greedy attempts
        for seed in range(20):
            if not time_ok():
                break
            edges = run_greedy(seed, prefer_formula=True)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
        
        # Apply local search with recoloring to the best solution
        if best_edges and time_ok():
            improved = local_search_with_recolor(best_edges, seed=42)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved
    
    # Final verification
    if best_edges:
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        for u, v, c in best_edges:
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
        
        if not check_all_cycles(adj, color_mat):
            # Fallback: return empty if invalid (should not happen)
            return []
    
    return best_edges if best_edges else []
