import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 117
    
    def time_ok():
        return time.time() < deadline
    
    best_edges = []
    best_count = 0
    
    def run_greedy(seed, prefer_formula=True):
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        def get_valid(u, v):
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
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not time_ok():
                break
            valid = get_valid(u, v)
            if valid:
                if prefer_formula:
                    pref = (u + v) % num_colors
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
    
    def local_search(edges, seed):
        """Try to add more edges via recoloring and swaps."""
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
        
        def get_valid(u, v):
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
        
        def try_add_edge(u, v):
            nonlocal cur_count
            valid = get_valid(u, v)
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
                return True
            return False
        
        def try_recolor(eu, ev, new_c):
            """Check if recoloring edge (eu,ev) to new_c is valid."""
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
                    if new_c in (c_ev_w, c_wx, c_xu):
                        return False
                    if c_ev_w in (c_wx, c_xu) or c_wx == c_xu:
                        return False
            return True
        
        for iteration in range(500):
            if not time_ok():
                break
            
            # Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            if not missing:
                break
            random.shuffle(missing)
            
            added = False
            for u, v in missing[:200]:
                if try_add_edge(u, v):
                    added = True
                    break
            
            if not added:
                # Try recoloring to enable new edges
                improved = False
                random.shuffle(edge_list)
                for eu, ev in edge_list[:100]:
                    old_c = color_mat[eu][ev]
                    other_colors = [c for c in range(num_colors) if c != old_c]
                    random.shuffle(other_colors)
                    for new_c in other_colors[:15]:
                        if try_recolor(eu, ev, new_c):
                            # Check if this enables adding a new edge
                            best_new_edge = None
                            for i in range(n):
                                if i == eu or i == ev:
                                    continue
                                for j in range(i+1, n):
                                    if j == eu or j == ev:
                                        continue
                                    if not adj[i][j]:
                                        # Check if this edge could be added with current colors
                                        # (we won't actually add it yet, just check potential)
                                        pass
                            # Apply recolor
                            color_mat[eu][ev] = new_c
                            color_mat[ev][eu] = new_c
                            for i, e in enumerate(cur_edges):
                                if (e[0] == eu and e[1] == ev) or (e[0] == ev and e[1] == eu):
                                    e[2] = new_c
                                    break
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    # Try swap: remove an edge, recolor, add different edge
                    swap_improved = False
                    random.shuffle(edge_list)
                    for eu, ev in edge_list[:50]:
                        old_c = color_mat[eu][ev]
                        # Temporarily remove edge
                        adj[eu][ev] = False
                        adj[ev][eu] = False
                        color_mat[eu][ev] = -1
                        color_mat[ev][eu] = -1
                        neighbors[eu].discard(ev)
                        neighbors[ev].discard(eu)
                        cur_count -= 1
                        edge_list.remove((eu, ev))
                        
                        # Try to add two new edges
                        added_count = 0
                        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
                        random.shuffle(missing)
                        for u, v in missing[:50]:
                            if try_add_edge(u, v):
                                added_count += 1
                                if added_count >= 2:
                                    break
                        
                        if added_count >= 1:
                            # Keep the swap
                            swap_improved = True
                            break
                        else:
                            # Revert
                            adj[eu][ev] = True
                            adj[ev][eu] = True
                            color_mat[eu][ev] = old_c
                            color_mat[ev][eu] = old_c
                            neighbors[eu].add(ev)
                            neighbors[ev].add(eu)
                            cur_count += 1
                            edge_list.append((eu, ev))
                    
                    if not swap_improved and not improved:
                        break
        
        return cur_edges
    
    # Phase 1: Greedy with formula preference
    for seed in range(3000):
        if not time_ok():
            break
        edges = run_greedy(seed, prefer_formula=True)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [e[:] for e in edges]
    
    # Phase 2: Local search on best solutions
    if best_edges and time_ok():
        for seed in range(200):
            if not time_ok():
                break
            improved = local_search(best_edges, seed)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = [e[:] for e in improved]
    
    return best_edges
