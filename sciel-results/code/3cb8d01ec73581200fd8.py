import time
import random
from itertools import combinations

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    def time_ok():
        return time.time() < deadline
    
    best_edges = []
    best_count = 0
    
    # Precompute all 4-cycles for fast violation checking
    # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da (in some order)
    # We'll represent a 4-cycle by its 4 edges (as frozensets of vertex pairs)
    
    def get_valid_colors_fast(adj, color_mat, neighbors, u, v):
        """
        Check which colors are valid for edge (u,v) given current graph.
        For each 4-cycle containing edge (u,v), the other 3 edges must have distinct colors
        that are all different from the color assigned to (u,v).
        
        A 4-cycle containing edge (u,v) has the form: u-v-w-x-u or u-v-x-w-u
        where w is a neighbor of v (w != u) and x is a neighbor of u (x != v)
        and edge (w,x) exists.
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
                # The 4-cycle u-v-w-x-u has edges (u,v), (v,w), (w,x), (x,u)
                # For this to be a valid rainbow 4-cycle, all 4 colors must be distinct.
                # So color(u,v) must differ from c_vw, c_wx, c_xu.
                # Also c_vw, c_wx, c_xu must be mutually distinct (otherwise the cycle
                # is already non-rainbow regardless of our choice).
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    # This 4-cycle is already invalid; no color for (u,v) can fix it
                    # unless we're only checking if adding (u,v) creates a new violation.
                    # Actually, if the existing 3 edges already have a repeat, then
                    # adding (u,v) with any color makes the 4-cycle have at most 3 distinct
                    # colors among the 4, so it's still not rainbow. But wait - the constraint
                    # is that EVERY 4-cycle must be rainbow. If there's already a non-rawbow
                    # 4-cycle in the graph, the graph is already invalid. But we're building
                    # incrementally, so we assume the current graph is valid.
                    # If c_vw, c_wx, c_xu are not all distinct, then no matter what color
                    # we assign to (u,v), the 4-cycle won't be rainbow. So we should not
                    # allow adding (u,v) at all in this case.
                    return set()
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
        return set(range(num_colors)) - forbidden
    
    def count_violations(adj, color_mat, neighbors):
        """Count the number of 4-cycles that are not rainbow."""
        violations = 0
        for u in range(n):
            for v in neighbors[u]:
                if v <= u:
                    continue
                for w in neighbors[v]:
                    if w == u:
                        continue
                    for x in neighbors[u]:
                        if x == v or x == w:
                            continue
                        if not adj[w][x]:
                            continue
                        # Check 4-cycle u-v-w-x-u
                        c1 = color_mat[u][v]
                        c2 = color_mat[v][w]
                        c3 = color_mat[w][x]
                        c4 = color_mat[x][u]
                        if len({c1, c2, c3, c4}) != 4:
                            violations += 1
        return violations
    
    def run_greedy(seed, prefer_formula=True):
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not time_ok():
                break
            valid = get_valid_colors_fast(adj, color_mat, neighbors, u, v)
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
        """Local search that adds edges and recolors to maximize edge count."""
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        
        for u, v, c in edges:
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        cur_edges = [e[:] for e in edges]
        cur_count = len(cur_edges)
        
        # Build edge list for recoloring
        edge_list = [(u, v) for u, v, c in edges]
        
        for iteration in range(300):
            if not time_ok():
                break
            
            # Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            if not missing:
                break
            random.shuffle(missing)
            
            added = False
            for u, v in missing[:150]:
                valid = get_valid_colors_fast(adj, color_mat, neighbors, u, v)
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
                # Try recoloring to enable more edge additions
                improved = False
                random.shuffle(edge_list)
                for eu, ev in edge_list[:80]:
                    old_c = color_mat[eu][ev]
                    other_colors = [c for c in range(num_colors) if c != old_c]
                    random.shuffle(other_colors)
                    for new_c in other_colors[:15]:
                        # Check if recoloring (eu, ev) to new_c keeps graph valid
                        # and potentially enables more edge additions
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
                                if new_c in (c_ev_w, c_wx, c_xu):
                                    valid = False
                                    break
                                if c_ev_w in (c_wx, c_xu) or c_wx == c_xu:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        # Accept the recolor
                        color_mat[eu][ev] = new_c
                        color_mat[ev][eu] = new_c
                        for i, e in enumerate(cur_edges):
                            if e[0] == eu and e[1] == ev:
                                e[2] = new_c
                                break
                        improved = True
                        break
                    if improved:
                        break
                
                if not improved:
                    break
        
        return cur_edges
    
    # Phase 1: Multiple greedy runs with different strategies
    for seed in range(2000):
        if not time_ok():
            break
        prefer = (seed % 3 != 0)  # 2/3 with formula, 1/3 random
        edges = run_greedy(seed, prefer_formula=prefer)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [e[:] for e in edges]
    
    # Phase 2: Local search on best solutions
    if best_edges and time_ok():
        for seed in range(300):
            if not time_ok():
                break
            improved = local_search(best_edges, seed)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = [e[:] for e in improved]
    
    # Phase 3: More aggressive local search with more iterations
    if best_edges and time_ok():
        for seed in range(200):
            if not time_ok():
                break
            improved = local_search(best_edges, seed + 10000)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = [e[:] for e in improved]
    
    return best_edges
