import time
import random
from collections import defaultdict

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 118

    def time_ok():
        return time.time() < deadline

    best_edges = []
    best_count = 0

    # Precompute all potential 4-cycles (as edge tuples) for fast checking
    # A 4-cycle is defined by 4 vertices a,b,c,d with edges (a,b),(b,c),(c,d),(d,a)
    # We store them as frozensets of edge pairs (min,max)
    all_4cycles = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # 3 ways to form a 4-cycle from 4 vertices
                    # (a,b,c,d): edges ab, bc, cd, da
                    e1 = (a, b)
                    e2 = (b, c)
                    e3 = (c, d)
                    e4 = (d, a)
                    all_4cycles.append(frozenset([e1, e2, e3, e4]))
                    # (a,b,d,c): edges ab, bd, dc, ca
                    e1 = (a, b)
                    e2 = (b, d)
                    e3 = (d, c)
                    e4 = (c, a)
                    all_4cycles.append(frozenset([e1, e2, e3, e4]))
                    # (a,c,b,d): edges ac, cb, bd, da
                    e1 = (a, c)
                    e2 = (c, b)
                    e3 = (b, d)
                    e4 = (d, a)
                    all_4cycles.append(frozenset([e1, e2, e3, e4]))

    # For each edge, precompute which 4-cycles contain it
    edge_to_cycles = defaultdict(list)
    for i, cyc in enumerate(all_4cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    def get_valid_colors(adj, color_mat, u, v):
        """Return set of valid colors for edge (u,v)."""
        if adj[u][v]:
            return set()
        
        forbidden = set()
        # Check all 4-cycles containing edge (u,v)
        for cyc_idx in edge_to_cycles[(min(u,v), max(u,v))]:
            cyc = all_4cycles[cyc_idx]
            # Get the other 3 edges in this cycle
            other_edges = []
            for e in cyc:
                if e != (u, v) and e != (v, u):
                    # Normalize
                    e_norm = (min(e), max(e))
                    if e_norm == (min(u,v), max(u,v)):
                        continue
                    other_edges.append(e_norm)
            
            if len(other_edges) != 3:
                continue
            
            # All 3 other edges must exist in the graph
            all_exist = True
            colors = []
            for oe in other_edges:
                if adj[oe[0]][oe[1]]:
                    colors.append(color_mat[oe[0]][oe[1]])
                else:
                    all_exist = False
                    break
            
            if not all_exist:
                continue
            
            # The 3 colors must be distinct and none can equal the new color
            if colors[0] == colors[1] or colors[0] == colors[2] or colors[1] == colors[2]:
                # This cycle is already violated - but that shouldn't happen in a valid state
                # Actually, if the current graph is valid, this can't happen
                # But for safety, if we're in an invalid state, we should still forbid
                # Actually, the constraint is that in the FINAL graph, every 4-cycle is rainbow
                # So if the other 3 edges have duplicate colors, adding any color to (u,v)
                # won't fix that cycle. But the cycle is already "broken" in the sense that
                # it's not rainbow. However, our invariant is that the current graph is valid.
                # So this case shouldn't occur.
                continue
            
            # All 3 colors are distinct, so new color must differ from all 3
            forbidden.add(colors[0])
            forbidden.add(colors[1])
            forbidden.add(colors[2])
        
        return set(range(num_colors)) - forbidden

    def run_greedy(seed):
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not time_ok():
                break
            valid = get_valid_colors(adj, color_mat, u, v)
            if valid:
                c = random.choice(list(valid))
                adj[u][v] = True
                adj[v][u] = True
                color_mat[u][v] = c
                color_mat[v][u] = c
                edges.append([u, v, c])
        
        return edges

    def local_search(edges, seed):
        """Try to add more edges and recolor to find better solutions."""
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        cur_edges = [e[:] for e in edges]
        cur_count = len(cur_edges)
        
        for u, v, c in cur_edges:
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
        
        for iteration in range(500):
            if not time_ok():
                break
            
            # Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            if missing:
                random.shuffle(missing)
                added = False
                for u, v in missing[:200]:
                    valid = get_valid_colors(adj, color_mat, u, v)
                    if valid:
                        c = random.choice(list(valid))
                        adj[u][v] = True
                        adj[v][u] = True
                        color_mat[u][v] = c
                        color_mat[v][u] = c
                        cur_edges.append([u, v, c])
                        cur_count += 1
                        added = True
                        break
                if added:
                    continue
            
            # Try recoloring to enable more additions
            if cur_edges:
                random.shuffle(cur_edges)
                improved = False
                for eu, ev, old_c in cur_edges[:100]:
                    other_colors = [c for c in range(num_colors) if c != old_c]
                    random.shuffle(other_colors)
                    for new_c in other_colors[:15]:
                        # Check if recoloring is valid
                        valid = True
                        for cyc_idx in edge_to_cycles[(min(eu,ev), max(eu,ev))]:
                            cyc = all_4cycles[cyc_idx]
                            other_edges = []
                            for e in cyc:
                                e_norm = (min(e), max(e))
                                if e_norm != (min(eu,ev), max(eu,ev)):
                                    other_edges.append(e_norm)
                            
                            if len(other_edges) != 3:
                                continue
                            
                            all_exist = True
                            colors = []
                            for oe in other_edges:
                                if adj[oe[0]][oe[1]]:
                                    colors.append(color_mat[oe[0]][oe[1]])
                                else:
                                    all_exist = False
                                    break
                            
                            if not all_exist:
                                continue
                            
                            if colors[0] == colors[1] or colors[0] == colors[2] or colors[1] == colors[2]:
                                continue
                            
                            if new_c in colors:
                                valid = False
                                break
                        
                        if valid:
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

    # Phase 1: Multiple greedy runs
    for seed in range(5000):
        if not time_ok():
            break
        edges = run_greedy(seed)
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
    
    return best_edges
