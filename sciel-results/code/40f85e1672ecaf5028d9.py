import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n = 20
    num_colors = 20
    
    def check_4_cycles(edge_set, color_map):
        """Check if all 4-cycles are rainbow. Returns True if valid."""
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all 4-cycles and check they're rainbow
        for a in range(n):
            for b in adj[a]:
                if b <= a:
                    continue
                for c in adj[b]:
                    if c == a or c <= b:
                        continue
                    for d in adj[c]:
                        if d == b or d == a:
                            continue
                        if d > c:
                            continue
                        if d in adj[a]:
                            # Found a 4-cycle: a-b-c-d-a
                            # Check colors are all distinct
                            colors = [color_map[(min(a,b),max(a,b))], 
                                      color_map[(min(b,c),max(b,c))],
                                      color_map[(min(c,d),max(c,d))],
                                      color_map[(min(d,a),max(d,a))]]
                            if len(set(colors)) != 4:
                                return False
        return True
    
    def try_add_edge(edge_set, color_map, u, v):
        """Try to add edge (u,v) with some color that keeps all 4-cycles rainbow."""
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return None
        
        # Find colors that would create a non-rainbow 4-cycle
        bad_colors = set()
        
        # For each pair of neighbors x of u and y of v such that x-y is an edge,
        # we'd form a 4-cycle u-x-y-v-u (if u-v is added)
        # Actually, the 4-cycle would be u-v-y-x-u or similar
        # Let me think more carefully:
        # Adding edge (u,v). For any x in adj[u] and y in adj[v] where (x,y) is an edge,
        # and x != v, y != u, we form a 4-cycle: u-v-y-x-u
        # The four edges are: (u,v) [new], (v,y), (y,x), (x,u)
        
        for x in adj_list[u]:
            for y in adj_list[v]:
                if x == v or y == u:
                    continue
                if (min(x,y), max(x,y)) in edge_set:
                    # 4-cycle: u-v-y-x-u
                    c_vy = color_map[(min(v,y), max(v,y))]
                    c_yx = color_map[(min(x,y), max(x,y))]
                    c_xu = color_map[(min(x,u), max(x,u))]
                    for c in range(num_colors):
                        if c == c_vy or c == c_yx or c == c_xu:
                            bad_colors.add(c)
        
        # Also need to consider other 4-cycles that might be formed
        # Actually, any 4-cycle involving the new edge (u,v) must have the form
        # u-v-y-x-u where x is neighbor of u and y is neighbor of v and x-y is an edge
        # OR u-v-x-y-u (same thing, just different order)
        
        # Try colors in order
        for c in range(num_colors):
            if c not in bad_colors:
                return c
        return None
    
    # Main search: try to build a dense graph
    best_edges = []
    best_count = 0
    
    for restart in range(50):
        if time.time() > deadline - 20:
            break
        
        edge_set = set()
        color_map = {}
        adj_list = {i: set() for i in range(n)}
        
        # Start with a random subset of edges
        all_edges = list(combinations(range(n), 2))
        random.shuffle(all_edges)
        
        # Try to add edges greedily
        for u, v in all_edges:
            if time.time() > deadline - 10:
                break
            
            key = (min(u,v), max(u,v))
            if key in edge_set:
                continue
            
            c = try_add_edge(edge_set, color_map, u, v)
            if c is not None:
                edge_set.add(key)
                color_map[key] = c
                adj_list[u].add(v)
                adj_list[v].add(u)
        
        # Local search: try to recolor edges to allow adding more
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for u, v in list(edge_set):
                key = (min(u,v), max(u,v))
                # Try to change the color of this edge
                old_color = color_map[key]
                
                # What colors are forbidden for this edge?
                bad = set()
                for x in adj_list[u]:
                    for y in adj_list[v]:
                        if x == v or y == u:
                            continue
                        if (min(x,y), max(x,y)) in edge_set:
                            c_vy = color_map[(min(v,y), max(v,y))]
                            c_yx = color_map[(min(x,y), max(x,y))]
                            c_xu = color_map[(min(x,u), max(x,u))]
                            for c in range(num_colors):
                                if c == c_vy or c == c_yx or c == c_xu:
                                    bad.add(c)
                
                # Find a different color that's not bad
                for new_c in range(num_colors):
                    if new_c != old_color and new_c not in bad:
                        color_map[key] = new_c
                        improved = True
                        break
            
            if not improved:
                break
        
        # Try to add more edges
        for u, v in all_edges:
            if time.time() > deadline - 2:
                break
            key = (min(u,v), max(u,v))
            if key in edge_set:
                continue
            c = try_add_edge(edge_set, color_map, u, v)
            if c is not None:
                edge_set.add(key)
                color_map[key] = c
                adj_list[u].add(v)
                adj_list[v].add(u)
        
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = [(min(k[0],k[1]), max(k[0],k[1]), color_map[k]) for k in edge_set]
    
    # Return the best result found
    if best_edges:
        return [[u, v, c] for u, v, c in best_edges]
    
    # Fallback: star
    return [[0, v, 0] for v in range(1, 20)]
