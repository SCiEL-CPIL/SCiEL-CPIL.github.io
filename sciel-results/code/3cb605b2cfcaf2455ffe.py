import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    n = 20
    num_colors = 20
    
    best_edges = None
    best_count = 0
    
    # Precompute: for fast constraint checking
    # adj[v] = set of neighbors of v
    # color[(u,v)] = color of edge (u,v), use frozenset or sorted tuple
    
    def try_add_edge(edges_set, adj, color_map, u, v, c):
        """Check if adding edge (u,v) with color c is valid."""
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in edges_set:
            return False
        
        # Check all 4-cycles that would be formed by adding (u,v)
        # A 4-cycle containing (u,v): u-v-a-b-u where a~v, b~u, a~b, a!=b, a!=u, b!=v
        # Also u-v-b-a-u (same cycle, different direction)
        # So: for a in adj[v], b in adj[u], a != b, a != u, b != v, and a~b
        #   check colors of (u,v), (v,a), (a,b), (b,u) are all distinct
        
        for a in adj[v]:
            if a == u:
                continue
            for b in adj[u]:
                if b == v or b == a:
                    continue
                # Check if a~b
                ab_key = (min(a,b), max(a,b))
                if ab_key not in edges_set:
                    continue
                # 4-cycle: u-v-a-b-u
                # Edges: (u,v) with color c, (v,a), (a,b), (b,u)
                va_key = (min(v,a), max(v,a))
                bu_key = (min(b,u), max(b,u))
                c_va = color_map.get(va_key)
                c_ab = color_map.get(ab_key)
                c_bu = color_map.get(bu_key)
                if c_va is None or c_ab is None or c_bu is None:
                    continue
                if len({c, c_va, c_ab, c_bu}) != 4:
                    return False
        return True
    
    # Greedy randomized construction
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        edges_set = set()
        adj = [set() for _ in range(n)]
        color_map = {}
        
        # Random order of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() > deadline - 5:
                break
            # Try each color
            colors = list(range(num_colors))
            random.shuffle(colors)
            added = False
            for c in colors:
                if try_add_edge(edges_set, adj, color_map, u, v, c):
                    key = (u, v)
                    edges_set.add(key)
                    adj[u].add(v)
                    adj[v].add(u)
                    color_map[key] = c
                    added = True
                    break
        
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges = [[u, v, color_map[(u,v)]] for (u,v) in edges_set]
    
    if best_edges is None:
        # Fallback: star
        best_edges = [[0, v, 0] for v in range(1, n)]
    
    return best_edges
