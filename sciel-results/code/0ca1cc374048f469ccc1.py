import time
import random
from itertools import combinations

_START_TIME = time.time()
DEADLINE = 120.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _make_bipartite_base(n=20, seed=0):
    """
    Construct K_{10,10} with a proper 4-cycle rainbow coloring.
    U = 0..9, V = 10..19.
    Color of edge (u, v) where v' = v - 10:
    c = (u + 2 * v') % 11
    This ensures every 4-cycle is rainbow.
    """
    edges = []
    for u in range(10):
        for v in range(10, 20):
            v_prime = v - 10
            c = (u + 2 * v_prime) % 11
            edges.append([u, v, c])
    return edges

def _check_validity(edges, n=20):
    """
    Check if the given edge list is a valid simple graph with rainbow 4-cycles.
    Returns (is_valid, num_edges)
    """
    # Check for duplicates and self-loops
    edge_set = set()
    for e in edges:
        u, v, c = e
        if u == v:
            return False, 0
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False, 0
        edge_set.add((u, v))
    
    # Build adjacency and color maps
    adj = {i: [] for i in range(n)}
    color = {}
    for u, v, c in edges:
        adj[u].append(v)
        adj[v].append(u)
        color[(min(u,v), max(u,v))] = c

    # Check all 4-cycles
    # A 4-cycle is u-v-w-x-u. 
    # We can iterate over all pairs of edges (u,v) and (w,x) that are disjoint and see if they form a C4.
    # Or iterate over u, v, w, x.
    # For n=20, number of 4-cycles is small enough.
    # Let's iterate over all subsets of 4 vertices? No, just find cycles.
    
    # Efficient check:
    # For each pair of vertices (u, v) that are connected, and for each pair (w, x) that are connected...
    # Actually, let's just iterate over all 4-tuples (u, v, w, x) with u < v, w < x, and check if they form a C4.
    # A C4 is u-v-w-x-u.
    # Edges: (u,v), (v,w), (w,x), (x,u).
    # We need u, v, w, x distinct.
    
    vertices = list(range(n))
    for u in vertices:
        for v in adj[u]:
            if v <= u: continue
            for w in adj[v]:
                if w == u: continue
                for x in adj[w]:
                    if x == u or x == v: continue
                    # Check if x is connected to u
                    if u in adj[x]:
                        # Check if it's a simple cycle u-v-w-x-u
                        # We need to ensure u, v, w, x are distinct.
                        # u != v (checked)
                        # v != w (checked)
                        # w != x (checked)
                        # x != u (checked)
                        # Also need to ensure we don't count the same cycle 8 times.
                        # We can enforce u < min(v, x) and v < w? No, that's tricky.
                        # Just check colors for now.
                        c1 = color.get((min(u,v), max(u,v)))
                        c2 = color.get((min(v,w), max(v,w)))
                        c3 = color.get((min(w,x), max(w,x)))
                        c4 = color.get((min(x,u), max(x,u)))
                        
                        if c1 is None or c2 is None or c3 is None or c4 is None:
                            return False, 0
                        
                        if len({c1, c2, c3, c4}) != 4:
                            return False, 0
    return True, len(edges)

def _greedy_add_edges(base_edges, n=20, time_limit=10.0):
    """
    Try to add more edges to the base graph.
    """
    # Convert base_edges to sets for fast lookup
    adj = {i: set() for i in range(n)}
    color = {}
    for u, v, c in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        color[(min(u,v), max(u,v))] = c
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    current_edges = base_edges[:]
    
    start_time = time.time()
    
    for u, v in all_edges:
        if (u, v) in color:
            continue
        
        if time.time() - start_time > time_limit:
            break
        
        # Check if adding (u, v) with some color creates a non-rainbow C4
        # We need to check all C4s that include (u, v).
        # A C4 including (u, v) must be u-v-x-y-u or u-v-y-x-u.
        # So x is a neighbor of v, y is a neighbor of x, and y is a neighbor of u.
        # Or x is neighbor of u, y is neighbor of x, and y is neighbor of v.
        
        valid_colors = set(range(20))
        
        # Find all C4s that would be created by adding (u, v)
        # Case 1: u-v-x-y-u
        # x in N(v), y in N(x), y in N(u)
        for x in adj[v]:
            if x == u: continue
            for y in adj[x]:
                if y == u or y == v: continue
                if y in adj[u]:
                    # C4: u-v-x-y-u
                    # Colors of existing edges:
                    c_vx = color.get((min(v,x), max(v,x)))
                    c_xy = color.get((min(x,y), max(x,y)))
                    c_yu = color.get((min(y,u), max(y,u)))
                    
                    if c_vx is None or c_xy is None or c_yu is None:
                        continue
                    
                    # The new edge (u,v) must have a color distinct from c_vx, c_xy, c_yu
                    # And the C4 must be rainbow.
                    # So c_uv != c_vx, c_uv != c_xy, c_uv != c_yu.
                    valid_colors -= {c_vx, c_xy, c_yu}
        
        # Case 2: u-v-y-x-u (same as above, just swapping x and y roles? No, the cycle is undirected)
        # The set of C4s containing (u,v) is exactly the set of paths of length 3 from u to v.
        # We checked paths v-x-y-u. This covers all.
        
        if not valid_colors:
            continue
        
        # Pick a color
        c = min(valid_colors)
        
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        color[(min(u,v), max(u,v))] = c
        current_edges.append([u, v, c])
        
    return current_edges

def construct():
    n = 20
    
    # Start with the bipartite base
    base = _make_bipartite_base(n)
    
    # Try to improve with greedy addition
    # We can try multiple random seeds for the greedy addition
    best_edges = base
    best_count = len(base)
    
    start_time = time.time()
    
    for seed in range(100):
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        # Make a copy of base to modify
        # The greedy function expects a list of [u, v, c]
        candidate = _greedy_add_edges(base, n, time_limit=5.0)
        
        if len(candidate) > best_count:
            best_count = len(candidate)
            best_edges = candidate
    
    return best_edges
