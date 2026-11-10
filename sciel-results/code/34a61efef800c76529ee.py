import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 118

def get_groetzsch_edges():
    # Grötzsch graph edges (11 vertices, 20 edges)
    # Triangle-free, 4-chromatic
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 0), (5, 2),
        (6, 1), (6, 3),
        (7, 2), (7, 4),
        (8, 3), (8, 0),
        (9, 4), (9, 1),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]

def generate_compositions(n, k):
    """Generate all compositions of n into k positive integers."""
    if k == 1:
        yield (n,)
        return
    for i in range(1, n - k + 2):
        for rest in generate_compositions(n - i, k - 1):
            yield (i,) + rest

def calc_blowup_edges(sizes, edges):
    """Calculate number of edges in the blow-up."""
    count = 0
    for u, v in edges:
        count += sizes[u] * sizes[v]
    return count

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 118
    
    n = 16
    groetzsch_edges = get_groetzsch_edges()
    num_nodes = 11
    
    best_count = 0
    best_sizes = None
    
    # Enumerate all compositions of 16 into 11 parts
    for sizes in generate_compositions(n, num_nodes):
        count = calc_blowup_edges(sizes, groetzsch_edges)
        if count > best_count:
            best_count = count
            best_sizes = sizes
            
    if best_sizes is None:
        # Fallback to a known good distribution
        best_sizes = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3)
        best_count = calc_blowup_edges(best_sizes, groetzsch_edges)

    # Construct the graph
    edges = []
    vertex_map = []
    for i in range(num_nodes):
        for _ in range(best_sizes[i]):
            vertex_map.append(i)
            
    # Map original vertices to new vertices
    # For each edge in Grötzsch, add all edges between the corresponding sets
    for u_orig, v_orig in groetzsch_edges:
        u_set = [v for v in range(n) if vertex_map[v] == u_orig]
        v_set = [v for v in range(n) if vertex_map[v] == v_orig]
        for u in u_set:
            for v in v_set:
                edges.append([u, v])
                
    return edges
