import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = n_vertices * d // 2  # 448
    
    # Precompute edges
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = idx
                idx += 1
    
    # Precompute squares (4-cycles)
    # A square is defined by two distinct dimensions i, j and a vertex u where bits i,j are 0 (or just any base).
    # Actually, simpler: iterate over all pairs of dimensions (i, j) and all vertices.
    # For a fixed pair (i, j), the squares are formed by vertices u where we toggle i and j.
    # To avoid duplicates, we can iterate u from 0 to 127.
    # A square is {u, u^1<<i, u^1<<j, u^1<<i^1<<j}.
    # The edges are (u, u^1<<i), (u^1<<i, u^1<<i^1<<j), (u^1<<i^1<<j, u^1<<j), (u^1<<j, u).
    # We need to ensure we don't count the same square 4 times (once for each vertex) or 2 times.
    # Let's just generate all 4-cycles by picking two dimensions i < j.
    # There are C(7,2) = 21 pairs.
    # For each pair, there are 2^(7-2) = 32 squares.
    # Total 672 squares.
    
    squares = []
    edge_squares = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all vertices, but only consider those where bit i is 0?
            # Actually, any vertex u defines a square with u, u^i, u^j, u^i^j.
            # To get unique squares, we can fix bit i to 0.
            # If bit i of u is 0, then u and u^i are distinct.
            # Let's iterate u from 0 to 127.
            # If (u & (1 << i)) == 0:
            #   The square vertices are u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j).
            #   The edges are:
            #   e1 = (min(u, u^(1<<i)), max(u, u^(1<<i)))
            #   e2 = (min(u^(1<<j), u^(1<<i)^(1<<j)), max(u^(1<<j), u^(1<<i)^(1<<j)))
            #   e3 = (min(u, u^(1<<j)), max(u, u^(1<<j)))
            #   e4 = (min(u^(1<<i), u^(1<<i)^(1<<j)), max(u^(1<<i), u^(1<<i)^(1<<j)))
            
            for u in range(n_vertices):
                if (u & (1 << i)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v3, v4), max(v3, v4))
                    e3 = (min(v1, v3), max(v1, v3))
                    e4 = (min(v2, v4), max(v2, v4))
                    
                    sq_edges = [edge_map[e1], edge_map[e2], edge_map[e3], edge_map[e4]]
                    squares.append(sq_edges)
                    
                    for e in sq_edges:
                        edge_squares[e].append(len(squares) - 1)
                        
    n_squares = len(squares)
    
    # State
    selected = np.zeros(n_edges, dtype=bool)
    square_counts = np.zeros(n_squares, dtype=np.int8)
    
    def can_add(e):
        for s in edge_squares[e]:
            if square_counts[s] >= 3:
                return False
        return True
        
    def add(e):
        selected[e] = True
        for s in edge_squares[e]:
            square_counts[s] += 1
            
    def remove(e):
        selected[e] = False
        for s in edge_squares[e]:
            square_counts[s] -= 1
            
    # Greedy Initialization
    edge_indices = list(range(n_edges))
    random.shuffle(edge_indices)
    
    current_score = 0
    for e in edge_indices:
        if can_add(e):
            add(e)
            current_score += 1
            
    best_score = current_score
    best_state = selected.copy()
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Local Search
    # Strategy:
    # 1. Try to add any valid edge.
    # 2. If stuck, remove a random edge and try again.
    
    # To speed up "try to add any valid edge", we can maintain a list of candidate edges?
    # Or just iterate through all edges in random order.
    # 448 edges is small. Iterating through all is fast.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle to avoid bias in which edge is found first
        random.shuffle(edge_indices)
        
        for e in edge_indices:
            if not selected[e] and can_add(e):
                add(e)
                current_score += 1
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_state = selected.copy()
                    
        if not improved:
            # Local maximum reached.
            # Remove a random edge to escape.
            # Pick a random selected edge
            selected_edges = np.where(selected)[0]
            if len(selected_edges) > 0:
                e_to_remove = selected_edges[random.randint(0, len(selected_edges) - 1)]
                remove(e_to_remove)
                current_score -= 1
                
    # Return the best state found
    out = []
    for e in range(n_edges):
        if best_state[e]:
            out.append(list(edges[e]))
            
    return out
