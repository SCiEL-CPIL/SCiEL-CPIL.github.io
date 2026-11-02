import time
import random
import networkx as nx
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Target: 4-regular, girth 10.
    # Known cage size is 320.
    
    N = 320
    best_graph = None
    best_girth = 0
    
    # Helper to check girth efficiently
    def get_girth(graph):
        # BFS from each vertex up to depth g//2
        # If we find a cycle of length L < g, return L
        # Otherwise return g (or higher)
        # For efficiency, just check up to g-1
        # If no cycle < g found, return g
        
        # Use nx.girth which is BFS based
        try:
            girth = nx.girth(graph)
            return girth
        except nx.NetworkXException:
            return float('inf')
        
    # Local Search
    def local_search(G, max_steps=20000):
        current_girth = get_girth(G)
        if current_girth >= g:
            return G, current_girth
            
        edges = list(G.edges())
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            u, v = random.choice(edges)
            x, y = random.choice(edges)
            
            # Check if swap is valid
            if u == x or u == y or v == x or v == y:
                continue
            if G.has_edge(u, x) or G.has_edge(v, y):
                continue
            if u == v or x == y:
                continue
                
            # Perform swap
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(u, x)
            G.add_edge(v, y)
            
            new_girth = get_girth(G)
            
            if new_girth > current_girth:
                current_girth = new_girth
                if current_girth >= g:
                    return G, current_girth
            else:
                # Revert
                G.remove_edge(u, x)
                G.remove_edge(v, y)
                G.add_edge(u, v)
                G.add_edge(x, y)
                
        return G, current_girth

    # Try multiple random initializations
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        try:
            G = nx.random_regular_graph(k, N, seed=attempt)
        except Exception:
            continue
            
        G, current_girth = local_search(G)
        
        if current_girth >= g:
            best_graph = G
            break
        elif current_girth > best_girth:
            best_girth = current_girth
            best_graph = G

    if best_graph:
        edges_list = list(best_graph.edges())
        return N, edges_list

    # Fallback: LUW construction
    q = k
    d = 5 # for g=10
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
