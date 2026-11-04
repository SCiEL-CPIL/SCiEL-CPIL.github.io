import random
import networkx as nx
from collections import deque

def construct(k, g):
    # For k=3, g=16, try to construct the 936-vertex graph via voltage assignment
    # on a base graph of 72 vertices with voltage group Z_13.
    if k != 3 or g != 16:
        # Fallback to LUW for other parameters
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # Specific case for k=3, g=16
    # Target n=936. Base graph n_base=72, group size 13.
    n_base = 72
    group_size = 13
    
    # Generate a random 3-regular graph on n_base vertices
    # Use a simple random regular graph generator
    while True:
        try:
            base_graph = nx.random_regular_graph(3, n_base, seed=42)
            break
        except Exception:
            continue
            
    # Check girth of base graph
    girth_base = nx.girth(base_graph)
    if girth_base < 8:
        # Try a different seed or structure
        # For now, proceed, but we need the lift to have girth 16.
        pass

    edges_base = list(base_graph.edges())
    
    # Assign random voltages in Z_13 to edges
    # We need to ensure that the product of voltages around any cycle of length < 16 is not 0.
    # This is hard to guarantee directly.
    # Instead, we can use a known good voltage assignment if we had it.
    # Since we don't, we will try a few random assignments.
    
    best_lift = None
    best_girth = 0
    best_n = float('inf')
    
    for attempt in range(100): # Limit attempts
        # Random voltages
        voltages = {}
        for u, v in edges_base:
            voltages[(u, v)] = random.randint(0, group_size - 1)
            voltages[(v, u)] = (group_size - voltages[(u, v)]) % group_size
            
        # Construct the lift
        # Vertices are (base_vertex, group_element)
        n_lift = n_base * group_size
        lift_graph = nx.Graph()
        lift_graph.add_nodes_from([(u, g) for u in range(n_base) for g in range(group_size)])
        
        for u, v in edges_base:
            vol = voltages[(u, v)]
            for g in range(group_size):
                lift_graph.add_edge((u, g), (v, (g + vol) % group_size))
                
        # Check girth of lift
        girth_lift = nx.girth(lift_graph)
        if girth_lift >= g:
            # We found a valid graph
            edges_lift = list(lift_graph.edges())
            return n_lift, edges_lift
        elif girth_lift > best_girth:
            best_girth = girth_lift
            best_lift = lift_graph

    # If no valid graph found with girth >= 16, return the best one found
    # This might be invalid if girth < 16, but it's better than LUW if n is small.
    # However, the task requires girth >= g.
    # If we can't find one, we should fall back to LUW to ensure validity.
    # But LUW is huge.
    # Let's try a few more seeds for the base graph.
    
    for seed in range(100, 200):
        base_graph = nx.random_regular_graph(3, n_base, seed=seed)
        edges_base = list(base_graph.edges())
        
        for attempt in range(10):
            voltages = {}
            for u, v in edges_base:
                voltages[(u, v)] = random.randint(0, group_size - 1)
                voltages[(v, u)] = (group_size - voltages[(u, v)]) % group_size
                
            n_lift = n_base * group_size
            lift_graph = nx.Graph()
            lift_graph.add_nodes_from([(u, g) for u in range(n_base) for g in range(group_size)])
            
            for u, v in edges_base:
                vol = voltages[(u, v)]
                for g in range(group_size):
                    lift_graph.add_edge((u, g), (v, (g + vol) % group_size))
                    
            girth_lift = nx.girth(lift_graph)
            if girth_lift >= g:
                edges_lift = list(lift_graph.edges())
                return n_lift, edges_lift

    # Fallback to LUW if random search fails
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
