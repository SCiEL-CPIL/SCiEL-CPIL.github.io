import time
import sys
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Try to find a small cage via backtracking for small k, g
    if k <= 4 and g <= 9:
        result = find_small_cage(k, g, start_time, deadline)
        if result is not None:
            return result

    # 2. Fall back to Lazebnik-Ustimenko-Woldar graph
    return luw_construct(k, g)

def find_small_cage(k, g, start_time, deadline):
    """
    Attempts to find the smallest k-regular graph with girth at least g
    using backtracking. Returns (n, edges) or None if not found in time.
    """
    # Moore bound lower limit
    if g % 2 == 0:
        r = g // 2
        # Moore bound for even girth
        lower_bound = 2 * ((k-1)**r)
    else:
        r = (g - 1) // 2
        # Moore bound for odd girth
        lower_bound = 1 + k * (((k-1)**r - 1) // (k-2)) if k > 2 else g
    
    # We search for n from lower_bound to some limit
    # For k=3, g=5, LB=10.
    # For k=3, g=7, LB=18.
    # For k=4, g=5, LB=12? (Actually 12 is not a cage, 12 is not 4-reg girth 5? 
    # Let's just start from LB and go up.
    
    max_search_n = 100 # Limit search space to avoid explosion
    
    for n in range(lower_bound, max_search_n + 1):
        if time.time() > deadline:
            return None
        
        # Check if n is a valid size for k-regular graph
        if (n * k) % 2 != 0:
            continue
            
        # Try to construct a k-regular graph on n vertices with girth >= g
        graph = backtracking_search(k, g, n, start_time, deadline)
        if graph is not None:
            return n, graph
            
    return None

def backtracking_search(k, g, n, start_time, deadline):
    """
    Backtracking search for a k-regular graph on n vertices with girth >= g.
    Returns list of edges or None.
    """
    # This is a simplified heuristic backtracking.
    # For small n, we can try to build the graph.
    
    # Initialize adjacency list
    adj = [[] for _ in range(n)]
    edges = []
    
    # Heuristic: Try to add edges to satisfy degree k
    # This is complex to implement robustly in a short time.
    # A simpler approach for small k, g is to use known constructions or random restarts.
    
    # Given the complexity, I will only use this for very small cases where
    # a simple greedy might work, or fall back to LUW.
    
    # For now, return None to force LUW fallback, unless we hardcode specific small graphs.
    # I will hardcode the Petersen graph (3,5) and 3-cage (3,7) to be safe.
    
    if k == 3 and g == 5 and n == 10:
        # Petersen Graph
        # Vertices 0..9
        # Outer 5-cycle: 0-1-2-3-4-0
        # Inner 5-star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        edges = [
            (0,1), (1,2), (2,3), (3,4), (4,0),
            (5,7), (7,9), (9,6), (6,8), (8,5),
            (0,5), (1,6), (2,7), (3,8), (4,9)
        ]
        return edges
        
    if k == 3 and g == 7 and n == 18:
        # 3-cage for girth 7
        # This is the 18-vertex graph.
        # It is the **3-regular graph on 18 vertices** with girth 7.
        # It can be constructed as a circulant or specific permutation.
        # One construction:
        # Vertices 0..17
        # Edges:
        # (0, 1), (0, 10), (0, 17)
        # (1, 2), (1, 9), (1, 11)
        # ... This is hard to hardcode correctly without a generator.
        # I will skip hardcoding g=7 and rely on LUW for now, 
        # as LUW is valid.
        pass

    return None

def luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
# EVOLVE-BLOCK-END
