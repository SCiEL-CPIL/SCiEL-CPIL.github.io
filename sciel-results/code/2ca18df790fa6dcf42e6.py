import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > deadline
    
    best_n = None
    best_edges = None
    
    # --- Method 1: Lazebnik-Ustimenko-Woldar with optimized parameters ---
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        if best_n is None or n_luw < best_n:
            best_n = n_luw
            best_edges = edges_luw
    except Exception:
        pass
    
    if check_deadline():
        if best_n is not None:
            return best_n, best_edges
        # Fallback if LUW fails (should not happen for valid k, g)
        return _luw_construct(k, g)
    
    # --- Method 2: Specific optimizations for small cases or known cages ---
    # For k=3, g=16, the known best is 936. We cannot easily construct it without hardcoded data.
    # However, we can check if a smaller construction exists via other means?
    # For now, LUW with minimal d is the best generic approach.
    
    # If we had time, we could try a BFS-based cage search, but it's risky.
    # We will stick with the optimized LUW.
    
    if best_n is None:
        best_n, best_edges = _luw_construct(k, g)
        
    return best_n, best_edges


def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Optimized to use the minimal d required to satisfy the girth constraint.
    """
    q = k
    # The girth of CD(d, q) is 2d + 2.
    # We need 2d + 2 >= g  =>  d >= (g - 2) / 2.
    # d must be an integer >= 3 for the standard construction to work well.
    d = max(3, (g - 2) // 2)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # For prime q, we use modular arithmetic
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the recurrence rule for the coordinates
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The pattern repeats every 4 steps: (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        # Note: The indices in the original LUW paper might vary, but this is a standard valid set
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # y[0] is implicitly 0 or handled by the side switch
            # The LUW graph is bipartite between side 0 and side 1.
            # Vertices are (side, tuple of d coords).
            # The "first" coordinate of the next side is the loop variable.
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

# EVOLVE-BLOCK-END
