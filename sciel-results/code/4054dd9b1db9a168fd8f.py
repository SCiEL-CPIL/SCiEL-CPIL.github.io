# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g):
    """
    Construct a LUW (Lubotzky-Phillips-Sarnak style) graph.
    For k=4, g=10, this should produce a valid graph.
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        # GF(4) arithmetic using XOR-based representation
        # Elements: 0, 1, 2, 3 where 2 is the primitive element
        # Addition is XOR
        # Multiplication table for GF(4)
        mul_table = [
            [0, 0, 0, 0],  # 0 * x = 0
            [0, 1, 2, 3],  # 1 * x = x
            [0, 2, 3, 1],  # 2 * x
            [0, 3, 1, 2],  # 3 * x
        ]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y  # In char 2, sub = add
        mul = lambda x, y: mul_table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    # Define the LUW rule
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        """Generate all neighbors of a vertex in the LUW graph."""
        side, x = vertex
        # x is a tuple of length d+1: x[0] is None (unused), x[1..d] are coordinates
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

    # Build the graph by BFS from start vertex
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from side 0 to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _try_alternative_luw(k, g):
    """
    Try alternative LUW parameterizations to potentially find smaller graphs.
    For k=4, g=10, we try different rule sets.
    """
    if k != 4 or g != 10:
        return None
        
    # Try different rule configurations
    best = None
    
    # Original rule
    rules_list = [
        {2: (1,1), 3: (2,1), 4: (1,2), 5: (1,3)},  # original
        {2: (1,1), 3: (2,1), 4: (1,2), 5: (3,1)},  # alternative
        {2: (1,1), 3: (2,1), 4: (1,2), 5: (1,1)},  # alternative
        {2: (1,1), 3: (2,1), 4: (1,2), 5: (4,1)},  # alternative
    ]
    
    for rule in rules_list:
        q = 4
        d = 5
        
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
        ]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: mul_table[x][y]
        
        def neighbours(vertex, rule=rule):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y))
        
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
        
        n = len(order)
        # Validate
        if _validate_graph(n, edges, k, g):
            if best is None or n < best[0]:
                best = (n, edges)
    
    return best

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # Primary: LUW construction
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Validate the LUW graph
    if not _validate_graph(best_n, best_edges, k, g):
        # LUW failed, try alternatives
        alt = _try_alternative_luw(k, g)
        if alt is not None:
            best_n, best_edges = alt
        else:
            # Last resort: try a known small graph or return empty
            # This should not happen for k=4, g=10
            return 0, []
    
    # Try to find smaller graph using alternative LUW parameterizations
    alt = _try_alternative_luw(k, g)
    if alt is not None and alt[0] < best_n:
        best_n, best_edges = alt
    
    # Check time
    if time.time() > deadline:
        return best_n, best_edges
    
    # For now, return the best found
    return best_n, best_edges
# EVOLVE-BLOCK-END
