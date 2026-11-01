"""Construct small 4-regular graphs of girth 10 (cage problem n(4,10))."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _build_edges_from_adjacency(adj, n):
    """Build a simple edge list from an adjacency dict {vertex: set(neighbors)}."""
    edges = []
    for u in range(n):
        for v in adj.get(u, ()):
            if u < v:
                edges.append((u, v))
    return edges


def _verify_girth(adj, n, g):
    """
    Verify girth >= g using non-backtracking BFS from every vertex.
    Returns True if girth >= g, False otherwise.
    """
    # For each vertex, do BFS with non-backtracking constraint
    # A non-backtracking walk of length L from u to v means no edge is traversed 
    # immediately back. We track (current_vertex, previous_vertex) states.
    
    for start in range(n):
        # State: (current, prev) where prev is the vertex we came from (or -1 for start)
        # Distance is the length of the walk
        visited = {}  # (current, prev) -> distance
        queue = deque()
        # Start: from 'start' with no previous vertex
        queue.append((start, -1, 1))
        visited[(start, -1)] = 1
        
        while queue:
            curr, prev, dist = queue.popleft()
            
            if dist >= g:
                # We've reached the required girth, no shorter cycle exists from this start
                break
            
            for nb in adj.get(curr, ()):
                if nb == prev:
                    continue  # non-backtracking
                state = (nb, curr)
                if state in visited:
                    # Found a cycle of length dist + 1
                    if dist + 1 < g:
                        return False
                else:
                    visited[state] = dist + 1
                    queue.append((nb, curr, dist + 1))
    return True


def _construct_luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    This is a q-regular bipartite graph with girth at least d + 5 for odd d.
    Returns (n, edges) for the connected component of the zero point.
    """
    q = k
    # We want girth >= g. The LUW graph CD(d,q) has girth >= d+5 (for odd d).
    # So we need d + 5 >= g => d >= g - 5.
    # d must be odd for the standard construction to guarantee girth d+5.
    d = g - 5
    if d % 2 == 0:
        d += 1  # make it odd, so girth >= d+5 >= g+1 > g, still valid
    
    # Actually, let's be more careful. The standard LUW CD(d,q) for prime q
    # is a q-regular graph with girth at least d+5 when d is odd.
    # For d even, girth is at least d+4.
    # Let's pick the smallest odd d such that d + 5 >= g.
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Rule for LUW graphs: for c = 2, ..., d, rule[c] = (a, b)
    # such that point p ~ line l iff l[c] - p[c] = l[a] * p[b]
    # The specific rule depends on d.
    # For the standard LUW construction:
    # d=3: rule[2]=(1,1), rule[3]=(2,1)
    # d=5: rule[2]=(1,1), rule[3]=(1,2), rule[4]=(2,1), rule[5]=(3,1)
    # General pattern for odd d:
    # Let's use the known rule from the seed program.
    
    rule = {}
    # The seed program uses this rule construction:
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: p[c] = l[c] - l[a] * p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    adj = [set() for _ in range(q * (q ** d))]  # pre-allocate worst case
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            u = index[vertex]
            v = index[other]
            if len(adj) <= v:
                adj.extend([set() for _ in range(v + 1 - len(adj))])
            if len(adj) <= u:
                adj.extend([set() for _ in range(u + 1 - len(adj))])
            adj[u].add(v)
            adj[v].add(u)
    
    n = len(order)
    # Trim adj to size n
    adj = adj[:n]
    edges = _build_edges_from_adjacency(adj, n)
    return n, edges


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    deadline = time.time() + 110  # 10 second safety margin
    
    # Start with the LUW graph as a baseline
    try:
        n_base, edges_base = _construct_luw_graph(k, g)
        # Verify it
        adj_base = [set() for _ in range(n_base)]
        for u, v in edges_base:
            adj_base[u].add(v)
            adj_base[v].add(u)
        if _verify_girth(adj_base, n_base, g):
            best_n = n_base
            best_edges = edges_base
        else:
            best_n = None
            best_edges = None
    except Exception:
        best_n = None
        best_edges = None
    
    # For now, just return the LUW graph if it's valid
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return an empty or trivial result
    return 0, []

# EVOLVE-BLOCK-END
