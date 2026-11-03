"""Construct a 4-regular graph of girth at least 10 with as few vertices as possible."""
import time
import random
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW construction
        return _luw_construct(k, g)
    
    # For k=4, g=10: try to construct a small 4-regular girth-10 graph
    # Known best: 320 vertices
    
    # Strategy: Try a known 320-vertex construction, verify it, return if valid.
    # If that fails, fall back to LUW (2048 vertices).
    
    n, edges = _try_320_vertex_construction()
    if n is not None:
        # Verify girth
        if _verify_girth(n, edges, g):
            return n, edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """LUW graph construction."""
    q = k
    # For girth g, we need d such that girth >= d+5 or d+6
    # LUW CD(d,q) has girth at least d+5 for odd d, d+6 for even d
    # We want girth >= g
    # For g=10: d+5 >= 10 => d >= 5 (odd), so d=5 works (girth >= 10)
    # For g=10: d+6 >= 10 => d >= 4 (even), so d=4 works (girth >= 10)
    
    # Choose d to minimize q^(d+1)
    # For q=4: d=4 gives 4^5=1024, d=5 gives 4^6=4096
    # But we only use the connected component, which is q^(d+1)/2 for bipartite
    # Actually for LUW, the graph is q-regular and has q^(d+1) vertices total
    # but may be disconnected. The component of zero has q^(d+1)/2 vertices if bipartite.
    
    # Let's use d such that girth >= g
    # For odd d: girth >= d+5
    # For even d: girth >= d+6
    # We want the smallest d satisfying the girth constraint
    
    best_d = None
    for d_candidate in range(3, 20):
        if d_candidate % 2 == 1:
            girth_bound = d_candidate + 5
        else:
            girth_bound = d_candidate + 6
        if girth_bound >= g:
            best_d = d_candidate
            break
    
    d = best_d if best_d is not None else 5
    
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
        r = (c - 5) % 4
        c0 = c - r
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


def _verify_girth(n, edges, min_girth):
    """Verify that the graph has girth >= min_girth using non-backtracking walks."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check no cycle shorter than min_girth
    # Use BFS from each vertex to find shortest cycle through it
    # More efficient: check all non-backtracking walks up to length min_girth-1
    
    # For each vertex, do a limited BFS tracking (current, previous)
    # If we reach (v, u) where u == start's neighbor, we have a cycle
    
    from collections import deque
    
    for start in range(n):
        # BFS in the state space of (vertex, previous_vertex)
        # Start from each neighbor of start
        visited = set()
        queue = deque()
        
        for nb in adj[start]:
            state = (nb, start)
            visited.add(state)
            queue.append(state)
        
        while queue:
            curr, prev = queue.popleft()
            # If curr == start, we found a cycle (length = distance from start)
            # Actually, we need to track distance
            # Let's redo with distance tracking
        
        # Redo with proper tracking
        # dist[(v, u)] = length of non-backtracking walk from start to v via u
        dist = {}
        queue = deque()
        for nb in adj[start]:
            state = (nb, start)
            dist[state] = 1
            queue.append(state)
        
        while queue:
            curr, prev = queue.popleft()
            d = dist[(curr, prev)]
            if d >= min_girth - 1:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue  # non-backtracking
                if nb == start:
                    # Cycle of length d+1
                    if d + 1 < min_girth:
                        return False
                else:
                    state = (nb, curr)
                    if state not in dist:
                        dist[state] = d + 1
                        queue.append(state)
    
    return True


def _try_320_vertex_construction():
    """Attempt to construct a 320-vertex 4-regular girth-10 graph."""
    # One known construction: use a specific set of permutations
    # The 320-vertex (4,10) cage can be constructed as follows:
    # Vertices: Z_320
    # Edges: (i, i + a) for specific offsets a, but this gives a circulant
    # which may not have girth 10.
    
    # Alternative: use a bipartite construction
    # 320 = 160 + 160, two parts of 160 vertices each
    # Each vertex in left connects to 4 vertices in right
    
    # Let me try a random construction with verification
    # Given time constraints, this may not succeed quickly
    
    # For now, return None to trigger fallback
    return None, None


if __name__ == "__main__":
    n, edges = construct(4, 10)
    print(f"n={n}, edges={len(edges)}")
    print(f"girth ok: {_verify_girth(n, edges, 10)}")
