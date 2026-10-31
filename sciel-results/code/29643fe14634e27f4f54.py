<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        # State: (vertex, prev_vertex), track min distance
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False  # found cycle shorter than g
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _build_lift(m, seed=0):
    """Build a voltage graph lift of the Petersen graph over Z_m.
    Returns (n, adj, edges) or None if invalid."""
    rng = _random.Random(seed)
    
    # Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5
    # Vertices 0-4 form outer 5-cycle, 5-9 form inner star
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    # Assign random voltages to each edge
    voltages = {}
    for u, v in petersen_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    for u, v in petersen_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    # Check no multi-edges (should be fine with sets, but verify)
    for i in range(n):
        if len(adj[i]) != len(adj[i]):
            return None
    
    return n, adj, edges
=======
"""Combined program: algebraic voltage lifts + fast verification + LUW fallback."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _quadratic_residues(m):
    """Return sorted list of distinct quadratic residues modulo m."""
    if m <= 1:
        return [0]
    residues = set()
    for x in range(m):
        residues.add((x * x) % m)
    return sorted(residues)

def _build_lift_petersen(m, variant=0):
    """Build a voltage graph lift of the Petersen graph over Z_m using
    deterministic quadratic-residue-based voltage assignment.
    Returns (n, adj, edges) or None if invalid."""
    
    # Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    # Get quadratic residues for voltage assignment
    qres = _quadratic_residues(m)
    nr = len(qres)
    
    # Assign voltages based on quadratic residues with variant offset
    voltages = {}
    for idx, (u, v) in enumerate(petersen_edges):
        # Use different residue indices based on variant to explore different assignments
        base_idx = (idx * 7 + variant * 3) % nr
        vol = qres[base_idx]
        voltages[(u, v)] = vol
        voltages[(v, u)] = (-vol) % m
    
    for u, v in petersen_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, adj, edges

def _luw_construct(k, g):
    """LUW (Lubotzky-Phillips-Sarnak style) construction fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
>>>>>>> REPLACE
