"""Construct small 4-regular graphs of high girth via algebraic and voltage constructions."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _verify(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g.
    Uses BFS from every vertex to detect short cycles."""
    # Check simple and degree
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Girth check via BFS from every vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (fallback)."""
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


def _generalized_hexagon_incidence():
    """Construct the point-line incidence graph of the generalized hexagon GQ(2,2)
    over GF(4). This gives a 4-regular bipartite graph with 224 vertices and
    girth 12 (which satisfies girth >= 10).
    
    The generalized quadrangle GQ(2,2) has (s+1)(s^2+1) = 3*5 = 15 points per line
    ... actually let me use the correct construction.
    
    For a 4-regular graph of girth 12, we need the incidence graph of a generalized
    hexagon of order (2,2). A generalized hexagon of order (s,t) has:
    - Each point on t+1 lines, each line on s+1 points
    - For (2,2): 3 points per line, 3 lines per point -> 3-regular, not 4-regular.
    
    Let me use a different approach: the point-line incidence graph of a generalized
    quadrangle GQ(3,1) or similar.
    
    Actually, for 4-regular girth 12, I need a bipartite graph where each vertex
    has degree 4 and no 4-cycles or 6-cycles or 8-cycles or 10-cycles.
    
    Let me use the Hoffman-Singleton type construction or a known 4-regular
    graph of girth 12.
    
    Actually, the simplest approach: use the LUW construction with d=7 which gives
    girth >= 12. For q=4, d=7:
    n = 2 * q^(d+1) / (q+1) ... let me just compute it.
    
    The LUW graph CD(d,q) has girth >= d+5. For d=7, girth >= 12.
    Number of vertices = 2 * (q^d + q^(d-1) + ... + 1) approximately.
    For q=4, d=7: 2*(4^7+4^6+...+1) = 2*(16384+4096+1024+256+64+16+4+1) = 2*21845 = 43690.
    That's way too many.
    
    Let me try a voltage graph approach instead. Take a 4-regular base graph of
    small size and lift it with a cyclic group to get girth 12.
    
    The smallest 4-regular graph is the complete graph K5 (5 vertices, degree 4).
    But K5 has triangles.
    
    The 4-regular graph on 6 vertices is K_{3,3} minus a perfect matching... no.
    The 4-regular graph on 6 vertices: each vertex connects to 4 others, so it's
    K6 minus a 1-regular graph (perfect matching). K6 has 15 edges, remove 3
    edges (perfect matching) to get 12 edges, 6*4/2 = 12. Yes.
    This graph has triangles (K6 minus matching still has triangles).
    
    For girth 12, I need a 4-regular base graph with no short cycles, then lift.
    
    Let me try a different structural approach: use the Clebsch graph or other
    known 4-regular graphs.
    
    Actually, let me just use the LUW construction which is guaranteed to work,
    and focus on making it efficient. The LUW graph for k=4, g=10 has d=5,
    and the number of vertices is manageable.
    """
    # For now, return None to indicate we should use LUW
    return None


def _voltage_lift_petersen(m, deadline):
    """Lift the Petersen graph with voltage group Z_m to get a 4-regular graph.
    
    The Petersen graph has 10 vertices and 15 edges. We assign voltages in Z_m
    to each edge such that the lift has girth >= 10.
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
    ]
    
    n = 10 * m
    best_edges = None
    
    # Try deterministic voltage assignments first
    # Known good voltage assignments for Petersen lift with girth >= 10:
    # We need all 5-cycles in Petersen to have non-zero voltage sum mod m.
    
    # Let's try a systematic search
    start_time = time.time()
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7919 + 13)
        
        # Assign random voltages
        voltages = [random.randint(1, m - 1) for _ in range(15)]
        
        # Build the lift
        edges = []
        for i, (u, v) in enumerate(petersen_edges):
            vol = voltages[i]
            for x in range(m):
                a = u * m + x
                b = v * m + ((x + vol) % m)
                edges.append((a, b))
        
        # Quick check: verify degree
        deg = [0] * n
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        if any(d != 4 for d in deg):
            continue
        
        # Verify girth
        if _verify(n, edges, 4, 10):
            best_edges = edges
            break
    
    if best_edges is not None:
        return n, best_edges
    
    return None


def _try_small_constructions(deadline):
    """Try to construct a 4-regular graph of girth 10 with fewer than 320 vertices."""
    # Try voltage lifts of small 4-regular base graphs
    
    # Petersen graph lift with various m values
    for m in [20, 24, 28, 30, 32]:
        if time.time() > deadline:
            break
        result = _voltage_lift_petersen(m, deadline)
        if result is not None:
            return result
    
    return None


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    if k == 4 and g == 10:
        # Try voltage graph lifts first (potentially smaller than LUW)
        result = _try_small_constructions(deadline)
        if result is not None:
            n, edges = result
            # Double-check
            if _verify(n, edges, k, g):
                return n, edges
    
    # Fallback: LUW graph
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
