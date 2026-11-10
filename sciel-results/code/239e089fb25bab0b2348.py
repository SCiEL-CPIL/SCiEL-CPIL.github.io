"""Search for a small 3-regular graph of girth >= g using voltage graph (covering) construction."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges)."""
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

def _girth_at_least_nb(adj, n, g, t0=None, deadline=140.0):
    """Return True iff graph (adjacency list) has girth >= g."""
    if g <= 2:
        return True
    
    max_depth = g - 1
    from collections import deque
    
    for s in range(n):
        visited = set()
        queue = deque()
        for u in adj[s]:
            state = (u, s)
            visited.add(state)
            queue.append((u, s, 1))
        
        while queue:
            v, p, depth = queue.popleft()
            
            if t0 is not None and (time.time() - t0) > deadline:
                return False
            
            if depth >= max_depth:
                continue
            
            nd = depth + 1
            for u in adj[v]:
                if u == p:
                    continue
                if u == s:
                    if nd < g:
                        return False
                    continue
                state = (u, v)
                if state not in visited:
                    visited.add(state)
                    queue.append((u, v, nd))
    return True

def _voltage_construct(k, g):
    """
    Construct a regular cover of a cubic base graph using voltage assignment.
    For k=3, g=16, we use a base graph of girth 8 (e.g., Petersen-like or specific cubic graph).
    We lift it over a group G of size M such that the lift has girth >= 16.
    """
    if k != 3:
        return None

    # Base graph: A cubic graph of girth 8.
    # The smallest cubic graph of girth 8 has 14 vertices? No, girth 6 is 10 (Petersen).
    # Girth 8: The (3,8)-cage is the McGee graph? No, McGee is (3,10) with 24 vertices.
    # (3,8)-cage has 14 vertices? Moore bound for (3,8) is 2*(1+2+4) = 14.
    # Does a (3,8)-cage exist? Yes, it is the 14-vertex graph.
    # Let's construct the (3,8)-cage.
    # Vertices 0..13.
    # It is the complement of the disjoint union of 7 K_2? No.
    # It is a well-known graph. Let's use a known construction or search for one.
    # Alternatively, we can use the Petersen graph (girth 5) and lift it?
    # If base girth is 5, and we want girth 16, the voltage must "spread out" cycles of length 5 to length >= 16.
    # A cycle of length L in the base lifts to cycles of length L * (order of voltage product).
    # If we use a base graph with girth 8, the shortest cycles are length 8.
    # We need the lift of any 8-cycle to have length >= 16.
    # This implies the voltage product of any 8-cycle must have order >= 2.
    # If we use a base graph with girth 10, shortest cycles are 10.
    # Lift must have length >= 16. Voltage product order must be >= 2 (since 10*1=10 < 16, 10*2=20 >= 16).
    
    # Let's try a base graph of girth 10: The (3,10)-cage (McGee graph, 24 vertices).
    # If we lift the McGee graph over a group of size M, the number of vertices is 24 * M.
    # We want the girth to be >= 16.
    # The shortest cycles in McGee are length 10.
    # If the voltage product of every 10-cycle has order >= 2, then the lifted cycles have length >= 20.
    # What about non-backtracking walks that close?
    # The girth of the lift is the minimum length of a closed non-backtracking walk in the lift.
    # This corresponds to a closed walk in the base where the voltage product is 0 (identity).
    # If all 10-cycles have voltage product of order 2, they don't close (product is not 0).
    # But maybe there are longer cycles in the base that close?
    # If we assign voltages randomly from a large enough group, the probability of a short cycle closing is low.
    
    # Let's use the McGee graph (3,10-cage).
    # We need to construct it.
    # McGee graph vertices: 0..23.
    # It is a well-known graph. I will hardcode its edges or construct it via a known method.
    # Actually, constructing the McGee graph from scratch is complex.
    # Let's try a simpler base: The Petersen graph (girth 5).
    # If we lift Petersen over a group of size M.
    # Shortest cycles in Petersen are 5.
    # We need lifted cycles to be >= 16.
    # So voltage product of any 5-cycle must have order >= 4 (5*3=15 < 16, 5*4=20 >= 16).
    # If we use a cyclic group of order 4, and assign voltages such that every 5-cycle has voltage sum = 1, 2, or 3 (mod 4), then the order is 2 or 4.
    # If order is 2, lifted length is 10 < 16. Bad.
    # So we need order >= 4. This means voltage sum must be 1 or 3 (odd).
    # Can we assign voltages in Z4 to Petersen edges such that every 5-cycle has odd sum?
    # Petersen has 12 five-cycles.
    # This is a constraint satisfaction problem.
    
    # Let's try a different approach: Use a base graph with girth 16?
    # That's what we are trying to find.
    
    # Let's go back to the voltage idea with a base graph of girth 8.
    # (3,8)-cage has 14 vertices.
    # If we lift it over a group of size M.
    # Shortest cycles in base are 8.
    # We need lifted cycles >= 16.
    # So voltage product of any 8-cycle must have order >= 2.
    # If we use Z2, and assign voltages such that every 8-cycle has sum 1 (mod 2), then the order is 2.
    # Lifted length = 8 * 2 = 16.
    # This would give a graph of size 14 * 2 = 28 vertices?
    # Wait, if the voltage product is 1 (mod 2), the walk does not close.
    # The lift of an 8-cycle becomes two 8-cycles? No.
    # If the voltage sum is 1, the lift of the cycle is a single cycle of length 16?
    # Let's check: A cycle of length L in the base with voltage sum v.
    # In the lift, starting at (v0, 0), after L steps we are at (v0, v).
    # To close, we need v = 0.
    # If v = 1, we don't close. We continue.
    # After another L steps, we are at (v0, 2v) = (v0, 0).
    # So the lifted cycle has length 2L = 16.
    # So if we can assign voltages in Z2 to the (3,8)-cage such that every 8-cycle has odd sum,
    # then the lift will have girth >= 16 (since 8-cycles become 16-cycles, and longer cycles become longer).
    # The size of the lift is 14 * 2 = 28 vertices.
    # But wait, is the (3,8)-cage bipartite?
    # If it is bipartite, all cycles are even.
    # Can we 2-color the edges such that every 8-cycle has odd sum?
    # This is equivalent to finding a cut? No.
    # Let's try to construct the (3,8)-cage and solve this CSP.
    
    # (3,8)-cage edges:
    # It is the graph with 14 vertices.
    # Let's list them.
    # Vertices: 0..13.
    # Edges:
    # 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-10, 10-11, 11-12, 12-13, 13-0 (cycle 14)
    # Plus chords to make it cubic.
    # 0-7, 1-8, 2-9, 3-10, 4-11, 5-12, 6-13?
    # Let's check girth.
    # This specific construction might not have girth 8.
    # The (3,8)-cage is unique.
    # Let's use a known adjacency list for the (3,8)-cage.
    # From literature:
    # 0: 1, 6, 11
    # 1: 0, 2, 12
    # 2: 1, 3, 13
    # 3: 2, 4, 0  (Wait, 3-0? 0 is connected to 1,6,11. So 3-0 is not in the list above. Let's re-verify.)
    
    # Let's use a reliable source or construct it via search.
    # Since we have time, we can search for a (3,8)-cage if we don't have it hardcoded.
    # But hardcoding is safer.
    
    # Here is the (3,8)-cage (14 vertices):
    edges_base = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 0),
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12), (7, 13)
    ]
    # Let's verify degrees.
    # 0: 1, 13, 6 (deg 3)
    # 1: 0, 2, 7 (deg 3)
    # 2: 1, 3, 8 (deg 3)
    # 3: 2, 4, 9 (deg 3)
    # 4: 3, 5, 10 (deg 3)
    # 5: 4, 6, 11 (deg 3)
    # 6: 5, 7, 0, 12 (deg 4) -> Oops.
    
    # Let's try another set.
    # The (3,8)-cage is the graph obtained from the 14-cycle by adding chords (0,7), (1,8), (2,9), (3,10), (4,11), (5,12), (6,13).
    # Let's check degrees:
    # 0: 1, 13, 7 (deg 3)
    # 1: 0, 2, 8 (deg 3)
    # 2: 1, 3, 9 (deg 3)
    # 3: 2, 4, 10 (deg 3)
    # 4: 3, 5, 11 (deg 3)
    # 5: 4, 6, 12 (deg 3)
    # 6: 5, 7, 13 (deg 3)
    # 7: 6, 8, 0 (deg 3)
    # 8: 7, 9, 1 (deg 3)
    # 9: 8, 10, 2 (deg 3)
    # 10: 9, 11, 3 (deg 3)
    # 11: 10, 12, 4 (deg 3)
    # 12: 11, 13, 5 (deg 3)
    # 13: 12, 0, 6 (deg 3)
    # This looks cubic.
    # Let's check girth.
    # Cycles:
    # 0-1-2-3-4-5-6-7-0 (length 8)
    # 0-7-8-9-10-11-12-13-0 (length 8)
    # 0-1-8-7-0 (length 4) -> Girth 4!
    # So this is not the (3,8)-cage.
    
    # The (3,8)-cage is actually the 14-vertex graph with no cycles shorter than 8.
    # It is not the one with chords (i, i+7).
    
    # Let's use the fact that the (3,8)-cage is the complement of the perfect matching in K_14? No.
    # It is a well-known graph. I will use a hardcoded adjacency list from a reliable source.
    # Since I don't have it, I will fall back to LUW if I can't construct it quickly.
    
    # However, the voltage idea is promising. Let's try to use the Petersen graph (girth 5) and lift it over Z4.
    # We need to assign voltages in Z4 to Petersen edges such that every 5-cycle has voltage sum 1 or 3 (odd).
    # Petersen has 12 five-cycles.
    # Let's try to solve this CSP.
    
    # Petersen edges:
    # Outer: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    
    # Let's try a random assignment and check.
    rng = random.Random(42)
    for _ in range(1000):
        voltages = {}
        for u, v in Petersen_edges:
            voltages[(u, v)] = rng.randint(0, 3)
            voltages[(v, u)] = (4 - voltages[(u, v)]) % 4
        
        # Check all 5-cycles
        ok = True
        for cycle in Petersen_5cycles:
            s = 0
            for i in range(5):
                u, v = cycle[i], cycle[(i+1)%5]
                s = (s + voltages[(u, v)]) % 4
            if s % 2 == 0:
                ok = False
                break
        if ok:
            # Construct the lift
            n = 10 * 4
            adj = [set() for _ in range(n)]
            for u, v in Petersen_edges:
                vu = voltages[(u, v)]
                for x in range(4):
                    u_lift = u * 4 + x
                    v_lift = v * 4 + (x + vu) % 4
                    adj[u_lift].add(v_lift)
                    adj[v_lift].add(u_lift)
            
            # Verify girth
            adj_list = [sorted(x) for x in adj]
            if _girth_at_least_nb(adj_list, n, 16):
                edges = []
                for i in range(n):
                    for j in adj_list[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    return None

# Define Petersen edges and cycles for the voltage construction
Petersen_edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
]

# Generate all 5-cycles in Petersen
Petersen_5cycles = []
def _find_cycles(adj, start, depth, path, visited):
    if depth == 5 and path[-1] == start:
        Petersen_5cycles.append(list(path))
        return
    if depth > 5:
        return
    for u in adj[path[-1]]:
        if u == start and depth == 5:
            _find_cycles(adj, start, depth + 1, path + [u], visited)
        elif u not in visited:
            visited.add(u)
            _find_cycles(adj, start, depth + 1, path + [u], visited)
            visited.remove(u)

_p_adj = [set() for _ in range(10)]
for u, v in Petersen_edges:
    _p_adj[u].add(v)
    _p_adj[v].add(u)
for i in range(10):
    _find_cycles(_p_adj, i, 1, [i], {i})

def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 140.0

    if k == 3 and g == 16:
        result = _voltage_construct(k, g)
        if result:
            return result

    # Fallback: LUW
    try:
        return _luw_construct(k, g)
    except Exception:
        raise

# EVOLVE-BLOCK-END
