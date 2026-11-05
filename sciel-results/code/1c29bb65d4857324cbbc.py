import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if k == 4 and g == 10:
        # Try to find a small 4-regular girth-10 graph
        # First: try known constructions and targeted searches
        
        # Strategy 1: Try specific known small constructions
        # The (4,10)-cage has 320 vertices. Let's try to construct it.
        
        # Try a family of 4-regular graphs based on Z_n x Z_m or similar
        # Actually, let's try a more systematic approach:
        # Search over "blow-up" of smaller cages or known algebraic constructions
        
        # For now, use the LUW construction as baseline but try to find smaller
        # by searching over a parameterized family
        
        # Try circulants with more structure (not just 2-step)
        # A 4-regular circulant needs 2 generating pairs: ±a, ±b
        # We already search these. Let's expand the search range.
        
        # Actually, let's try a different family: graphs from Z_p x Z_q
        # or use the fact that n(4,10) = 320 = 2^6 * 5
        
        # Try constructing via known methods
        # The (4,10)-cage is known to exist on 320 vertices.
        # Let's try a targeted construction.
        
        # Approach: Use a 4-regular graph on 320 vertices with specific structure
        # One known construction uses the group (Z_5)^4 or similar
        
        # Let's try: vertices = Z_5 x Z_5 x Z_4 x Z_4 (5*5*4*4 = 400, too many)
        # Or Z_5 x Z_5 x Z_8 (5*5*8 = 200, not 320)
        # 320 = 2^6 * 5 = 64 * 5
        
        # Try Z_5 x Z_64? That's 320.
        # Or Z_32 x Z_10 = 320
        
        # Let me try a different approach: use a known explicit edge list
        # for the (4,10)-cage if available, otherwise fall back to LUW.
        
        # For now, let's improve the circulant search to be faster and wider,
        # and also try a "twisted circulant" family.
        
        # Improved circulant search with early termination
        n = 320
        best_result = None
        
        # Try all pairs (a, b) with 1 <= a < b <= n//2
        # But use a faster girth check
        found = False
        for a in range(1, 20):
            if time.time() > deadline - 5:
                break
            for b in range(a + 1, n // 2 + 1):
                if time.time() > deadline - 5:
                    break
                if _check_circulant_girth(n, a, b, 10):
                    edges = _get_circulant_edges(n, a, b)
                    return n, edges
                    found = True
        
        if not found:
            # Fall back to LUW construction
            return _luw_construct(k, g)
    
    return _luw_construct(k, g)


def _check_circulant_girth(n, a, b, g):
    """Check if circulant graph C(n, a, b) has girth >= g."""
    # Non-backtracking BFS from vertex 0
    # State: (vertex, last_edge_type) where last_edge_type in {a_plus, a_minus, b_plus, b_minus}
    # Edge types: 0=a+, 1=a-, 2=b+, 3=b-
    # Backtracking: can't use a+ after a- or a- after a+, etc.
    
    max_len = g - 1
    # dist[state] = shortest length to reach this state
    dist = {}
    queue = deque()
    
    # Start: from vertex 0, take each of the 4 edges
    for et in range(4):
        if et == 0:
            nv = a % n
        elif et == 1:
            nv = (-a) % n
        elif et == 2:
            nv = b % n
        else:
            nv = (-b) % n
        state = (nv, et)
        dist[state] = 1
        queue.append((nv, et, 1))
    
    while queue:
        v, last_et, length = queue.popleft()
        
        if length >= max_len:
            continue
        
        for et in range(4):
            # Backtracking check: can't reverse the same edge
            if (last_et == 0 and et == 1) or (last_et == 1 and et == 0):
                continue
            if (last_et == 2 and et == 3) or (last_et == 3 and et == 2):
                continue
            
            if et == 0:
                nv = (v + a) % n
            elif et == 1:
                nv = (v - a) % n
            elif et == 2:
                nv = (v + b) % n
            else:
                nv = (v - b) % n
            
            new_length = length + 1
            
            if nv == 0 and new_length < g:
                return False
            
            state = (nv, et)
            if state not in dist or dist[state] > new_length:
                dist[state] = new_length
                if new_length < max_len:
                    queue.append((nv, et, new_length))
    
    return True


def _get_circulant_edges(n, a, b):
    """Get edges of circulant graph C(n, a, b)."""
    edges = set()
    for i in range(n):
        for s in (a, b):
            j = (i + s) % n
            if i < j:
                edges.add((i, j))
    return sorted(edges)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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
