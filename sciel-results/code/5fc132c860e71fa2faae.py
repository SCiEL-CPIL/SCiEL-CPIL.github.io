"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise. Optimized with early termination."""
    for start in range(n):
        # State: (vertex, prev_vertex) -> depth
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

def _build_and_check_lift(m, voltages, g):
    """Build a voltage graph lift of the Petersen graph with given voltages.
    Returns (n, edges) if valid girth >= g, else None."""
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in petersen_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    if _verify_girth_fast(adj, n, g):
        return n, edges
    return None

def _search_voltages_systematic(m, g, max_iters=2000, deadline=None):
    """Systematic search over voltage assignments using iterative improvement.
    Start from random assignment, then try local moves to eliminate short cycles."""
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    edge_list = petersen_edges
    
    best_result = None
    rng = _random.Random(42)
    
    for it in range(max_iters):
        if deadline and _time.time() > deadline:
            break
        
        # Initialize voltages randomly
        voltages = {}
        for u, v in edge_list:
            voltages[(u, v)] = rng.randint(0, m - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        
        # Try to improve by changing one voltage at a time
        improved = True
        while improved and (not deadline or _time.time() < deadline):
            improved = False
            for u, v in edge_list:
                if deadline and _time.time() > deadline:
                    break
                old_vol = voltages[(u, v)]
                # Try all possible voltage values for this edge
                for new_vol in range(m):
                    if new_vol == old_vol:
                        continue
                    voltages[(u, v)] = new_vol
                    voltages[(v, u)] = (-new_vol) % m
                    
                    result = _build_and_check_lift(m, voltages, g)
                    if result is not None:
                        best_result = result
                        return best_result
                    else:
                        # Revert
                        voltages[(u, v)] = old_vol
                        voltages[(v, u)] = (-old_vol) % m
        
        # If we made it through all edges without finding a solution,
        # do a random perturbation
        for _ in range(3):
            u, v = rng.choice(edge_list)
            voltages[(u, v)] = rng.randint(0, m - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    return best_result

def _build_lift(m, seed=0):
    """Build a voltage graph lift of the Petersen graph over Z_m with random voltages.
    Returns (n, adj, edges) or None if invalid."""
    rng = _random.Random(seed)
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
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
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, adj, edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
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
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    # Strategy 1: Systematic voltage search on Petersen lifts
    m_min = max(1, (moore + 9) // 10)
    
    # Try m values starting from minimum
    for m in range(m_min, m_min + 150):
        if _time.time() > deadline - 5:
            break
        n = 10 * m
        if n < moore:
            continue
        
        # First try systematic search
        result = _search_voltages_systematic(m, g, max_iters=5, deadline=deadline - 3)
        if result is not None:
            return result
        
        # Then try random lifts
        for attempt in range(30):
            if _time.time() > deadline - 3:
                break
            lift_result = _build_lift(m, seed=attempt * 1000 + m)
            if lift_result is None:
                continue
            n_lift, adj, edges = lift_result
            if _verify_girth_fast(adj, n_lift, g):
                return n_lift, edges
    
    # Strategy 2: Random 3-regular graphs
    if _time.time() < deadline - 10:
        n_try = moore
        while n_try < 10000 and _time.time() < deadline - 10:
            if n_try % 2 != 0:
                n_try += 1
                continue
            for attempt in range(50):
                if _time.time() > deadline - 5:
                    break
                stubs = []
                for v in range(n_try):
                    for _ in range(3):
                        stubs.append(v)
                _random.shuffle(stubs)
                adj = [set() for _ in range(n_try)]
                edges = []
                valid = True
                for i in range(0, n_try * 3, 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v or v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                if not valid:
                    continue
                visited = set([0])
                stack = [0]
                while stack:
                    v = stack.pop()
                    for w in adj[v]:
                        if w not in visited:
                            visited.add(w)
                            stack.append(w)
                if len(visited) != n_try:
                    continue
                if _verify_girth_fast(adj, n_try, g):
                    return n_try, edges
            n_try += 64
    
    # Fallback: LUW construction
    q = 3
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
# EVOLVE-BLOCK-END
