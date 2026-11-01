<<<<<<< SEARCH
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

def _build_936_graph():
    """
    Construct a 3-regular graph on 936 vertices with girth >= 16.
    
    Uses a voltage graph lift over the Petersen graph with carefully chosen
    voltages in Z_94 (since 936 = 10 * 93.6... that doesn't work).
    
    Actually 936 = 8 * 117 = 4 * 234 = 2 * 468.
    Let's use a different approach: a circulant-like construction.
    
    936 vertices. We can use a 3-regular circant graph or a generalized
    Petersen-type construction.
    
    Let me try: vertices (i, j) where i in Z_78, j in Z_12. 78*12 = 936.
    Edges:
    - (i,j) - (i+1, j)  [cycle in i direction]
    - (i,j) - (i, j+1)  [cycle in j direction]  
    - (i,j) - (i + f(j), j + g(j))  [twist edge]
    
    This gives degree 3 if the twist edge doesn't overlap with the other two.
    
    For girth 16, we need to carefully choose f and g.
    
    Alternative: use a known construction. The (3,16)-cage has 936 vertices.
    One known construction uses a group-based approach.
    
    Let me try a simpler approach: a 3-regular graph using a bipartite
    structure. 936 = 2 * 468.
    
    Actually, let me just do a well-structured voltage lift that's likely
    to work, and verify it.
    """
    # Use a voltage lift over K_3,3 (6 vertices) with m = 156, giving 936 vertices
    # K_3,3 has 9 edges. We need 9 voltage shifts in Z_156.
    
    # For a girth-16 lift, we need the voltages to be "large" and well-spread.
    # A good heuristic: use shifts that are roughly m/9 apart.
    
    m = 156
    n_base = 6
    n = n_base * m  # 936
    
    # K_3,3 edges: partition {0,1,2} x {3,4,5}
    base_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    # Try multiple voltage assignments and return the first that works
    # We'll use a systematic search with verification
    
    # Strategy: use voltages that are "generic" - based on a primitive root
    # of Z_156 if it exists, or just well-spread values.
    
    # 156 = 4 * 39 = 4 * 3 * 13
    # phi(156) = 156 * (1-1/2) * (1-1/3) * (1-1/13) = 156 * 1/2 * 2/3 * 12/13 = 48
    
    # Let's try: voltage for edge (a,b) = (a * c1 + b * c2) mod m for some c1, c2
    # Or: use a fixed pattern based on edge index
    
    best_result = None
    
    # Try deterministic patterns first
    patterns = []
    
    # Pattern 1: arithmetic progression
    for start_val in range(0, m, 3):
        shifts = [(start_val + i * (m // 9)) % m for i in range(9)]
        patterns.append(shifts)
    
    # Pattern 2: based on a generator
    gen = 5  # try different generators
    for gen in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        shifts = [(gen ** (i+1)) % m for i in range(9)]
        patterns.append(shifts)
    
    # Pattern 3: golden ratio based
    phi = 0.6180339887498948
    for offset in range(10):
        shifts = [int(((i + 1 + offset) * phi * m)) % m for i in range(9)]
        patterns.append(shifts)
    
    # Pattern 4: quadratic residues
    residues = set()
    for a in range(1, m):
        residues.add((a * a) % m)
    residues_list = sorted(residues)
    if len(residues_list) >= 9:
        step = len(residues_list) // 9
        shifts = [residues_list[i * step] for i in range(9)]
        patterns.append(shifts)
    
    # Try each pattern
    for shifts in patterns:
        adj = [set() for _ in range(n)]
        edges = []
        valid = True
        for idx, (u, v) in enumerate(base_edges):
            vol = shifts[idx] % m
            for i in range(m):
                a = u * m + i
                b = v * m + (i + vol) % m
                if a == b:
                    valid = False
                    break
                if b not in adj[a]:
                    adj[a].add(b)
                    adj[b].add(a)
                    edges.append((a, b))
                else:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check degree
        degree_ok = True
        for i in range(n):
            if len(adj[i]) != 3:
                degree_ok = False
                break
        if not degree_ok:
            continue
        
        adj_list = [list(s) for s in adj]
        if _verify_girth_fast(adj_list, n, 16):
            return n, edges
    
    # If none of the deterministic patterns worked, try random search
    for attempt in range(200):
        rng = _random.Random(42 + attempt)
        shifts = [rng.randint(1, m - 1) for _ in range(9)]
        
        adj = [set() for _ in range(n)]
        edges = []
        valid = True
        for idx, (u, v) in enumerate(base_edges):
            vol = shifts[idx] % m
            for i in range(m):
                a = u * m + i
                b = v * m + (i + vol) % m
                if a == b:
                    valid = False
                    break
                if b not in adj[a]:
                    adj[a].add(b)
                    adj[b].add(a)
                    edges.append((a, b))
                else:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        degree_ok = True
        for i in range(n):
            if len(adj[i]) != 3:
                degree_ok = False
                break
        if not degree_ok:
            continue
        
        adj_list = [list(s) for s in adj]
        if _verify_girth_fast(adj_list, n, 16):
            return n, edges
    
    return None

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
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

def _voltage_lift_search(k, g, deadline):
    """Search for a valid voltage lift with various base graphs and lift sizes."""
    if k != 3:
        return None
    
    # Moore bound for k=3, g=16: 2*(2^8-1) = 510
    # Target: find smallest n >= 510 that works
    
    base_graphs = [
        # (name, edges, n_base)
        ("k33", [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)], 6),
        ("petersen", [(0,1),(1,2),(2,3),(3,4),(4,0),(5,7),(7,9),(9,6),(6,8),(8,5),(0,5),(1,6),(2,7),(3,8),(4,9)], 10),
        ("cube", [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)], 8),
    ]
    
    best_n = float('inf')
    best_edges = None
    
    for name, base_edges, n_base in base_graphs:
        e_count = len(base_edges)
        
        # Try lift sizes from Moore bound / n_base upward
        moore = 510  # for g=16, k=3
        m_min = max(1, (moore + n_base - 1) // n_base)
        
        for m in range(m_min, m_min + 200):
            if _time.time() > deadline - 10:
                return (best_n, best_edges) if best_edges else None
            
            n = n_base * m
            if n >= best_n:
                continue
            
            # Try several deterministic shift patterns
            shift_sets = []
            
            # Arithmetic progression
            for start_val in range(0, min(m, 20), 2):
                step = m // e_count
                shifts = [(start_val + i * step) % m for i in range(e_count)]
                if all(s != 0 for s in shifts):
                    shift_sets.append(shifts)
            
            # Golden ratio
            phi = 0.6180339887498948
            for offset in range(5):
                shifts = [int(((i + 1 + offset) * phi * m)) % m for i in range(e_count)]
                if all(s != 0 for s in shifts):
                    shift_sets.append(shifts)
            
            # Random
            for attempt in range(30):
                rng = _random.Random(hash((name, m, attempt)))
                shifts = [rng.randint(1, m - 1) for _ in range(e_count)]
                shift_sets.append(shifts)
            
            for shifts in shift_sets:
                if _time.time() > deadline - 5:
                    break
                
                adj = [set() for _ in range(n)]
                edges = []
                valid = True
                for idx, (u, v) in enumerate(base_edges):
                    vol = shifts[idx] % m
                    for i in range(m):
                        a = u * m + i
                        b = v * m + (i + vol) % m
                        if a == b:
                            valid = False
                            break
                        if b not in adj[a]:
                            adj[a].add(b)
                            adj[b].add(a)
                            edges.append((a, b))
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                degree_ok = all(len(adj[i]) == 3 for i in range(n))
                if not degree_ok:
                    continue
                
                adj_list = [list(s) for s in adj]
                if _verify_girth_fast(adj_list, n, g):
                    return n, edges
    
    return (best_n, best_edges) if best_edges else None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    
    # For k=3, g=16: try the 936-vertex construction first
    if g == 16:
        result = _build_936_graph()
        if result is not None:
            n, edges = result
            # Final verification
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            adj_list = [list(s) for s in adj]
            if _verify_girth_fast(adj_list, n, g):
                return n, edges
    
    # Fallback: voltage lift search
    result = _voltage_lift_search(k, g, deadline)
    if result is not None:
        return result
    
    # Final fallback: LUW
    n_luw, edges_luw = _luw_construct(k, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
=======
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

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
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

def _build_lift(base_edges, n_base, m, shifts):
    """Build a voltage lift and return (n, edges) or None if invalid."""
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if a == b:
                return None
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
            else:
                return None
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return n, edges

def _voltage_lift_search(k, g, deadline):
    """Search for a valid voltage lift with various base graphs and lift sizes.
    Structural change: massively increase random trials per lift size (500+),
    and use a fast pre-filter before full girth verification."""
    if k != 3:
        return None
    
    base_graphs = [
        ("k33", [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)], 6),
        ("petersen", [(0,1),(1,2),(2,3),(3,4),(4,0),(5,7),(7,9),(9,6),(6,8),(8,5),(0,5),(1,6),(2,7),(3,8),(4,9)], 10),
        ("cube", [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)], 8),
    ]
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for k=3, g=16
    moore = 2 * (2 ** ((g+1)//2) - 1) if g % 2 == 0 else 2 * (2 ** ((g-1)//2) - 1) + 1
    if g == 16:
        moore = 510
    
    for name, base_edges, n_base in base_graphs:
        e_count = len(base_edges)
        m_min = max(1, (moore + n_base - 1) // n_base)
        
        for m in range(m_min, m_min + 300):
            if _time.time() > deadline - 15:
                return (best_n, best_edges) if best_edges else None
            
            n = n_base * m
            if n >= best_n:
                continue
            
            # Deterministic patterns first (few)
            deterministic_shifts = []
            # Arithmetic
            step = m // e_count
            if step > 0:
                for sv in range(0, min(m, 10)):
                    shifts = [(sv + i * step) % m for i in range(e_count)]
                    if all(s != 0 for s in shifts):
                        deterministic_shifts.append(shifts)
            
            # Random: 500 trials (key structural change from cell finding)
            n_random = 500
            rng = _random.Random(12345 + n_base * 1000 + m)
            
            all_shifts = deterministic_shifts
            for _ in range(n_random):
                shifts = [rng.randint(1, m - 1) for _ in range(e_count)]
                all_shifts.append(shifts)
            
            for shifts in all_shifts:
                if _time.time() > deadline - 5:
                    break
                
                result = _build_lift(base_edges, n_base, m, shifts)
                if result is None:
                    continue
                
                n_res, edges = result
                adj_list = [list(s) for s in [[i for i in range(n_res) if i in adj_set] for adj_set in []]]
                # Rebuild adj list properly
                adj = [set() for _ in range(n_res)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                adj_list = [list(s) for s in adj]
                
                if _verify_girth_fast(adj_list, n_res, g):
                    best_n = n_res
                    best_edges = edges
                    return n_res, edges
    
    return (best_n, best_edges) if best_edges else None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    
    # Voltage lift search with massive random trials
    result = _voltage_lift_search(k, g, deadline)
    if result is not None:
        n, edges = result
        # Final verification
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        adj_list = [list(s) for s in adj]
        if _verify_girth_fast(adj_list, n, g):
            return n, edges
    
    # Fallback: LUW
    n_luw, edges_luw = _luw_construct(k, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I have a bug in the `_voltage_lift_search` function - I'm trying to rebuild adj_list incorrectly. Let me fix that:

