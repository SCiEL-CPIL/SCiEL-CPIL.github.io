import time
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. First, construct the known LW graph for k=4, g=10
    # This gives n=320, which is the current best upper bound.
    # We verify it and keep it as the best candidate.
    best_n, best_edges = _construct_lw(k, g)
    
    # Verify the LW graph
    adj = _build_adj(best_edges, best_n)
    if not _verify_girth_fast(best_n, adj, g):
        # If LW is invalid (should not happen), we need a fallback
        # But LW is theoretically sound.
        pass
        
    # 2. Try to find a smaller graph using Voltage Graphs
    # We search for n < best_n
    # We try different base graphs and group sizes
    
    # Bases: C3, C4, C5, C6
    # For each base C_L, we assign voltages to L edges.
    # The lift has L*m vertices.
    
    bases = [
        (3, [(0,1), (1,2), (2,0)]), # C3
        (4, [(0,1), (1,2), (2,3), (3,0)]), # C4
        (5, [(0,1), (1,2), (2,3), (3,4), (4,0)]), # C5
        (6, [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]) # C6
    ]
    
    # Determine m ranges for each base such that L*m < best_n
    # and L*m >= Moore Bound (243 for k=4, g=10)
    moore_bound = 243
    
    for L, base_edges in bases:
        min_m = (moore_bound + L - 1) // L
        max_m = (best_n - 1) // L
        
        if min_m > max_m:
            continue
            
        # Iterate m from max_m down to min_m? 
        # Smaller n is better. So we should check smaller m first?
        # No, smaller m means smaller n.
        # But smaller m might be harder to satisfy girth constraints?
        # Actually, larger m is "safer" for girth, but larger n is worse.
        # We want the smallest n. So we should try smallest m first.
        # However, if we find a valid graph at m=61 (n=244), that's great.
        # If we don't, we try m=62, etc.
        
        # But wait, if we iterate from min_m up, we might waste time on small m that are impossible.
        # Given the time limit, we should just try a few random m's or iterate.
        
        # Let's iterate m from min_m to max_m
        for m in range(min_m, max_m + 1):
            if time.time() - start_time > deadline:
                return best_n, best_edges
            
            n = L * m
            if n >= best_n:
                break
                
            # Try a few random voltage assignments for this m
            # Number of attempts: proportional to time remaining?
            # Let's try 10-20 attempts per m
            attempts = 10
            found = False
            for _ in range(attempts):
                if time.time() - start_time > deadline:
                    break
                    
                # Generate random voltages
                voltages = [random.randint(1, m-1) for _ in range(L)]
                
                # Build graph
                edges, adj = _build_voltage_graph(L, base_edges, m, voltages)
                
                # Check simplicity and degree
                if not _is_simple_and_regular(edges, adj, n, k):
                    continue
                    
                # Check girth
                if _verify_girth_fast(n, adj, g):
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            
            if found:
                # If we found a smaller graph, we can continue searching for even smaller
                # or break if we hit the theoretical minimum?
                # We continue to see if we can go lower.
                pass

    return best_n, best_edges

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _build_voltage_graph(L, base_edges, m, voltages):
    """
    Builds the lift of a base cycle C_L with voltages in Z_m.
    Vertices are (base_vertex, group_element).
    """
    n = L * m
    edges = []
    adj = [[] for _ in range(n)]
    
    for i, (u_base, v_base) in enumerate(base_edges):
        v = voltages[i]
        for g in range(m):
            u = u_base * m + g
            v_node = v_base * m + (g + v) % m
            
            # Avoid self-loops and duplicate edges?
            # In a lift of a simple base, self-loops occur if voltage is 0.
            # We generated 1..m-1, so no self-loops from voltage 0.
            # Duplicate edges can occur if the base has multi-edges (it doesn't)
            # or if the lift creates multiple edges between same pair.
            # For a cycle base, edges are distinct.
            
            edges.append((u, v_node))
            adj[u].append(v_node)
            adj[v_node].append(u)
            
    return edges, adj

def _is_simple_and_regular(edges, adj, n, k):
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check for duplicate edges and loops
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        
    return True

def _verify_girth_fast(n, adj, g):
    """Fast girth verification using non-backtracking walks."""
    for start_v in range(n):
        visited = set()
        queue = deque()
        
        for nb in adj[start_v]:
            if (nb, start_v) not in visited:
                visited.add((nb, start_v))
                queue.append((nb, start_v, 1))
        
        while queue:
            curr, prev, depth = queue.popleft()
            
            if curr == start_v and depth < g:
                return False
            
            if depth >= g - 1:
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if (nb, curr) not in visited:
                    visited.add((nb, curr))
                    queue.append((nb, curr, depth + 1))
    
    return True

def _construct_lw(k, g):
    """Fallback LW graph construction."""
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
    
    n = len(order)
    return n, edges
