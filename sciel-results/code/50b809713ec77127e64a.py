import time
import random
import numpy as np
from collections import deque

# Keep the original LUW code as a fallback
def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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

def _petersen_lift(m, seed=None):
    """
    Construct a 3-regular graph by lifting the Petersen graph.
    
    The Petersen graph has 10 vertices and is 3-regular with girth 5.
    A voltage graph lift with m sheets produces a 3-regular graph on 10*m vertices.
    
    We use a random assignment of Z_m voltages to the Petersen edges,
    then the lift has vertices (i, j) where i in 0..9 (Petersen vertex)
    and j in 0..m-1 (sheet). Edge (i, j) -- (i', j + v(e) mod m) for Petersen edge e.
    """
    if seed is not None:
        random.seed(seed)
    
    # Petersen graph edges (standard labeling)
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    n_edges = len(petersen_edges)
    
    # Assign random voltages in Z_m to each edge
    voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
    
    # Build the lifted graph
    n = 10 * m
    edges = []
    for idx, (u, v) in enumerate(petersen_edges):
        vol = voltages[idx]
        for j in range(m):
            # Edge from (u, j) to (v, j + vol) mod m
            a = u * m + j
            b = v * m + ((j + vol) % m)
            edges.append((a, b))
    
    return n, edges

def _find_short_cycles(n, edges, target_girth, max_depth):
    """
    Find all cycles of length < target_girth using BFS from each vertex.
    Returns a list of cycles (each cycle is a list of vertex indices).
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycles = []
    for start in range(n):
        # BFS with parent tracking to reconstruct cycles
        # visited[v] = (depth, parent)
        visited = {}
        visited[start] = (0, -1)
        queue = deque([(start, 0, -1)])
        
        while queue:
            node, depth, parent = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor not in visited:
                    visited[neighbor] = (depth + 1, node)
                    queue.append((neighbor, depth + 1, node))
                else:
                    v_depth, v_parent = visited[neighbor]
                    # Cycle length
                    cycle_len = depth + v_depth + 1
                    if cycle_len < target_girth and cycle_len >= 3:
                        # Reconstruct the cycle
                        # Path from start to node
                        path1 = []
                        cur = node
                        while cur != -1:
                            path1.append(cur)
                            cur = visited[cur][1]
                        path1.reverse()
                        
                        # Path from start to neighbor
                        path2 = []
                        cur = neighbor
                        while cur != -1:
                            path2.append(cur)
                            cur = visited[cur][1]
                        path2.reverse()
                        
                        # The cycle is path1 + reversed(path2) minus the common prefix
                        # Find LCP
                        lcp = 0
                        min_len = min(len(path1), len(path2))
                        while lcp < min_len and path1[lcp] == path2[lcp]:
                            lcp += 1
                        
                        cycle = path1[lcp:] + path2[lcp:][::-1]
                        # Normalize: rotate so smallest vertex is first
                        if cycle:
                            min_idx = cycle.index(min(cycle))
                            cycle = cycle[min_idx:] + cycle[:min_idx]
                        cycles.append(tuple(cycle))
    
    return cycles

def _switch_edge(n, edges, adj, u, v, u2, v2):
    """
    Switch edges (u,v) and (u2,v2) to (u,v2) and (u2,v).
    Returns True if the switch is valid (no loops, no multi-edges).
    """
    if u == v2 or u2 == v:
        return False
    if (u, v2) in adj or (v2, u) in adj:
        return False
    if (u2, v) in adj or (v, u2) in adj:
        return False
    return True

def _try_improve_girth(n, edges, adj, target_girth, max_iter, deadline):
    """
    Local search: find short cycles and try to break them via edge switches.
    """
    edges_set = set()
    for u, v in edges:
        edges_set.add((min(u,v), max(u,v)))
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find a short cycle
        max_depth = (target_girth - 1) // 2
        cycles = _find_short_cycles(n, edges, target_girth, max_depth)
        
        if not cycles:
            return n, edges, adj
        
        cycle = cycles[0]
        cycle_len = len(cycle)
        
        # Try to break the cycle by switching an edge in the cycle
        # with an edge outside the cycle
        cycle_edges = []
        for i in range(cycle_len):
            a, b = cycle[i], cycle[(i+1) % cycle_len]
            cycle_edges.append((a, b))
        
        found_switch = False
        for (a, b) in cycle_edges:
            if time.time() > deadline:
                return n, edges, adj
            
            # Try to switch edge (a,b) with some other edge (c,d) not in the cycle
            # We need c != a, c != b, d != a, d != b for a clean switch
            # New edges: (a,d) and (c,b) must not exist
            
            for c in adj[a]:
                if c == b or c in cycle:
                    continue
                for d in adj[c]:
                    if d == c or d in cycle or d == b:
                        continue
                    if _switch_edge(n, edges, adj, a, b, c, d):
                        # Perform the switch
                        edges.remove((a, b)) if (a, b) in edges else edges.remove((b, a))
                        edges.remove((c, d)) if (c, d) in edges else edges.remove((d, c))
                        edges.append((a, d))
                        edges.append((c, b))
                        
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].remove(d)
                        adj[d].remove(c)
                        
                        adj[a].append(d)
                        adj[d].append(a)
                        adj[c].append(b)
                        adj[b].append(c)
                        
                        found_switch = True
                        break
                if found_switch:
                    break
            if found_switch:
                break
        
        if not found_switch:
            # Try random switches
            random.shuffle(edges)
            for i in range(min(100, len(edges))):
                if time.time() > deadline:
                    return n, edges, adj
                a, b = edges[i]
                # Find a random edge to switch with
                for j in range(min(100, len(edges))):
                    if i == j:
                        continue
                    c, d = edges[j]
                    if len(set([a,b,c,d])) < 4:
                        continue
                    if _switch_edge(n, edges, adj, a, b, c, d):
                        edges[i] = (a, d)
                        edges[j] = (c, b)
                        
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].remove(d)
                        adj[d].remove(c)
                        
                        adj[a].append(d)
                        adj[d].append(a)
                        adj[c].append(b)
                        adj[b].append(c)
                        
                        found_switch = True
                        break
                if found_switch:
                    break
            if not found_switch:
                break
    
    return n, edges, adj

def _check_girth_fast(n, adj, min_girth):
    """Fast girth check using BFS from each vertex."""
    max_depth = (min_girth - 1) // 2
    
    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])
        
        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _generate_random_3reg(n, seed=None):
    if seed is not None:
        random.seed(seed)
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    
    edges = set()
    attempts = 0
    while attempts < 100:
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((u, v))
        if valid:
            return list(edges)
        attempts += 1
    return None

def _check_girth(n, edges, min_girth):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex to check for cycles < min_girth
    # We only need to check up to depth floor((min_girth-1)/2)
    max_depth = (min_girth - 1) // 2
    
    for start in range(n):
        # visited[v] = depth
        visited = [-1] * n
        visited[start] = 0
        queue = [(start, 0, -1)] # node, depth, parent
        
        while queue:
            node, depth, parent = queue.pop(0)
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    # Found a cycle
                    # Cycle length is depth + visited[neighbor] + 1
                    # But we must ensure it's a simple cycle and not just a back edge to an ancestor
                    # In BFS, if we find a visited neighbor, the cycle length is depth + visited[neighbor] + 1
                    # However, this might not be the shortest cycle through that edge.
                    # But if ANY cycle is < min_girth, we fail.
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    if k != 3:
        # Fallback for non-3-regular
        return _luw_construct(k, g)
    
    best_result = None
    
    # Strategy: Petersen lifts with local search
    # Try different lift sizes m
    # n = 10*m vertices
    # We want to find a lift with girth >= g
    
    # Start with smaller lifts and increase
    # For g=16, we need girth 16
    # Petersen has girth 5, so lifts may have higher girth
    
    m_values = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300]
    
    for m in m_values:
        if time.time() > deadline:
            break
        
        n = 10 * m
        
        # Try multiple random voltage assignments
        for seed in range(10):
            if time.time() > deadline:
                break
            
            n, edges = _petersen_lift(m, seed=seed)
            
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check if girth is already sufficient
            if _check_girth_fast(n, adj, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, list(edges))
                break  # Found a good one for this m
            
            # Try local search to improve girth
            if time.time() > deadline - 20:
                continue
            
            max_iter = 5000
            n2, edges2, adj2 = _try_improve_girth(n, edges, adj, g, max_iter, deadline - 10)
            
            # Verify
            if _check_girth_fast(n2, adj2, g):
                if best_result is None or n2 < best_result[0]:
                    best_result = (n2, list(edges2))
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback to LUW
    return _luw_construct(k, g)
