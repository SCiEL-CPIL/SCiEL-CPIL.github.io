import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k != 4:
        return 0, []
        
    # Priority 1: Voltage graph construction
    # This is likely to find the optimal or near-optimal solution for small k, g
    result = _voltage_search(k, g, deadline)
    if result is not None:
        n, edges = result
        if _verify(n, edges, k, g):
            return n, edges
            
    # Priority 2: Fallback to LUW if voltage search fails (unlikely to be better but safe)
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        if _verify(n, edges, k, g):
            return n, edges
            
    return 0, []

def _voltage_search(k, g, deadline):
    """
    Search for a voltage graph lift with girth >= g.
    Tries base graphs K_{4,4} and Octahedron.
    """
    if k != 4:
        return None
        
    base_graphs = [
        ("K44", _get_k44_edges()),
        ("Oct", _get_octahedron_edges())
    ]
    
    best_result = None
    best_n = float('inf')
    
    # We want to minimize n = m * |V_base|
    # So we should try small m first, but also larger base graphs might allow smaller m?
    # Actually, for fixed girth, larger m generally helps satisfy constraints.
    # We want the smallest n.
    # Let's try a range of m values.
    
    # For K44 (8 vertices), n = 8m.
    # For Oct (6 vertices), n = 6m.
    # If we find a solution with K44, m=40 -> n=320.
    # If we find a solution with Oct, m=54 -> n=324.
    # So K44 is slightly better if m is similar.
    
    # We will iterate m from 10 to 100.
    # To be efficient, we can interleave base graphs or just try K44 first as it's a standard cage base.
    
    for base_name, base_edges in base_graphs:
        base_n = len(set(v for e in base_edges for v in e))
        
        # Estimate required m.
        # Girth g means no closed walks of length < g.
        # Number of walks grows exponentially.
        # Let's try m in a range that yields n < 1000.
        max_m = min(200, int(1000 / base_n) + 10)
        
        for m in range(10, max_m):
            if time.time() > deadline - 20:
                break
                
            n = m * base_n
            if n >= best_n:
                continue
                
            # Try to find voltages
            voltages = _find_voltages(base_edges, m, g, deadline - 10)
            if voltages is not None:
                lift_edges = _build_lift(base_edges, voltages, m)
                if _verify(n, lift_edges, k, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, lift_edges)
                        
    return best_result

def _get_k44_edges():
    """K_{4,4} edges."""
    edges = []
    for i in range(4):
        for j in range(4):
            edges.append((i, j + 4))
    return edges

def _get_octahedron_edges():
    """Octahedron edges (6 vertices, 4-regular)."""
    opposites = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
    edges = []
    for i in range(6):
        for j in range(i + 1, 6):
            if j != opposites[i]:
                edges.append((i, j))
    return edges

def _enumerate_walks(base_edges, max_len):
    """
    Enumerate all non-backtracking closed walks of length 3 to max_len-1.
    Returns a list of lists of (edge_idx, sign).
    sign is +1 if traversed in direction of base_edges, -1 otherwise.
    """
    base_n = len(set(v for e in base_edges for v in e))
    adj = [[] for _ in range(base_n)]
    edge_map = {}
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        edge_map[(u, v)] = (idx, 1)
        edge_map[(v, u)] = (idx, -1)
        
    walks = []
    
    for L in range(3, max_len):
        # Start from each directed edge
        for u, v in base_edges:
            idx, sign = edge_map[(u, v)]
            # State: (prev, cur, path)
            # Path starts with this edge
            path = [(idx, sign)]
            prev = u
            cur = v
            
            # BFS to extend path to length L
            # We need to close back to u
            # We use a queue of (prev, cur, path_tuple)
            # To save memory, we can do DFS with recursion limit or iterative
            # Given small size, DFS is fine.
            
            def dfs(prev, cur, path):
                if len(path) == L:
                    if cur == u:
                        # Check if it's a valid closed walk
                        # It is non-backtracking by construction
                        walks.append(path[:])
                    return
                for (next_v, eidx) in adj[cur]:
                    if next_v == prev:
                        continue
                    e_u, e_v = base_edges[eidx]
                    if e_u == cur and e_v == next_v:
                        s = 1
                    else:
                        s = -1
                    dfs(cur, next_v, path + [(eidx, s)])
            
            dfs(prev, cur, path)
            
    return walks

def _find_voltages(base_edges, m, g, deadline):
    """
    Find voltages in Z_m such that no closed non-backtracking walk of length < g sums to 0.
    Uses Simulated Annealing.
    """
    n_edges = len(base_edges)
    walks = _enumerate_walks(base_edges, g)
    
    if not walks:
        return [0] * n_edges
        
    # Pre-compute the "signature" of each walk: list of (edge_idx, sign)
    # We will count violations: sum(v[e] * s) % m == 0
    
    # Initial random voltages
    random.seed(42) # Deterministic for reproducibility in search, but we can vary if needed
    voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
    
    def count_violations(vols):
        count = 0
        for walk in walks:
            s = 0
            for eidx, sign in walk:
                s = (s + sign * vols[eidx]) % m
            if s == 0:
                count += 1
        return count
        
    current_violations = count_violations(voltages)
    if current_violations == 0:
        return voltages[:]
        
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.95
    steps = 0
    max_steps = 5000 # Adjust based on time
    
    while T > T_min and time.time() < deadline:
        # Pick a random edge to change
        e_idx = random.randint(0, n_edges - 1)
        old_v = voltages[e_idx]
        new_v = random.randint(0, m - 1)
        if new_v == old_v:
            continue
            
        voltages[e_idx] = new_v
        new_violations = count_violations(voltages)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_violations = new_violations
        else:
            voltages[e_idx] = old_v
            
        T *= cooling
        steps += 1
        
        if current_violations == 0:
            return voltages[:]
            
        # Restart if stuck
        if steps % 1000 == 0 and current_violations > 0:
             # Maybe try a different random seed or reset
             pass
             
    if current_violations == 0:
        return voltages[:]
    return None

def _build_lift(base_edges, voltages, m):
    """Build the lifted graph edges."""
    base_n = len(set(v for e in base_edges for v in e))
    edges = []
    
    for (u, v), vol in zip(base_edges, voltages):
        for s in range(m):
            su = u * m + s
            sv = v * m + (s + vol) % m
            if su < sv:
                edges.append((su, sv))
            else:
                edges.append((sv, su))
                
    return sorted(set(edges))

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        return 0, []
        
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
    edges_list = sorted(set(edges))
    return n, edges_list

def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    
    if len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    INF = 10 ** 9
    for (u, v) in edges:
        dist = {}
        q = []
        for w in adj[u]:
            if w == v:
                continue
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True
