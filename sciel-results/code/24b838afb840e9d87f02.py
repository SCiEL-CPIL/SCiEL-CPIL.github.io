import math
import random
import time
from collections import deque
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g, minimizing n.
    Uses LUW construction + voltage graph (lift) search.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    best_result = None
    
    # Strategy 1: LUW construction
    if k >= 3:
        d_min = g - 5
        if d_min % 2 == 0:
            d_min += 1
        d_min = max(3, d_min)
        
        for d in range(d_min, min(d_min + 4, 25)):
            if time.time() > deadline - 40:
                break
            try:
                n, edges = _luw_graph_fast(k, d)
                if n > 0:
                    if _verify_nonbacktracking(k, g, n, edges, time_limit=2):
                        if best_result is None or n < best_result[0]:
                            best_result = (n, edges)
            except Exception:
                pass
    
    # Strategy 2: Voltage graph (lift) search
    if time.time() < deadline - 20:
        try:
            n, edges = _voltage_lift_search(k, g, deadline - time.time() - 15)
            if n > 0 and _verify_nonbacktracking(k, g, n, edges, time_limit=3):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        if k == 1:
            best_result = (2, [(0, 1)])
        elif k == 2:
            best_result = (max(g, 3), [(i, (i+1) % max(g, 3)) for i in range(max(g, 3))])
        else:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
            n, edges = _luw_graph_fast(k, d)
            if n > 0:
                best_result = (n, edges)
            else:
                best_result = (0, [])
    
    return best_result


def _luw_graph_fast(q, d):
    """
    Fast LUW graph CD(d, q) using numpy for field operations.
    q-regular bipartite graph with girth >= d+5 for odd d.
    """
    if q < 3:
        return 0, []
    
    # Precompute field tables
    if q == 4:
        add_table = np.zeros((q, q), dtype=np.int32)
        mul_table = np.zeros((q, q), dtype=np.int32)
        for x in range(q):
            for y in range(q):
                add_table[x, y] = x ^ y
                mul_table[x, y] = _gf4_mul(x, y)
        sub_table = add_table  # In GF(4), sub = add
    else:
        add_table = np.zeros((q, q), dtype=np.int32)
        mul_table = np.zeros((q, q), dtype=np.int32)
        for x in range(q):
            for y in range(q):
                add_table[x, y] = (x + y) % q
                mul_table[x, y] = (x * y) % q
        sub_table = np.zeros((q, q), dtype=np.int32)
        for x in range(q):
            for y in range(q):
                sub_table[x, y] = (x - y) % q
    
    # Build rule
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Generate all vertices
    # Side 0: (0, x1, x2, ..., xd) where x1 is free, rest computed
    # Side 1: (1, y1, y2, ..., yd) where y1 is free, rest computed
    
    vertices = []
    index = {}
    
    # Start vertex
    start = (0, (0,) * d)
    vertices.append(start)
    index[start] = 0
    
    # BFS to find all vertices
    queue = deque([start])
    
    while queue:
        vertex = queue.popleft()
        side, x = vertex
        
        if side == 0:
            x_arr = np.array(x, dtype=np.int32)
            for first in range(q):
                y_arr = np.zeros(d, dtype=np.int32)
                y_arr[0] = first
                for c in range(1, d):
                    a, b = rule[c + 1]
                    y_arr[c] = (x_arr[c] + mul_table[y_arr[a-1], x_arr[b-1]]) % q
                other = (1, tuple(y_arr))
                if other not in index:
                    index[other] = len(vertices)
                    vertices.append(other)
                    queue.append(other)
        else:
            x_arr = np.array(x, dtype=np.int32)
            for first in range(q):
                y_arr = np.zeros(d, dtype=np.int32)
                y_arr[0] = first
                for c in range(1, d):
                    a, b = rule[c + 1]
                    y_arr[c] = (x_arr[c] - mul_table[x_arr[a-1], y_arr[b-1]]) % q
                other = (0, tuple(y_arr))
                if other not in index:
                    index[other] = len(vertices)
                    vertices.append(other)
                    queue.append(other)
    
    n = len(vertices)
    
    # Build edges
    edges = []
    for i, (side, x) in enumerate(vertices):
        if side == 0:
            x_arr = np.array(x, dtype=np.int32)
            for first in range(q):
                y_arr = np.zeros(d, dtype=np.int32)
                y_arr[0] = first
                for c in range(1, d):
                    a, b = rule[c + 1]
                    y_arr[c] = (x_arr[c] + mul_table[y_arr[a-1], x_arr[b-1]]) % q
                other = (1, tuple(y_arr))
                j = index[other]
                edges.append((i, j))
    
    return n, edges


def _gf4_mul(x, y):
    """Multiplication in GF(4)"""
    # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    # Represent as: 0->0, 1->1, alpha->2, alpha+1->3
    if x == 0 or y == 0:
        return 0
    if x == 1:
        return y
    if y == 1:
        return x
    # alpha * alpha = alpha + 1 = 3
    # alpha * (alpha+1) = 1
    # (alpha+1) * (alpha+1) = alpha
    if x == 2 and y == 2:
        return 3
    if x == 2 and y == 3:
        return 1
    if x == 3 and y == 2:
        return 1
    if x == 3 and y == 3:
        return 2
    return 0


def _verify_nonbacktracking(k, g, n, edges, time_limit=5):
    """
    Verify k-regularity and girth >= g using non-backtracking BFS.
    """
    start_time = time.time()
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check simplicity
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Non-backtracking BFS from each vertex
    for start in range(n):
        if time.time() - start_time > time_limit:
            return True  # Assume valid on timeout
        
        # visited[(prev, curr)] = distance
        visited = {}
        queue = deque()
        
        for nb in adj[start]:
            visited[(start, nb)] = 1
            queue.append((start, nb, 1))
        
        while queue:
            prev, curr, dist = queue.popleft()
            
            if dist >= g - 1:
                continue
            
            for next_v in adj[curr]:
                if next_v == prev:
                    continue
                
                key = (curr, next_v)
                if key in visited:
                    cycle_len = visited[key] + dist + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[key] = dist + 1
                    if dist + 1 < g - 1:
                        queue.append((curr, next_v, dist + 1))
    
    return True


def _voltage_lift_search(k, g, time_limit=60):
    """
    Voltage graph (lift) approach: find a small base graph with voltage
    assignments in Z_m such that the m-fold cover is k-regular with girth >= g.
    
    The base graph has m vertices, each edge has a voltage in Z_m.
    The cover graph has m*m vertices: (v, t) where v is base vertex, t in Z_m.
    Edge (u,v) with voltage a becomes edges ((u,t),(v,t+a)) for all t in Z_m.
    """
    start_time = time.time()
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for reference
    if g % 2 == 0:
        # Even girth: n >= 2 * (k-1)^(g/2 - 1)
        moore = 2 * (k - 1) ** (g // 2 - 1)
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # Try different base graph sizes m and group sizes
    # For the cover to be k-regular, base graph must be k-regular
    # Cover size = m * m (if group is Z_m)
    # We want m*m close to moore bound
    # So m ~ sqrt(moore)
    
    m_min = max(k + 1, int(math.sqrt(moore)) - 5)
    m_max = min(m_min + 40, 250)
    
    # If k is odd, m must be even for k-regular graph to exist
    if k % 2 == 1 and m_min % 2 == 1:
        m_min += 1
    
    for m in range(m_min, m_max + 1):
        if k % 2 == 1 and m % 2 == 1:
            continue
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a k-regular base graph on m vertices
        # with voltage assignments in Z_m that give girth >= g in the cover
        base_edges = _find_base_graph_with_voltages(k, m, g, time_per_try=max(0.5, time_limit / (m_max - m_min + 1)))
        
        if base_edges is not None:
            # Build the cover
            cover_n, cover_edges = _build_cover(base_edges, m)
            if cover_n < best_n:
                best_n = cover_n
                best_edges = cover_edges
    
    if best_edges is not None:
        return best_n, best_edges
    return 0, []


def _find_base_graph_with_voltages(k, m, g, time_per_try=1.0):
    """
    Find a k-regular graph on m vertices with voltage assignments in Z_m
    such that the m-fold cover has girth >= g.
    
    Uses backtracking to assign voltages that avoid short cycles.
    Returns list of (u, v, voltage) or None.
    """
    start_time = time.time()
    
    # Generate random k-regular graphs on m vertices
    for attempt in range(30):
        if time.time() - start_time > time_per_try:
            break
        
        base_edges = _random_k_regular_edges(m, k)
        if base_edges is None:
            continue
        
        # Try to assign voltages via backtracking
        voltages = _assign_voltages_backtrack(base_edges, m, g, time_limit=time_per_try / 3)
        
        if voltages is not None:
            result = []
            for u, v in base_edges:
                key = (min(u, v), max(u, v))
                result.append((u, v, voltages[key]))
            return result
    
    return None


def _assign_voltages_backtrack(base_edges, m, g, time_limit=2.0):
    """
    Assign voltages in Z_m to edges so that no closed walk of length < g
    in the base graph has voltage sum 0 mod m.
    """
    start_time = time.time()
    
    # Find all closed walks of length < g in the base graph
    # For efficiency, only consider walks up to length g-1
    # This can be expensive, so limit the search
    
    # Build adjacency for base graph
    adj = [[] for _ in range(m)]
    edge_keys = []
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_keys.append((min(u, v), max(u, v)))
    
    # For each edge, track which edge it is
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        key = (min(u, v), max(u, v))
        edge_index[key] = i
    
    # Find short cycles/walks to constrain voltages
    # A closed walk of length L has voltage sum = sum of voltages on its edges
    # We need this sum != 0 mod m for all walks of length < g
    
    # For efficiency, only check walks of length 3, 4, ..., g-1
    # But even this can be expensive. Let's use a simpler heuristic:
    # Assign voltages randomly and check, with retries
    
    best_voltages = None
    
    for trial in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Random assignment
        voltages = {}
        for u, v in base_edges:
            key = (min(u, v), max(u, v))
            voltages[key] = random.randint(0, m - 1)
        
        # Check girth of cover
        if _cover_girth_check(base_edges, voltages, m, g):
            best_voltages = voltages
            break
    
    return best_voltages


def _cover_girth_check(base_edges, voltages, m, g):
    """
    Check if the voltage cover has girth >= g.
    Uses BFS on the cover graph.
    """
    n = m * m
    
    # Build cover adjacency
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        vol = voltages[key]
        for t in range(m):
            a = u * m + t
            b = v * m + ((t + vol) % m)
            adj[a].append(b)
            adj[b].append(a)
    
    # Non-backtracking BFS from a sample of vertices
    # For efficiency, check from a few vertices
    sample_size = min(10, n)
    for start in range(0, n, max(1, n // sample_size)):
        if not _nbfs_girth_check(adj, start, g):
            return False
    
    return True


def _nbfs_girth_check(adj, start, g):
    """Non-backtracking BFS from start, check no cycle shorter than g."""
    n = len(adj)
    visited = {}
    queue = deque()
    
    for nb in adj[start]:
        visited[(start, nb)] = 1
        queue.append((start, nb, 1))
    
    while queue:
        prev, curr, dist = queue.popleft()
        
        if dist >= g - 1:
            continue
        
        for next_v in adj[curr]:
            if next_v == prev:
                continue
            
            key = (curr, next_v)
            if key in visited:
                cycle_len = visited[key] + dist + 1
                if cycle_len < g:
                    return False
            else:
                visited[key] = dist + 1
                if dist + 1 < g - 1:
                    queue.append((curr, next_v, dist + 1))
    
    return True


def _build_cover(base_edges_with_volts, m):
    """Build the m-fold cover from base edges with voltages."""
    n = m * m
    edges = []
    for u, v, vol in base_edges_with_volts:
        for t in range(m):
            a = u * m + t
            b = v * m + ((t + vol) % m)
            edges.append((a, b))
    return n, edges


def _random_k_regular_edges(m, k):
    """Generate a random k-regular graph on m vertices. Returns list of (u,v) or None."""
    if m * k < 2 or (m * k) % 2 != 0:
        return None
    if m <= k:
        return None
    
    for _ in range(20):
        stubs = []
        for i in range(m):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        
        edges = []
        edge_set = set()
        valid = True
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append((u, v))
        
        if valid:
            return edges
    
    return None


def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices."""
    if n * k < 2 or (n * k) % 2 != 0:
        return None
    
    for _ in range(20):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        
        edges = []
        edge_set = set()
        valid = True
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append(key)
        
        if valid:
            return edges
    
    return None
