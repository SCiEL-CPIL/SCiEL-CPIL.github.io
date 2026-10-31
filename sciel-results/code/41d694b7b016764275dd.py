import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        # Fallback for other k (should not be hit for this task)
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
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


def _voltage_lift_search(k, g, deadline, seed=42):
    """
    Voltage lift of the Petersen graph to get a 3-regular graph with high girth.
    
    The Petersen graph is 3-regular with girth 5.
    A voltage lift with modulus m produces a 3-regular graph on 10m vertices.
    A cycle in the lift of length L corresponds to a closed walk in the base of length L
    whose voltage sum is 0 mod m.
    
    For the Petersen graph, the only cycles have lengths 5, 6, 8, 9, 10.
    If we assign voltages such that no cycle of length < g has voltage sum 0 mod m,
    then the lift has no cycle of length < g.
    
    Actually, this is a sufficient but not necessary condition. A non-simple closed walk
    could have voltage sum 0 even if all simple cycles have non-zero sum.
    However, for small m and small base, this is a good heuristic.
    
    We search for voltages using random assignment + local search.
    """
    if k != 3:
        return None
    
    random.seed(seed)
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    m_edges = len(petersen_edges)  # 15
    
    # Build adjacency with edge indices
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))
    
    # Find all simple cycles in Petersen graph
    cycles = []
    seen = set()
    
    def dfs_cycles(start, curr, visited, edge_path, length):
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs_cycles(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
    
    for start in range(n_base):
        visited = {start}
        dfs_cycles(start, start, visited, [], 0)
    
    # We need to find voltages v[e] for e in 0..14 such that
    # for every cycle C, sum(v[e] for e in C) != 0 mod m
    # And the lift graph has girth >= g.
    
    # Try different moduli m (primes are good for avoiding zero sums)
    # For g=16, we need to avoid cycles of length 5,6,7,8,9,10,11,12,13,14,15
    # Petersen has cycles of length 5, 6, 8, 9, 10
    # For each such cycle C of length L, we need m/gcd(sum, m) * L >= g
    # If sum != 0 mod m, then m/gcd >= 2 (at least), so we need 2L >= g, i.e., L >= g/2 = 8
    # For L=5: need 5 * (m/gcd) >= 16 => m/gcd >= 4 (since 5*3=15 < 16)
    # For L=6: need 6 * (m/gcd) >= 16 => m/gcd >= 3 (since 6*2=12 < 16)
    # For L=8: need 8 * (m/gcd) >= 16 => m/gcd >= 2
    # For L=9: need 9 * (m/gcd) >= 16 => m/gcd >= 2
    # For L=10: need 10 * (m/gcd) >= 16 => m/gcd >= 2
    
    # So for prime m, sum != 0 implies m/gcd = m, so we need:
    # L=5: m >= 4 (so m >= 5)
    # L=6: m >= 3 (so m >= 5)
    # L=8,9,10: m >= 2 (so m >= 5)
    # Thus any prime m >= 5 works IF all cycles have non-zero sum.
    
    # But we also need to handle non-simple walks. Let's just verify empirically.
    
    primes_to_try = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for m in primes_to_try:
        if time.time() > deadline:
            break
        
        # Try random voltage assignments
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(1, m - 1) for _ in range(m_edges)]
            
            # Check all cycles
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build the lift
            n = 10 * m
            edges = []
            for idx, (u, v) in enumerate(petersen_edges):
                vol = voltages[idx]
                for j in range(m):
                    a = u * m + j
                    b = v * m + ((j + vol) % m)
                    edges.append((a, b))
            
            # Verify girth
            if _check_girth_fast(n, edges, g):
                return n, edges
        
        # If random fails, try a more structured approach
        # Assign voltages based on edge positions
        # This is a heuristic; may not work
        pass
    
    return None


def _bipartite_double_cover(k, g, deadline, seed=42):
    """
    Another approach: start with a 3-regular graph and take its bipartite double cover.
    The bipartite double cover of a graph G on n vertices is a 3-regular bipartite graph
    on 2n vertices. If G has girth h, the cover has girth 2h or 2h+1 (if G is non-bipartite).
    
    So if we can find a 3-regular graph with girth 8, its double cover has girth 16 or 17.
    But finding a 3-regular graph with girth 8 on few vertices is also hard.
    The smallest known is the (3,8)-cage, which has 40 vertices (the (3,8)-cage is the 40-vertex graph).
    Its double cover would have 80 vertices. But does it have girth 16?
    If the (3,8)-cage is non-bipartite, the cover has girth 17.
    If it's bipartite, the cover has girth 16.
    
    Actually, let's try a different approach.
    Use the (3,7)-cage or similar and see if we can get girth 16.
    This is getting complicated. Let's stick with the voltage lift.
    """
    return None


def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin

    if k != 3:
        return _luw_construct(k, g)

    # Try voltage lift first (structural change)
    result = _voltage_lift_search(k, g, deadline - 20, seed=42)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
