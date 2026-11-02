import time
import sys
import random
import numpy as np
from collections import deque
from itertools import product

def _bfs_girth(adj):
    """Compute the girth of an undirected graph via BFS from each vertex."""
    n = len(adj)
    best = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle = dist[u] + dist[v] + 1
                    if cycle < best:
                        best = cycle
        if best <= 2:
            break
    return best if best < float('inf') else 0

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (from seed)."""
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

def _petersen_edges():
    """Return the 15 edges of the Petersen graph on vertices 0..9."""
    # Standard Petersen: outer 5-cycle (0-1-2-3-4-0), inner star (5-7-9-6-8-5), spokes (i, i+5)
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star (Petersen's inner is a 5-pointed star: 5-7, 7-9, 9-6, 6-8, 8-5)
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i + 1) % 5]))
    # Spokes
    for i in range(5):
        edges.append((i, i + 5))
    return edges

def _check_girth_nb_fast(adj, min_girth):
    """Check if graph has girth >= min_girth using non-backtracking walks."""
    n = len(adj)
    for s in range(n):
        for v in adj[s]:
            dist = {}
            dist[(s, v)] = 1
            q = deque([(s, v)])
            while q:
                u, w = q.popleft()
                d = dist[(u, w)]
                if d >= min_girth:
                    continue
                for x in adj[w]:
                    if x == u:
                        continue
                    if (w, x) in dist:
                        if d + 1 < min_girth:
                            return False
                    else:
                        dist[(w, x)] = d + 1
                        q.append((w, x))
    return True

def _voltage_lift_petersen(voltages, group_order):
    """
    Construct a voltage graph lift of the Petersen graph over Z_{group_order}.
    voltages: dict mapping edge (u,v) to an integer voltage in [1, group_order-1].
    Returns (n, edges) or None if invalid.
    """
    petersen_edges = _petersen_edges()
    n = 10 * group_order
    
    # Build adjacency for the lifted graph
    # Vertex (i, k) where i in 0..9, k in 0..group_order-1
    # Edge between (u, k) and (v, (k + voltages[(u,v)]) % group_order)
    # and also (v, (k - voltages[(u,v)]) % group_order) for the reverse direction
    # Actually, for an undirected voltage graph, we assign a voltage to each edge.
    # The lift has vertices (v, g) for v in V, g in G.
    # For each edge e=(u,v) with voltage alpha, we have edges:
    #   (u, g) -- (v, g + alpha)
    #   (v, g) -- (u, g - alpha)
    # Wait, that's not right for undirected. Let's think.
    # In a voltage graph, each edge has a voltage. The lift is a covering graph.
    # For an undirected edge e={u,v} with voltage alpha, the edges in the lift are:
    #   {(u, g), (v, g + alpha)} for all g in G
    #   {(v, g), (u, g - alpha)} for all g in G
    # But these are the same set of edges if we consider undirected.
    # So for each g, we add edge between (u, g) and (v, (g + alpha) % group_order).
    
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    for u, v in petersen_edges:
        alpha = voltages[(u, v)]
        for g in range(group_order):
            u_lift = u * group_order + g
            v_lift = v * group_order + (g + alpha) % group_order
            if u_lift < v_lift:
                edge_key = (u_lift, v_lift)
            else:
                edge_key = (v_lift, u_lift)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return None
    
    # Check for duplicate edges (shouldn't happen with proper voltage assignment)
    # Already handled by edge_set
    
    # Check girth
    if not _check_girth_nb_fast(adj, 10):
        return None
    
    edges = list(edge_set)
    return n, edges

def _construct_4_10_cage():
    """Construct a 4-regular graph of girth >= 10 on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n_target = 320
    group_order = 32
    petersen_edges = _petersen_edges()
    
    # We need to assign voltages to each of the 15 edges of the Petersen graph.
    # Each voltage is in {1, 2, ..., 31}.
    # The girth of the lift is determined by the minimum length of non-backtracking
    # closed walks in the Petersen graph whose total voltage sums to 0 mod 32.
    # A non-backtracking closed walk of length L in the Petersen graph corresponds
    # to a cycle in the lift if and only if the sum of voltages along the walk
    # is 0 mod 32.
    
    # To maximize the girth, we want to avoid short non-backtracking closed walks
    # with zero voltage sum.
    
    # The Petersen graph has girth 5. So we need to ensure no closed walk of
    # length 5, 6, 7, 8, or 9 has zero voltage sum.
    
    # Let's try a systematic search over voltage assignments.
    # There are 31^15 possible assignments, which is too many.
    # But we can use symmetry and pruning.
    
    # First, let's try some structured assignments.
    # Idea: assign voltages based on the structure of the Petersen graph.
    
    # The Petersen graph has a natural symmetry. Let's label the edges:
    # Outer cycle: e0=(0,1), e1=(1,2), e2=(2,3), e3=(3,4), e4=(4,0)
    # Inner star: e5=(5,7), e6=(7,9), e7=(9,6), e8=(6,8), e9=(8,5)
    # Spokes: e10=(0,5), e11=(1,6), e12=(2,7), e13=(3,8), e14=(4,9)
    
    # Let's try assigning small distinct primes or similar.
    
    # Try a random search with a time limit.
    random.seed(42)
    
    # Precompute all non-backtracking closed walks of length 5-9 in the Petersen graph.
    # For each such walk, we need the sum of voltages to be non-zero mod 32.
    
    # Build adjacency list for Petersen
    p_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        p_adj[u].append(v)
        p_adj[v].append(u)
    
    # Find all non-backtracking closed walks of length 5-9
    # A non-backtracking walk is (v0, v1, ..., vL) where v0=vL and vi != vi-2 for i>=2.
    # For a closed walk, we also need vL != v_{L-2}, i.e., v0 != v_{L-2}.
    
    def find_nonbacktracking_closed_walks(length):
        walks = []
        for start in range(10):
            # DFS with non-backtracking constraint
            stack = [(start, [start], start)]  # (current, path, prev)
            while stack:
                curr, path, prev = stack.pop()
                if len(path) == length + 1:
                    if path[-1] == start:
                        # Check non-backtracking at the end: path[-1] != path[-3]
                        if len(path) >= 3 and path[-1] != path[-3]:
                            walks.append(path)
                    continue
                for nxt in p_adj[curr]:
                    if nxt == prev:
                        continue
                    stack.append((nxt, path + [nxt], curr))
        return walks
    
    # This might be slow for length 9. Let's be more careful.
    # Actually, let's just try random voltage assignments and check the girth directly.
    # The _check_girth_nb_fast function will do the actual verification.
    
    best_result = None
    
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        # Generate a random voltage assignment
        voltages = {}
        for u, v in petersen_edges:
            voltages[(u, v)] = random.randint(1, group_order - 1)
        
        result = _voltage_lift_petersen(voltages, group_order)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return _luw_construct(4, 10)

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k == 4 and g == 10:
        try:
            n, edges = _construct_4_10_cage()
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            for i in range(n):
                if len(adj[i]) != k:
                    raise ValueError(f"Vertex {i} has degree {len(adj[i])}")
            girth = _bfs_girth(adj)
            if girth >= g:
                return n, edges
        except Exception:
            pass
        n, edges = _luw_construct(k, g)
        return n, edges
    else:
        n, edges = _luw_construct(k, g)
        return n, edges
