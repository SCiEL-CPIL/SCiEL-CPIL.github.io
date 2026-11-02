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
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i + 1) % 5]))
    # Spokes
    for i in range(5):
        edges.append((i, i + 5))
    return edges

def _find_nb_closed_walks(adj, max_len):
    """Find all non-backtracking closed walks of length up to max_len."""
    n = len(adj)
    walks = []
    for start in range(n):
        # (current, path, prev)
        stack = [(start, [start], start)]
        while stack:
            curr, path, prev = stack.pop()
            if len(path) > max_len:
                continue
            if len(path) >= 5 and path[-1] == start:
                # Check non-backtracking at the end
                if path[-1] != path[-3]:
                    # Also need to ensure it's a simple cycle? No, closed walk.
                    # But for girth, we care about cycles.
                    # A closed walk of length L contains a cycle of length <= L.
                    # If we avoid all closed walks of length < 10, we avoid all cycles < 10.
                    walks.append(path)
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                stack.append((nxt, path + [nxt], curr))
    return walks

def _construct_4_10_cage():
    start_time = time.time()
    deadline = start_time + 110
    
    # Petersen graph setup
    petersen_edges = _petersen_edges()
    p_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        p_adj[u].append(v)
        p_adj[v].append(u)
        
    # Map edges to indices for voltage assignment
    edge_index = {}
    edge_list = []
    for i, (u, v) in enumerate(petersen_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        edge_list.append((u, v))
        
    # Find all NB closed walks of length 5-9
    # We need the sequence of edges and their directions
    # Walk: v0 -> v1 -> ... -> vL=v0
    # Edge i is (u, v). If we go u->v, coeff +1. If v->u, coeff -1.
    
    constraints = []
    
    # Generate walks
    # To be efficient, we can just generate them directly
    for length in range(5, 10):
        for start in range(10):
            stack = [(start, [start], start)]
            while stack:
                curr, path, prev = stack.pop()
                if len(path) == length + 1:
                    if path[-1] == start and path[-1] != path[-3]:
                        # Valid NB closed walk
                        # Extract edge coefficients
                        coeffs = [0] * 15
                        valid = True
                        for i in range(length):
                            u = path[i]
                            v = path[i+1]
                            idx = edge_index.get((u, v))
                            if idx is None:
                                valid = False
                                break
                            # Determine direction
                            orig_u, orig_v = edge_list[idx]
                            if u == orig_u and v == orig_v:
                                coeffs[idx] += 1
                            else:
                                coeffs[idx] -= 1
                        if valid:
                            # Constraint: sum(coeffs[i] * voltage[i]) != 0 mod 32
                            constraints.append(coeffs)
                else:
                    for nxt in p_adj[curr]:
                        if nxt == prev:
                            continue
                        stack.append((nxt, path + [nxt], curr))
                        
    if time.time() > deadline:
        return _luw_construct(4, 10)
        
    # Backtracking search
    # Variables: 15 voltages, domain 1..31
    # Constraints: sum(c_i * v_i) != 0 mod 32
    
    voltages = [None] * 15
    group_order = 32
    
    # Sort constraints by number of non-zero coefficients to prune early?
    # Or just use them as is.
    
    def check_constraints(assigned_up_to):
        # Check all constraints that are fully assigned
        for c in constraints:
            s = 0
            complete = True
            for i in range(15):
                if c[i] != 0:
                    if voltages[i] is None:
                        complete = False
                        break
                    s += c[i] * voltages[i]
            if complete:
                if s % group_order == 0:
                    return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == 15:
            return check_constraints(15)
        
        for val in range(1, group_order):
            voltages[idx] = val
            if check_constraints(idx):
                if backtrack(idx + 1):
                    return True
        voltages[idx] = None
        return False

    # Heuristic: Order variables by degree in constraints?
    # For now, just try 0..14
    
    if backtrack(0):
        # Construct the graph
        n = 10 * group_order
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        for i, (u, v) in enumerate(petersen_edges):
            alpha = voltages[i]
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
        
        # Verify
        if _check_girth_nb_fast(adj, 10):
            return n, list(edge_set)
            
    return _luw_construct(4, 10)

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
