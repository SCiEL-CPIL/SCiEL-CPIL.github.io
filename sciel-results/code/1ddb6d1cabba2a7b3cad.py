import time
import random
import sys
import numpy as np
from collections import deque

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for k-regular graphs with high girth."""
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

def _girth_ok(adj, n, g):
    """Check if graph has girth >= g using BFS from each vertex."""
    max_depth = (g - 1) // 2
    for v in range(n):
        # BFS with parent tracking to detect backtracking
        visited = {v: -1}  # node -> parent
        queue = deque([(v, -1, 0)])
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Found a cycle: path v->...->node->neighbor and v->...->neighbor
                    # Cycle length <= depth + 1 + depth_of(neighbor)
                    # Since both depths <= max_depth, cycle length <= 2*max_depth + 1 < g
                    return False
                visited[neighbor] = node
                queue.append((neighbor, node, depth + 1))
    return True

def _polish_2switch(adj, n, g, deadline):
    """Try to improve girth or maintain it via 2-switches. Returns True if girth is still OK."""
    # Build edge set for quick lookup
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
    
    # Randomly try 2-switches
    for _ in range(500):
        if time.time() > deadline:
            break
        # Pick two random edges (a,b) and (c,d)
        e1 = random.choice(list(edge_set))
        e2 = random.choice(list(edge_set))
        a, b = e1
        c, d = e2
        # Ensure they share no vertices
        if len({a, b, c, d}) != 4:
            continue
        # Try switching: remove (a,b),(c,d), add (a,c),(b,d)
        new_e1 = (min(a, c), max(a, c))
        new_e2 = (min(b, d), max(b, d))
        if new_e1 in edge_set or new_e2 in edge_set or new_e1 == new_e2:
            continue
        # Apply switch
        edge_set.discard(e1)
        edge_set.discard(e2)
        edge_set.add(new_e1)
        edge_set.add(new_e2)
        # Update adjacency
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        adj[a].append(c)
        adj[c].append(a)
        adj[b].append(d)
        adj[d].append(b)
        # Check girth
        if not _girth_ok(adj, n, g):
            # Revert
            edge_set.discard(new_e1)
            edge_set.discard(new_e2)
            edge_set.add(e1)
            edge_set.add(e2)
            adj[a].remove(c)
            adj[c].remove(a)
            adj[b].remove(d)
            adj[d].remove(b)
            adj[a].append(b)
            adj[b].append(a)
            adj[c].append(d)
            adj[d].append(c)
            continue
    
    return _girth_ok(adj, n, g)

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # First, get the LUW construction as a baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Verify the LUW graph is valid
    adj = [[] for _ in range(n_luw)]
    for u, v in edges_luw:
        adj[u].append(v)
        adj[v].append(u)
    
    if not _girth_ok(adj, n_luw, g):
        # Fallback: return as-is (shouldn't happen)
        return n_luw, edges_luw
    
    best_n = n_luw
    best_edges = list(edges_luw)
    
    # Try to improve via 2-switches (this maintains regularity and can sometimes reduce girth issues)
    # But we want to MINIMIZE n, not maximize girth. So 2-switches won't reduce n.
    # Instead, let's try random restarts with smaller n.
    
    # For k=4, g=10: lower bound is 243, known best is 320
    # Try random bipartite regular graphs for n < best_n
    for n_try in range(244, best_n, 2):
        if time.time() > deadline:
            break
        
        max_attempts = 200
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random bipartite k-regular graph
            half = n_try // 2
            left = list(range(half))
            right = list(range(half, n_try))
            
            left_stubs = []
            for v in left:
                for _ in range(k):
                    left_stubs.append(v)
            right_stubs = []
            for v in right:
                for _ in range(k):
                    right_stubs.append(v)
            
            random.shuffle(right_stubs)
            
            edge_set = set()
            valid = True
            new_edges = []
            for i in range(len(left_stubs)):
                u = left_stubs[i]
                v = right_stubs[i]
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
                new_edges.append((u, v))
            
            if not valid:
                continue
            
            # Check girth
            adj_try = [[] for _ in range(n_try)]
            for u, v in new_edges:
                adj_try[u].append(v)
                adj_try[v].append(u)
            
            if _girth_ok(adj_try, n_try, g):
                best_n = n_try
                best_edges = new_edges
                break
    
    return best_n, best_edges
