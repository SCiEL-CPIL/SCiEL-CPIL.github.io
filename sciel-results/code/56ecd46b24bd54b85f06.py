"""Construct a 4-regular graph with girth >= 10 on as few vertices as possible."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _find_short_cycle(adj, n, max_len):
    """Find a specific short cycle (length <= max_len) using non-backtracking BFS.
    Returns a list of vertices forming the cycle, or None."""
    for start in range(n):
        # Non-backtracking BFS
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start))
            visited[(v, start)] = (1, [start, v])
        
        while queue:
            u, prev = queue.popleft()
            depth, path = visited[(u, prev)]
            if depth >= max_len:
                continue
            for w in adj[u]:
                if w == prev:
                    continue
                if w == start and depth + 1 >= 3:
                    # Found a cycle
                    if depth + 1 <= max_len:
                        return path + [start]
                elif (w, u) not in visited:
                    visited[(w, u)] = (depth + 1, path + [w])
                    queue.append((w, u))
    return None

def _count_short_cycles_fast(adj, n, limit):
    """Count the number of short cycles (length < limit) using non-backtracking BFS.
    This is faster than finding specific cycles when we just need a count."""
    max_depth = (limit - 1) // 2
    count = 0
    for start in range(n):
        # Non-backtracking BFS from start
        # State: (current_vertex, previous_vertex) -> depth
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start))
            visited[(v, start)] = 1
        
        while queue:
            u, prev = queue.popleft()
            depth = visited[(u, prev)]
            if depth >= max_depth:
                continue
            for w in adj[u]:
                if w == prev:
                    continue
                if w == start:
                    # Found a cycle of length depth + 1
                    if depth + 1 < limit:
                        count += 1
                elif (w, u) not in visited:
                    visited[(w, u)] = depth + 1
                    queue.append((w, u))
    return count

def _random_k_regular(n, k, rng):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        raise ValueError("n*k must be even")
    
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    rng.shuffle(half_edges)
    
    adj = [set() for _ in range(n)]
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    
    # Fix multi-edges/self-loops
    for _ in range(200):
        bad = False
        for u in range(n):
            if len(adj[u]) != k:
                bad = True
                break
        if not bad:
            break
        
        # Find a vertex with wrong degree and fix
        for u in range(n):
            if len(adj[u]) < k:
                candidates = [v for v in range(n) if v != u and v not in adj[u] and len(adj[v]) < k]
                if candidates:
                    v = rng.choice(candidates)
                    adj[u].add(v)
                    adj[v].add(u)
            elif len(adj[u]) > k:
                v = rng.choice(list(adj[u]))
                adj[u].discard(v)
                adj[v].discard(u)
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    for i in range(n):
        if len(adj_list[i]) != k:
            raise ValueError(f"Vertex {i} has degree {len(adj_list[i])} instead of {k}")
    return adj_list

def _apply_two_switch_fast(adj, n, k, edge_to_swap, rng):
    """Apply a 2-switch involving a specific edge (u, v) from adj.
    Returns new_adj if successful, else None."""
    u, v = edge_to_swap
    if v not in adj[u]:
        return None
        
    # Collect all edges not incident to u or v
    valid_edges = []
    for i in range(n):
        if i == u or i == v:
            continue
        for j in adj[i]:
            if i < j and j != u and j != v:
                # Check if swap is valid
                if j not in adj[u] and i not in adj[v]:
                    valid_edges.append((i, j))
    
    if not valid_edges:
        return None
        
    x, y = rng.choice(valid_edges)
    
    # Apply swap
    new_adj = [list(s) for s in adj]
    
    # Remove (u, v)
    new_adj[u].remove(v)
    new_adj[v].remove(u)
    
    # Remove (x, y)
    new_adj[x].remove(y)
    new_adj[y].remove(x)
    
    # Add (u, x)
    new_adj[u].append(x)
    new_adj[x].append(u)
    
    # Add (v, y)
    new_adj[v].append(y)
    new_adj[y].append(v)
    
    # Sort and verify
    for i in range(n):
        new_adj[i] = sorted(new_adj[i])
        if len(new_adj[i]) != k:
            return None
            
    return new_adj

def _improve_girth(adj, n, k, target_girth, rng, deadline):
    """Try to increase girth to target_girth using targeted 2-switches."""
    current_adj = [list(s) for s in adj]
    consecutive_failures = 0
    
    while time.time() < deadline:
        # Find a short cycle
        cycle = _find_short_cycle(current_adj, n, target_girth - 1)
        if not cycle:
            return current_adj  # Success
        
        # Pick a random edge from the cycle
        L = len(cycle) - 1
        edge_idx = rng.randint(0, L - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % L]
        
        # Try to apply a 2-switch on this edge
        accepted = False
        for _ in range(20):
            new_adj = _apply_two_switch_fast(current_adj, n, k, (u, v), rng)
            if not new_adj:
                continue
            # Check girth
            if _find_short_cycle(new_adj, n, target_girth - 1) is None:
                current_adj = new_adj
                accepted = True
                consecutive_failures = 0
                break
        
        if not accepted:
            consecutive_failures += 1
            if consecutive_failures > 50:
                break  # Give up on this initialization
        
    return current_adj

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 110
    
    # Moore bound
    if g % 2 == 0:
        moore = 1 + k * (k - 1) ** (g // 2 - 1)
    else:
        moore = 2 + k * (k - 1) ** ((g - 1) // 2 - 1) if (g-1)//2 - 1 >= 0 else 2 + k
    
    rng = random.Random(42)
    
    best_n = None
    best_edges = None
    
    # Try sizes from known best downwards
    start_sizes = [320, 300, 280, 260, 250, 245, 244]
    start_sizes = [s for s in start_sizes if s >= moore and s * k % 2 == 0]
    
    for n_try in start_sizes:
        if time.time() > deadline - 20:
            break
            
        # Try multiple random initializations
        for init in range(15):
            if time.time() > deadline - 10:
                break
                
            try:
                adj = _random_k_regular(n_try, k, rng)
            except Exception:
                continue
            
            # Check if already good
            if _find_short_cycle(adj, n_try, g - 1) is None:
                edges = []
                for u in range(n_try):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges
                break  # Found valid graph for this n_try
            
            # Try to improve
            improved_adj = _improve_girth(adj, n_try, k, g, rng, deadline - 5)
            
            if _find_short_cycle(improved_adj, n_try, g - 1) is None:
                edges = []
                for u in range(n_try):
                    for v in improved_adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges
                break  # Found valid graph for this n_try
    
    if best_n is not None and best_edges is not None:
        return best_n, best_edges
        
    # Fallback: LUW construction
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

# EVOLVE-BLOCK-END
